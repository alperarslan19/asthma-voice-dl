"""
Faz 1 — Harmonize ses önbelleği (D-008, D-015, D-018, D-021, D-030)

BİLİMSEL AMAÇ
    Her kaydı BİR KEZ, deterministik olarak, bütün modellerin aynı biçimde göreceği hâle getirmek:
      çöz → (44.1 kHz ise 48 kHz'e) → kenar sessizliği sınırlarını bul → 32 / 16 kHz'e indir
      → 32 kHz yolunda 11.0 kHz alçak geçiren → sınırlardan kes → tepe normalizasyonu (−1 dBFS).
    Böylece (1) ön işleme deney boyunca sabit kalır, (2) iki kodlama zincirinin (Apple / FFmpeg) bant
    genişliği farkı girdiden silinir (D-021), (3) kayıt başındaki/sonundaki sessizliğin uzunluğu
    (sesle ilgisiz bir ipucu) girdiden çıkar (D-015). Ses "temizlenmez": gürültü giderme YOK.

VERİNİN YAPISI
    <audio_root>/<relpath>       .m4a dosyaları (soundData.zip'ten açılmış)
    recording_map.csv            katılımcı × slot; status (OK/MISSING/...), relpath, file_sha256, sr, channels,
                                 tag_encoder (zincir) — scripts/audit_audio.py üretir

ÇIKTILAR (<out_dir>, Drive data_derived/audio_cache_v1/; katılımcı düzeyi → git'e GİRMEZ)
    audio_32k.f32, audio_16k.f32   tüm kayıtlar uç uca, little-endian float32 (np.memmap ile okunur)
    index.csv                      kayıt başına: participant_id, slot, offset_32k, n_32k, offset_16k, n_16k,
                                   kırpma sınırları, uygulanan kazanç, bant genişliği (önce / sonra) ...
    cache_info.json                parametreler, sürümler, sha256'lar
    <report_dir>/audio_cache_report.{json,md}   yalnız agrega (git'e girebilir)

OKUMA (sonraki script'lerde)
    idx = pd.read_csv("index.csv"); a = np.memmap("audio_32k.f32", dtype="<f4", mode="r")
    r = idx.iloc[i]; x = np.asarray(a[r.offset_32k : r.offset_32k + r.n_32k])

ÇALIŞTIRMA
    python scripts/build_audio_cache.py --audio-root /content/audio --recording-map .../recording_map.csv \
        --out-dir /content/audio_cache_v1 --report-dir reports/audio_cache
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.signal import firwin, oaconvolve, resample_poly
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_audio import decode, effective_bandwidth_khz, frame_db  # noqa: E402  (denetimle AYNI tanımlar)

SOURCE_SR = 48000
TARGETS = {32000: {"up": 2, "down": 3, "lowpass_hz": 11000.0},   # D-008, D-021 (kesim: EXP-003 Bölüm A)
           16000: {"up": 1, "down": 3, "lowpass_hz": None}}      # resample_poly'nin kendi filtresi 8 kHz'te keser
FIR_TAPS, FIR_BETA = 511, 8.6        # −6 dB @ 11.0 kHz; 10.8 kHz'e kadar düz; 11.27 kHz'te ~−94 dB
SILENCE_DB, SNR_GATE_DB, FLOOR_MARGIN_DB = 35.0, 20.0, 10.0   # audit_audio.py ile aynı aktif-kare kuralı
FRAME_WIN_S, FRAME_HOP_S = 0.025, 0.010
TRIM_MARGIN_S = 0.10                 # ilk/son aktif karenin dışında bırakılan pay
PEAK_TARGET_DBFS = -1.0
PEAK_TARGET = float(10 ** (PEAK_TARGET_DBFS / 20))
N_DETERMINISM_CHECK = 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def to_source_sr(x: np.ndarray, sr: int) -> np.ndarray:
    """Ortak kaynak SR = 48 kHz (D-021). Yalnız bilinen oranlar kabul edilir."""
    if sr == SOURCE_SR:
        return x
    if sr == 44100:
        return resample_poly(x, 160, 147).astype(np.float32)
    raise ValueError(f"beklenmeyen örnekleme hızı: {sr}")


def trim_bounds(x: np.ndarray, sr: int) -> dict:
    """Kenar sessizliği sınırları (saniye). Kural ses denetimiyle aynı: eşik = en yüksek kare − 35 dB;
    belirgin bir gürültü tabanı varsa (p95 − p10 > 20 dB) en az taban + 10 dB. İç sessizliklere dokunulmaz."""
    db = frame_db(x, sr, FRAME_WIN_S, FRAME_HOP_S)
    floor, speech = float(np.percentile(db, 10)), float(np.percentile(db, 95))
    thr = db.max() - SILENCE_DB
    if speech - floor > SNR_GATE_DB:
        thr = max(thr, floor + FLOOR_MARGIN_DB)
    idx = np.flatnonzero(db > thr)
    assert len(idx), "aktif kare yok (olamaz: en yüksek kare her zaman eşiğin üstünde)"
    dur = len(x) / sr
    start = max(0.0, idx[0] * FRAME_HOP_S - TRIM_MARGIN_S)
    end = min(dur, idx[-1] * FRAME_HOP_S + FRAME_WIN_S + TRIM_MARGIN_S)
    assert 0 <= start < end <= dur
    return {"trim_start_s": start, "trim_end_s": end, "src_duration_s": dur,
            "lead_trimmed_s": start, "trail_trimmed_s": dur - end, "snr_proxy_db": speech - floor}


_FIR_CACHE: dict = {}


def lowpass(y: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    key = (sr, cutoff)
    if key not in _FIR_CACHE:
        _FIR_CACHE[key] = firwin(FIR_TAPS, cutoff, fs=sr, window=("kaiser", FIR_BETA))
    # Simetrik FIR + 'same' → doğrusal faz, sıfır gecikme
    return oaconvolve(y, _FIR_CACHE[key], mode="same")


def render(x48: np.ndarray, bounds: dict, sr: int) -> tuple[np.ndarray, float]:
    """Tek bir hedef SR sürümü. Resample ve filtre TÜM sinyalde (kenar geçişi olmasın), sonra kesim."""
    cfg = TARGETS[sr]
    y = resample_poly(x48.astype(np.float64), cfg["up"], cfg["down"])
    if cfg["lowpass_hz"]:
        y = lowpass(y, sr, cfg["lowpass_hz"])
    a, b = int(round(bounds["trim_start_s"] * sr)), int(round(bounds["trim_end_s"] * sr))
    y = y[a:b]
    peak = float(np.abs(y).max())
    assert peak > 0, "sessiz kayıt"
    gain = PEAK_TARGET / peak
    y = (y * gain).astype(np.float32)
    assert np.isfinite(y).all()
    return y, float(20 * np.log10(gain))


def process_file(path: Path, sr: int, channels: int) -> tuple[dict, dict]:
    X = decode(path, sr, channels)                     # (n, ch), native SR, float32 (denetimle aynı)
    x = X.mean(axis=1).astype(np.float32)
    assert x.ndim == 1 and len(x) > 0
    x48 = to_source_sr(x, sr)
    b = trim_bounds(x48, SOURCE_SR)
    b["bw_before_khz"] = effective_bandwidth_khz(x48, SOURCE_SR)
    out = {}
    for tsr in TARGETS:
        y, g = render(x48, b, tsr)
        out[tsr] = y
        b[f"gain_db_{tsr // 1000}k"] = g
    b["bw_after_32k_khz"] = effective_bandwidth_khz(out[32000], 32000)
    return b, out


def chain_of(encoder: pd.Series) -> pd.Series:
    return pd.Series(np.where(encoder.astype(str).str.startswith("Lavf"), "ffmpeg", "apple"), index=encoder.index)


def tool_version(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True).stdout.splitlines()[0]
    except Exception:
        return "unknown"


QS = {"min": 0, "p1": 0.01, "p5": 0.05, "medyan": 0.5, "p95": 0.95, "p99": 0.99, "max": 1}


def q(s: pd.Series) -> dict:
    return {name: round(float(s.quantile(v)), 3) for name, v in QS.items()}


def fmt(d: dict) -> str:
    return " · ".join(f"{k} {v}" for k, v in d.items())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-root", required=True, type=Path)
    ap.add_argument("--recording-map", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--report-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    rm = pd.read_csv(args.recording_map)
    need = {"participant_id", "slot", "status", "relpath", "file_sha256", "sr", "channels", "tag_encoder"}
    assert need <= set(rm.columns), f"recording_map.csv eksik sütun: {need - set(rm.columns)}"
    ok = rm[rm["status"] == "OK"].sort_values(["participant_id", "slot"]).reset_index(drop=True)
    assert not ok.duplicated(["participant_id", "slot"]).any(), "aynı (katılımcı, slot) iki kez OK"
    ok["chain"] = chain_of(ok["tag_encoder"])
    print(f"İşlenecek: {len(ok)} kayıt, {ok.participant_id.nunique()} katılımcı; SR {ok.sr.value_counts().to_dict()}")

    # 1) Veri sürümü: her dosya denetimdeki dosyayla birebir aynı mı?
    bad = []
    for r in ok.itertuples():
        p = args.audio_root / r.relpath
        if not p.exists() or sha256_file(p) != r.file_sha256:
            bad.append(r.relpath)
    assert not bad, f"{len(bad)} dosya eksik ya da sha256 farklı (veri sürümü değişmiş olabilir), ör. {bad[:3]}"
    print("sha256 kontrolü: tüm dosyalar denetimdekiyle aynı.")

    # 2) İşle ve diske uç uca yaz
    files = {sr: open(args.out_dir / f"audio_{sr // 1000}k.f32", "wb") for sr in TARGETS}
    offsets = {sr: 0 for sr in TARGETS}
    rows = []
    try:
        for i, r in enumerate(ok.itertuples()):
            b, out = process_file(args.audio_root / r.relpath, int(r.sr), int(r.channels))
            row = {"participant_id": r.participant_id, "slot": r.slot, "relpath": r.relpath,
                   "src_sha256": r.file_sha256, "chain": r.chain, "src_sr": int(r.sr), **b}
            for sr, y in out.items():
                files[sr].write(y.astype("<f4").tobytes())
                row[f"offset_{sr // 1000}k"], row[f"n_{sr // 1000}k"] = offsets[sr], len(y)
                offsets[sr] += len(y)
            rows.append(row)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(ok)}  ({time.time() - t0:.0f} s)", flush=True)
    finally:
        for f in files.values():
            f.close()
    idx = pd.DataFrame(rows)
    idx.to_csv(args.out_dir / "index.csv", index=False, lineterminator="\n")

    # 3) Doğrulama
    mm = {sr: np.memmap(args.out_dir / f"audio_{sr // 1000}k.f32", dtype="<f4", mode="r") for sr in TARGETS}
    for sr in TARGETS:
        k = f"{sr // 1000}k"
        assert len(mm[sr]) == offsets[sr] == idx[f"n_{k}"].sum(), f"{k}: dosya uzunluğu dizinle uyuşmuyor"
        assert (idx[f"offset_{k}"].diff().fillna(0).iloc[1:] == idx[f"n_{k}"].iloc[:-1].values).all(), f"{k}: ofsetler bitişik değil"
        assert np.isfinite(mm[sr]).all(), f"{k}: NaN/Inf var"
        peaks = np.array([np.abs(mm[sr][o:o + n]).max() for o, n in zip(idx[f"offset_{k}"], idx[f"n_{k}"])])
        assert np.allclose(peaks, PEAK_TARGET, atol=1e-6), f"{k}: tepe değeri hedefte değil"
        ratio = idx[f"n_{k}"] / (idx["trim_end_s"] - idx["trim_start_s"]) / sr
        assert ratio.between(0.999, 1.001).all(), f"{k}: örnek sayısı süreyle uyuşmuyor"
    for i, r in enumerate(ok.head(N_DETERMINISM_CHECK).itertuples()):  # belirlenimcilik
        _, out = process_file(args.audio_root / r.relpath, int(r.sr), int(r.channels))
        for sr, y in out.items():
            k = f"{sr // 1000}k"
            o, n = int(idx.loc[i, f"offset_{k}"]), int(idx.loc[i, f"n_{k}"])
            assert np.array_equal(np.asarray(mm[sr][o:o + n]), y), f"belirlenimci değil: kayıt {i}, {k}"

    # 4) Agrega rapor (katılımcı ID'si yok)
    kept = idx["trim_end_s"] - idx["trim_start_s"]
    bw_auc = {}
    for col in ["bw_before_khz", "bw_after_32k_khz"]:
        d = idx.dropna(subset=[col])
        bw_auc[col] = round(float(roc_auc_score(d["chain"] == "ffmpeg", d[col])), 3)
    info = {
        "decisions": ["D-008", "D-015", "D-018", "D-021", "D-030"],
        "params": {"source_sr": SOURCE_SR, "targets": {str(k): v for k, v in TARGETS.items()},
                   "fir": {"taps": FIR_TAPS, "kaiser_beta": FIR_BETA},
                   "trim": {"silence_db": SILENCE_DB, "snr_gate_db": SNR_GATE_DB, "floor_margin_db": FLOOR_MARGIN_DB,
                            "frame_win_s": FRAME_WIN_S, "frame_hop_s": FRAME_HOP_S, "margin_s": TRIM_MARGIN_S},
                   "peak_target_dbfs": PEAK_TARGET_DBFS, "dtype": "<f4 (little-endian float32)"},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                     "pandas": pd.__version__, "ffmpeg": tool_version(["ffmpeg", "-version"])},
        "inputs": {"recording_map_sha256": sha256_file(args.recording_map), "n_recordings": int(len(idx)),
                   "n_participants": int(idx.participant_id.nunique()),
                   "per_slot": {int(k): int(v) for k, v in idx.slot.value_counts().sort_index().items()},
                   "src_sr": {int(k): int(v) for k, v in idx.src_sr.value_counts().items()},
                   "chain": {k: int(v) for k, v in idx.chain.value_counts().items()}},
        "files_sha256": {p.name: sha256_file(p) for p in sorted(args.out_dir.glob("*")) if p.suffix in {".f32", ".csv"}},
        "sizes_gb": {f"{sr // 1000}k": round(offsets[sr] * 4 / 1e9, 3) for sr in TARGETS},
        "checks": {"finite": True, "peak_equals_target": True, "offsets_contiguous": True,
                   "deterministic_first_n": N_DETERMINISM_CHECK, "source_sha256_match": True},
        "runtime_s": round(time.time() - t0, 1),
    }
    report = {
        **{k: info[k] for k in ["decisions", "params", "versions", "inputs", "files_sha256", "sizes_gb", "checks", "runtime_s"]},
        "trim": {"lead_trimmed_s": q(idx["lead_trimmed_s"]), "trail_trimmed_s": q(idx["trail_trimmed_s"]),
                 "kept_duration_s": q(kept), "n_kept_lt_4s": int((kept < 4).sum()), "n_kept_lt_2s": int((kept < 2).sum()),
                 "share_low_snr_peak_rule_only": round(float((idx["snr_proxy_db"] <= SNR_GATE_DB).mean()), 3)},
        "gain_db": {f"{sr // 1000}k": q(idx[f"gain_db_{sr // 1000}k"]) for sr in TARGETS},
        "bandwidth_khz_by_chain": {col: {c: q(g[col].dropna()) for c, g in idx.groupby("chain")}
                                   for col in ["bw_before_khz", "bw_after_32k_khz"]},
        "chain_auc_from_bandwidth": bw_auc,
    }
    (args.out_dir / "cache_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    (args.report_dir / "audio_cache_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    T = report["trim"]
    L = ["# Ses önbelleği raporu (otomatik üretildi; katılımcı ID'si içermez)", "",
         f"- Kayıt: {info['inputs']['n_recordings']}, katılımcı: {info['inputs']['n_participants']}; slot başına {info['inputs']['per_slot']}",
         f"- Kaynak SR: {info['inputs']['src_sr']}; zincir: {info['inputs']['chain']}",
         f"- Kontroller: kaynak sha256 = denetim ✓, NaN/Inf yok ✓, tepe = {PEAK_TARGET_DBFS} dBFS ✓, ofsetler bitişik ✓, "
         f"ilk {N_DETERMINISM_CHECK} kayıt yeniden işlendiğinde birebir aynı ✓",
         f"- Boyut: {info['sizes_gb']} GB; süre {info['runtime_s']} s; sürümler {info['versions']}", "",
         "## Kenar kırpma (D-015)", "",
         f"- Baştan kırpılan (s): {fmt(T['lead_trimmed_s'])}",
         f"- Sondan kırpılan (s): {fmt(T['trail_trimmed_s'])}",
         f"- Kalan süre (s): {fmt(T['kept_duration_s'])}",
         f"- Kalan süresi 4 s'den kısa: {T['n_kept_lt_4s']} kayıt; 2 s'den kısa: {T['n_kept_lt_2s']} kayıt (D-015: eğitimde sıfırla doldurulur)",
         f"- Düşük SNR nedeniyle yalnız tepeye-göre eşik kullanılan kayıt payı: {T['share_low_snr_peak_rule_only']}", "",
         "## Normalizasyon kazancı (dB)", ""] + [f"- {k}: {fmt(v)}" for k, v in report["gain_db"].items()] + ["",
         "## Bant genişliği (D-021) — denetimle aynı ölçüt", "",
         "| zincir | önce (48 kHz) medyan / max | sonra (32 kHz yolu) medyan / max |", "|---|---|---|"]
    for c in sorted(idx.chain.unique()):
        bb, ba = report["bandwidth_khz_by_chain"]["bw_before_khz"][c], report["bandwidth_khz_by_chain"]["bw_after_32k_khz"][c]
        L.append(f"| {c} | {bb['medyan']} / {bb['max']} | {ba['medyan']} / {ba['max']} |")
    L += ["", f"Bant genişliğinden zincir ayrımı (AUC; 0.5 = ayrım yok): önce {bw_auc['bw_before_khz']}, "
              f"sonra {bw_auc['bw_after_32k_khz']}.",
          "Not: Filtre 11 kHz'in üstünü herkes için aynı biçimde siler; \"sonra\" sütununda bütün değerler ~11.0–11.3 kHz "
          "aralığında olmalı. Sonraki AUC 0.5'ten farklıysa bunun nedeni, kodlayıcıların 11 kHz'in ALTINDAKİ spektral "
          "şekillendirmesidir; filtre onu silemez (D-021 riski). Gömme düzeyinde zincir probu değerlendirme aşamasında (D-028).", "",
          "## Dosya sha256'ları", ""] + [f"- `{k}`: `{v}`" for k, v in info["files_sha256"].items()]
    (args.report_dir / "audio_cache_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
