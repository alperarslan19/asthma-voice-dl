"""
Faz 0 / Adım 2-3-4 — Ses verisi denetimi + katılımcı↔kayıt↔görev eşlemesi + sızıntı/confounder kontrolleri

BİLİMSEL AMAÇ
    Model eğitiminden ÖNCE şu soruları sayılarla cevaplamak:
      1. soundData'da gerçekte ne var? (kapsayıcı, codec, bitrate, SR, kanal, süre)   -> ENVANTER
      2. Her katılımcının her görevi (slot 1..7) için tam olarak bir dosya var mı?    -> EŞLEME
      3. Aynı ses dosyası iki farklı katılımcıda var mı?                              -> DUPLICATE LEAKAGE
      4. Kayıt koşulları ve dosya meta verisi (kodlayıcı, cihaz, yazılım sürümü,
         saat, gürültü tabanı, sessizlik…) etikete ya da kayıt dönemine göre
         farklılaşıyor mu?                                                            -> ZAMANSAL CONFOUNDER'IN İZİ

ADLANDIRMA (veri ekibi, düzeltilmiş bildirim 2026-10-08) [FROM DATA TEAM]
    <katılımcı ID>_<slot>.m4a   ör. 101001_2.m4a
    slot: 1=aaa (sürdürülmüş ünlü), 2=araba, 3=ana, 4=ordu, 5=gelecek, 6=titiz, 7=ünlem
    Son dosya 101344_7.m4a -> beklenen 344 × 7 = 2408 dosya; 48 kHz mono.
    Script yine de her uzantıyı kabul eder ve uzantı/codec tekdüzeliğini raporlar (beklentiyi doğrular).
    Dosyaların yarısından fazlası bu kalıba uymazsa EŞLEME YAPILMAZ: script envanteri
    yazar ve çıkış kodu 2 ile durur. Tahmin yürütmez.

GİZLİLİK
    iPhone dosyaları GPS konumu içerebilir. Script konum etiketinin yalnızca VAR/YOK
    bilgisini kaydeder; değerini hiçbir çıktıya yazmaz.

ÇIKTILAR
    <out_private>/  (katılımcı düzeyinde -> Drive'da kalır, git'e GİRMEZ)
        audio_inventory.csv      bulunan HER ses dosyası
        recording_map.csv        katılımcı × slot -> dosya (ANA EŞLEME TABLOSU)
        near_duplicate_pairs.csv dinleyerek kontrol edilecek şüpheli çiftler
    <out_report>/   (yalnızca agrega -> git'e girebilir)
        audio_audit.md, audio_audit.json

GEREKSİNİM: ffmpeg + ffprobe (Colab'da kurulu), numpy, pandas, scikit-learn
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

AUDIO_EXT = {".wav", ".m4a", ".mp4", ".mp3", ".aac", ".flac", ".ogg", ".opus", ".webm", ".caf", ".3gp", ".mov", ".aiff", ".aif"}
RE_NAME = re.compile(r"^(?P<pid>\d{6})_(?P<slot>\d{1,2})$")
RE_PID_IN_PATH = re.compile(r"(?<!\d)(\d{6})(?!\d)")
N_SLOTS = 7
LOCAL_TZ = "Europe/Istanbul"
NUISANCE_NUMERIC = ["duration_s", "bandwidth_khz", "lead_silence_s", "trail_silence_s", "noise_floor_db", "speech_level_db",
                    "snr_proxy_db", "rms_dbfs", "peak", "clip_frac", "active_frac", "audio_bit_rate", "creation_hour_local"]
META_CATEGORICAL = ["ext", "container", "codec", "codec_profile", "sr", "channels", "n_video_streams",
                    "tag_major_brand", "tag_encoder", "tag_apple_make", "tag_apple_model", "tag_apple_software",
                    "audio_handler", "has_location_tag"]


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ffprobe(path: Path) -> dict:
    """Kapsayıcı + ilk ses akışı + meta veri etiketleri. Konum etiketinin DEĞERİ okunmaz."""
    cmd = ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return {"probe_error": r.stderr.strip()[:200]}
    j = json.loads(r.stdout or "{}")
    streams = j.get("streams", [])
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    if not audio:
        return {"probe_error": "ses akışı yok", "n_video_streams": sum(s.get("codec_type") == "video" for s in streams)}
    a, f = audio[0], j.get("format", {})
    ftags = {k.lower(): v for k, v in (f.get("tags") or {}).items()}
    atags = {k.lower(): v for k, v in (a.get("tags") or {}).items()}
    out = {
        "container": f.get("format_name"), "codec": a.get("codec_name"), "codec_profile": a.get("profile"),
        "sr": int(a["sample_rate"]) if a.get("sample_rate") else None, "channels": a.get("channels"),
        "audio_bit_rate": int(a["bit_rate"]) if a.get("bit_rate") else None,
        "n_audio_streams": len(audio), "n_video_streams": sum(s.get("codec_type") == "video" for s in streams),
        "probe_duration_s": float(f["duration"]) if f.get("duration") else None,
        "tag_creation_time": ftags.get("creation_time") or atags.get("creation_time"),
        "tag_major_brand": ftags.get("major_brand"), "tag_encoder": ftags.get("encoder") or atags.get("encoder"),
        "tag_apple_make": ftags.get("com.apple.quicktime.make"),
        "tag_apple_model": ftags.get("com.apple.quicktime.model"),
        "tag_apple_software": ftags.get("com.apple.quicktime.software"),
        "audio_handler": atags.get("handler_name"),
        "has_location_tag": any("location" in k for k in list(ftags) + list(atags)),
    }
    ct = out["tag_creation_time"]
    if ct:
        try:
            ts = pd.Timestamp(ct)
            ts = (ts.tz_localize("UTC") if ts.tzinfo is None else ts).tz_convert(LOCAL_TZ)
            out["creation_local"] = ts.isoformat()
            out["creation_date_local"] = str(ts.date())
            out["creation_hour_local"] = ts.hour + ts.minute / 60
        except (ValueError, TypeError):
            out["creation_parse_error"] = str(ct)[:40]
    return out


def decode(path: Path, sr: int, channels: int) -> np.ndarray:
    """İlk ses akışını ORİJİNAL SR ve kanal sayısında float32 PCM'e çöz. Şekil: (n_samples, channels).
    Video akışı varsa yok sayılır. Burada resample YAPILMAZ (denetim ham veri üzerinde)."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-vn", "-f", "f32le",
           "-acodec", "pcm_f32le", "-ac", str(channels), "-ar", str(sr), "-"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode(errors="ignore")[:200])
    x = np.frombuffer(r.stdout, dtype=np.float32)
    assert x.size % channels == 0, "Çözülen örnek sayısı kanal sayısına bölünmüyor"
    return x.reshape(-1, channels)


def frame_db(x: np.ndarray, sr: int, win_s=0.025, hop_s=0.010) -> np.ndarray:
    win, hop = int(round(win_s * sr)), int(round(hop_s * sr))
    if len(x) < win:
        return np.array([20 * np.log10(np.sqrt(np.mean(x**2)) + 1e-12)])
    frames = np.lib.stride_tricks.sliding_window_view(x, win)[::hop]
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    return 20 * np.log10(rms + 1e-12)


def effective_bandwidth_khz(x: np.ndarray, sr: int, n_fft: int = 4096, drop_db: float = 40.0) -> float:
    """Etkin bant genişliği (sezgisel): uzun-dönem güç spektrumunun, 2–8 kHz medyanının drop_db altına
    son kez düştüğü frekans. Kayıplı kodlayıcının (AAC) alçak geçiren kesimini yakalar. Mutlak değerden
    çok DOSYALAR / DÖNEMLER ARASI DEĞİŞİMİ görmek için kullanılır."""
    if len(x) < n_fft:
        return float("nan")
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[:: n_fft // 2] * np.hanning(n_fft)
    psd = (np.abs(np.fft.rfft(frames, axis=1)) ** 2).mean(axis=0)
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    psd_db = 10 * np.log10(psd + 1e-20)
    band = (freqs >= 2000) & (freqs <= min(8000, sr / 2))
    if not band.any():
        return float("nan")
    above = freqs[psd_db > np.median(psd_db[band]) - drop_db]
    return float(above.max() / 1000) if len(above) else float("nan")


def fingerprint(x: np.ndarray, sr: int, n_bands=64, n_time=200) -> np.ndarray:
    """Zaman yapısını koruyan spektrogram özeti (yakın-kopya tespiti için; model özelliği DEĞİL).
    Çift merkezleme (bant ortalaması + kare ortalaması çıkarılır): farklı kayıtlarda da benzer olan
    kaba yapı (spektral eğim, ses yüksekliği zarfı) atılır, kayda özgü ince desen kalır.
    Sentetik testte: birebir kopya 1.00, yeniden kodlanmış kopya ~0.95, farklı kayıtlar <= 0.71."""
    n_fft = int(2 ** np.ceil(np.log2(0.032 * sr)))
    hop = n_fft // 2
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop] * np.hanning(n_fft)
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    edges = np.geomspace(100, min(8000, 0.99 * sr / 2), n_bands + 1)
    bands = np.stack([spec[:, (freqs >= lo) & (freqs < hi)].sum(1) for lo, hi in zip(edges[:-1], edges[1:])], 1)
    bands = np.log10(bands + 1e-10)
    if len(bands) < n_time:  # çok kısa kayıt: kareleri tekrarla ki şekil sabit kalsın
        bands = np.repeat(bands, int(np.ceil(n_time / len(bands))), axis=0)
    fp = np.stack([c.mean(0) for c in np.array_split(bands, n_time, axis=0)])
    fp = fp - fp.mean(axis=0, keepdims=True)
    fp = fp - fp.mean(axis=1, keepdims=True)
    fp = fp.ravel()
    assert fp.shape == (n_time * n_bands,)
    return ((fp - fp.mean()) / (fp.std() + 1e-8)).astype(np.float32)


def analyze_file(root: Path, path: Path, silence_db: float) -> tuple[dict, np.ndarray | None]:
    rel = path.relative_to(root)
    row = {"relpath": str(rel), "ext": path.suffix.lower(), "size_bytes": path.stat().st_size,
           "file_sha256": sha256_file(path)}
    m = RE_NAME.match(path.stem)
    row["participant_id"] = int(m.group("pid")) if m else None
    row["slot"] = int(m.group("slot")) if m else None
    ids_in_dirs = {i for part in rel.parts[:-1] for i in RE_PID_IN_PATH.findall(part)}
    row["pid_dir_conflict"] = bool(m and ids_in_dirs and ids_in_dirs != {m.group("pid")})
    row.update(ffprobe(path))
    fp = None
    if row.get("sr") and row.get("channels"):
        try:
            X = decode(path, row["sr"], row["channels"])
            x = X.mean(axis=1)
            row["duration_s"] = len(x) / row["sr"]
            row["max_interchannel_absdiff"] = float(np.abs(X - X[:, :1]).max()) if X.shape[1] > 1 else 0.0
            row["peak"] = float(np.abs(x).max()) if len(x) else np.nan
            row["clip_frac"] = float(np.mean(np.abs(X) >= 0.999)) if len(x) else np.nan
            row["rms_dbfs"] = float(20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12))
            db = frame_db(x, row["sr"])
            row["noise_floor_db"] = float(np.percentile(db, 10))
            row["speech_level_db"] = float(np.percentile(db, 95))
            row["snr_proxy_db"] = row["speech_level_db"] - row["noise_floor_db"]
            # Aktif kare eşiği: tepenin silence_db altı; belirgin bir gürültü tabanı varsa (p95-p10 > 20 dB)
            # en az 'taban + 10 dB'. Kesintisiz fonasyonda yalnız tepeye-göre eşik kullanılır.
            thr = db.max() - silence_db
            if row["snr_proxy_db"] > 20:
                thr = max(thr, row["noise_floor_db"] + 10)
            active = db > thr
            row["active_frac"] = float(active.mean())
            idx = np.flatnonzero(active)
            row["lead_silence_s"] = float(idx[0] * 0.010) if len(idx) else np.nan
            row["trail_silence_s"] = float((len(db) - 1 - idx[-1]) * 0.010) if len(idx) else np.nan
            row["pcm_md5"] = hashlib.md5(np.round(np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()).hexdigest()
            row["bandwidth_khz"] = effective_bandwidth_khz(x, row["sr"])
            fp = fingerprint(x, row["sr"])
        except Exception as e:  # hatayı yutma: kaydet ve raporla
            row["decode_error"] = str(e)[:200]
    return row, fp


def table_md(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)  # 'tabulate' gerekir
    except ImportError:
        return "```\n" + df.to_string(index=False) + "\n```"


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-root", required=True, type=Path, help="soundData klasörü (açılmış)")
    ap.add_argument("--participants", required=True, type=Path, help="audit_clinical.py çıktısı participants.csv")
    ap.add_argument("--out-private", required=True, type=Path)
    ap.add_argument("--out-report", required=True, type=Path)
    ap.add_argument("--silence-db", type=float, default=35.0, help="tepe kare enerjisinin kaç dB altı 'sessizlik'")
    ap.add_argument("--near-dup-threshold", type=float, default=0.90,
                    help="sentetik testte birebir kopya 1.00, yeniden kodlanmış ~0.95, farklı kayıtlar <=0.71; gerçek veride kalibre edilmeli")
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 2)))
    args = ap.parse_args()
    args.out_private.mkdir(parents=True, exist_ok=True)
    args.out_report.mkdir(parents=True, exist_ok=True)
    assert args.audio_root.is_dir(), f"Ses klasörü bulunamadı: {args.audio_root}"

    P = pd.read_csv(args.participants)
    assert P["participant_id"].is_unique
    known_ids = set(P["participant_id"])
    R: dict = {"audio_root": str(args.audio_root),
               "settings": {"silence_db": args.silence_db, "near_dup_threshold": args.near_dup_threshold}}

    # ---------------- 1) ENVANTER (varsayımsız) ----------------
    all_files = sorted(p for p in args.audio_root.rglob("*") if p.is_file() and not p.name.startswith("._"))
    audio = [p for p in all_files if p.suffix.lower() in AUDIO_EXT]
    other = [p for p in all_files if p.suffix.lower() not in AUDIO_EXT]
    R["n_files_total"], R["n_audio_files"] = len(all_files), len(audio)
    R["non_audio_files"] = [str(p.relative_to(args.audio_root)) for p in other][:20]
    print(f"[envanter] {len(audio)} ses dosyası, {len(other)} diğer dosya", flush=True)
    assert len(audio) > 0, "Hiç ses dosyası bulunamadı — klasör yolu doğru mu?"

    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(lambda p: analyze_file(args.audio_root, p, args.silence_db), audio))
    inv = pd.DataFrame([r for r, _ in results])
    fps = [f for _, f in results]
    inv.to_csv(args.out_private / "audio_inventory.csv", index=False)

    def vc(col):
        return inv[col].value_counts(dropna=False).rename(lambda v: str(v)).to_dict() if col in inv else {}
    R["formats"] = {c: vc(c) for c in ["ext", "container", "codec", "codec_profile", "sr", "channels",
                                       "n_video_streams", "tag_major_brand", "tag_encoder", "tag_apple_model",
                                       "tag_apple_software", "has_location_tag"]}
    R["formats"]["n_probe_errors"] = int(inv["probe_error"].notna().sum()) if "probe_error" in inv else 0
    R["formats"]["n_decode_errors"] = int(inv["decode_error"].notna().sum()) if "decode_error" in inv else 0
    if "max_interchannel_absdiff" in inv:
        R["formats"]["n_fake_stereo"] = int(((inv["channels"] > 1) & (inv["max_interchannel_absdiff"] < 1e-6)).sum())
    if "audio_bit_rate" in inv:
        R["audio_bit_rate"] = inv["audio_bit_rate"].describe().round(0).to_dict()
    if "bandwidth_khz" in inv:
        R["bandwidth_khz"] = inv["bandwidth_khz"].describe(percentiles=[.01, .05, .5, .95, .99]).round(2).to_dict()
    if "duration_s" in inv:
        R["duration_s"] = inv["duration_s"].describe(percentiles=[.01, .05, .5, .95, .99]).round(3).to_dict()
    if R["formats"].get("has_location_tag", {}).get("True"):
        R["privacy_warning"] = "Dosyalarda konum etiketi var: ham dosyaları kimseyle paylaşma; önbellek WAV/NPY'leri etiket taşımaz."

    # ---------------- 2) EŞLEME ----------------
    parsed = inv["participant_id"].notna()
    R["naming"] = {"frac_matching_ID_slot_pattern": float(parsed.mean()),
                   "n_pid_dir_conflict": int(inv["pid_dir_conflict"].sum()),
                   "n_slot_out_of_range": int((parsed & ~inv["slot"].between(1, N_SLOTS)).sum()),
                   "n_pid_not_in_clinical": int((parsed & ~inv["participant_id"].isin(known_ids)).sum())}
    if parsed.mean() <= 0.5:
        R["mapping"] = "YAPILMADI: dosya adlandırması '<ID>_<slot>' kalıbıyla uyuşmuyor"
        (args.out_report / "audio_audit.json").write_text(json.dumps(R, indent=2, ensure_ascii=False, default=str))
        print("\n*** DUR: Dosya adlarının çoğu '<6 haneli ID>_<slot>' kalıbında değil. Eşleme tahminle YAPILMADI.")
        print("*** audio_inventory.csv'deki relpath sütununun ilk 30 satırını paylaş:")
        print(inv[["relpath"]].head(30).to_string())
        sys.exit(2)

    valid = inv[parsed & inv["slot"].between(1, N_SLOTS) & inv["participant_id"].isin(known_ids)].copy()
    valid["participant_id"] = valid["participant_id"].astype(int)
    valid["slot"] = valid["slot"].astype(int)
    E = pd.DataFrame([(int(pid), s) for pid in P["participant_id"] for s in range(1, N_SLOTS + 1)],
                     columns=["participant_id", "slot"]).merge(P[["participant_id", "label"]], on="participant_id")
    counts = valid.groupby(["participant_id", "slot"]).size().rename("n_files").reset_index()
    M = E.merge(counts, on=["participant_id", "slot"], how="left").fillna({"n_files": 0})
    M["status"] = np.select([M["n_files"] == 1, M["n_files"] == 0], ["OK", "MISSING"], default="MULTIPLE")
    # Meta veri sütunları da taşınmalı: aksi halde aşağıdaki meta-veri ↔ dönem analizi sessizce boş kalır
    # (testte yakalanan hata).
    keep = list(dict.fromkeys(["participant_id", "slot", "relpath", "file_sha256", "pcm_md5", "creation_local",
                               "creation_date_local", *META_CATEGORICAL, *NUISANCE_NUMERIC]))
    single = valid.drop_duplicates(["participant_id", "slot"], keep=False)  # yalnız tekil eşleşmeler
    M = M.merge(single[[c for c in keep if c in single.columns]], on=["participant_id", "slot"], how="left")
    M.to_csv(args.out_private / "recording_map.csv", index=False)

    unmatched = inv[~inv["relpath"].isin(valid["relpath"])]
    per_pid = M.groupby("participant_id")["status"].apply(lambda s: (s == "OK").sum())
    paper_ids = set(P.loc[P["in_paper_cohort"], "participant_id"])
    R["mapping"] = {
        "status_counts": M["status"].value_counts().to_dict(),
        "participants_with_all_7_ok": int((per_pid == N_SLOTS).sum()),
        "participants_with_zero_files": per_pid[per_pid == 0].index.tolist(),
        "participants_partial": per_pid[(per_pid > 0) & (per_pid < N_SLOTS)].index.tolist(),
        "paper_cohort_all_7_ok": int(per_pid[per_pid.index.isin(paper_ids)].eq(N_SLOTS).sum()),
        "paper_cohort_size": len(paper_ids),
        "missing_by_slot": M[M["status"] == "MISSING"].groupby("slot").size().to_dict(),
        "multiple_examples": M[M["status"] == "MULTIPLE"][["participant_id", "slot"]].head(10).to_dict("records"),
        "n_unmatched_audio_files": int(len(unmatched)),
        "unmatched_examples": unmatched["relpath"].head(20).tolist(),
    }
    ok = M[M["status"] == "OK"]
    if len(ok):
        R["mapping"]["same_file_in_two_slots_within_participant"] = int(
            ok.duplicated(["participant_id", "file_sha256"], keep=False).sum())

    # ---------------- 3) DUPLICATE LEAKAGE ----------------
    dups = {}
    for key in ["file_sha256", "pcm_md5"]:
        if key in inv:
            g = inv.dropna(subset=[key])
            multi = g[g.duplicated(key, keep=False)]
            cross = multi.groupby(key)["participant_id"].nunique()
            dups[key] = {"n_groups": int(multi[key].nunique()), "n_groups_across_participants": int((cross > 1).sum())}
    R["exact_duplicates"] = dups
    idx = [i for i, f in enumerate(fps) if f is not None]
    near = []
    if len(idx) > 1:
        F = np.stack([fps[i] for i in idx])
        C = (F @ F.T) / F.shape[1]
        pid = inv.loc[idx, "participant_id"].to_numpy()
        iu = np.triu_indices(len(idx), 1)
        cross = pid[iu[0]] != pid[iu[1]]
        vals, a, b = C[iu][cross], iu[0][cross], iu[1][cross]
        for k in np.argsort(-vals)[:50]:
            near.append({"corr": float(vals[k]), "a": inv["relpath"].iloc[idx[a[k]]], "b": inv["relpath"].iloc[idx[b[k]]],
                         "flag": bool(vals[k] >= args.near_dup_threshold)})
        R["near_duplicates"] = {"n_cross_participant_pairs_above_threshold": int((vals >= args.near_dup_threshold).sum()),
                                "cross_participant_corr_p99": float(np.percentile(vals, 99)) if len(vals) else None,
                                "max_corr": float(vals.max()) if len(vals) else None}
    pd.DataFrame(near).to_csv(args.out_private / "near_duplicate_pairs.csv", index=False)

    # ---------------- 4) META VERİ: dosyadaki kayıt zamanı CSV tarihiyle uyuşuyor mu? ----------------
    ok = ok.merge(P[["participant_id", "in_paper_cohort", "period", "collection_day"]], on="participant_id")
    if "creation_date_local" in ok and ok["creation_date_local"].notna().any():
        d = ok.dropna(subset=["creation_date_local"])
        both = d.dropna(subset=["collection_day"])
        delta = (pd.to_datetime(both["creation_date_local"]) - pd.to_datetime(both["collection_day"])).dt.days
        R["creation_time_vs_csv_date"] = {
            "n_files_with_creation_time": int(len(d)), "n_compared": int(len(both)),
            "frac_same_day": float((delta == 0).mean()) if len(both) else None,
            "frac_within_1_day": float((delta.abs() <= 1).mean()) if len(both) else None,
            "delta_days_quantiles": delta.quantile([0, .05, .5, .95, 1]).to_dict() if len(both) else None,
            "n_participants_csv_date_missing_but_file_has_date": int(d[d["collection_day"].isna()]["participant_id"].nunique()),
            "n_distinct_creation_dates": int(d["creation_date_local"].nunique()),
        }
    # ---------------- 5) CONFOUNDER: meta veri ve kayıt koşulları etiketle / dönemle ilişkili mi? ----------------
    okp = ok[ok["in_paper_cohort"]]
    meta = {}
    for col in META_CATEGORICAL:
        if col not in okp:
            continue
        s = okp[col].astype(str)
        if s.nunique() <= 1:
            meta[col] = {"uniform": True, "value": s.iloc[0] if len(s) else None}
            continue
        entry = {"uniform": False}
        ct = pd.crosstab(s, okp["label"])
        entry["by_label"] = ct.rename(columns={1: "asthma", 0: "healthy"}).to_dict("index")
        if ct.shape[1] == 2:
            entry["label_chi2_p"] = float(stats.chi2_contingency(ct).pvalue)
        pat = okp[(okp["label"] == 1) & okp["period"].isin(["early", "late"])]
        ctp = pd.crosstab(pat[col].astype(str), pat["period"])
        entry["patients_by_period"] = ctp.to_dict("index")
        if ctp.shape[0] > 1 and ctp.shape[1] == 2:
            entry["period_chi2_p"] = float(stats.chi2_contingency(ctp).pvalue)
        meta[col] = entry
    R["metadata_vs_label_and_period"] = meta

    conf = []
    for slot, d in okp.groupby("slot"):
        for feat in NUISANCE_NUMERIC:
            if feat not in d or d[feat].notna().sum() < 10:
                continue
            x = d[[feat, "label", "period"]].dropna()
            row = {"slot": int(slot), "feature": feat, "n": int(len(x))}
            if x["label"].nunique() == 2 and len(x) > 10:
                row["auc_label"] = float(roc_auc_score(x["label"], x[feat]))
            pts = x[(x["label"] == 1) & x["period"].isin(["early", "late"])]
            if pts["period"].nunique() == 2:
                row["auc_period_within_patients"] = float(roc_auc_score(pts["period"] == "late", pts[feat]))
            conf.append(row)
    C = pd.DataFrame(conf)
    for target, col in [("label", "auc_label"), ("period", "auc_period_within_patients")]:
        if col in C:
            C[f"{target}_dist"] = (C[col] - 0.5).abs()
            R[f"nuisance_top_{target}"] = C.sort_values(f"{target}_dist", ascending=False).head(15)[
                ["slot", "feature", "n", col]].round(3).to_dict("records")
        else:
            R[f"nuisance_top_{target}"] = "yetersiz veri"
    R["nuisance_note"] = ("AUC 0.5 = ilişki yok. 'period' tablosu YALNIZ hastalarda: etiket sabitken ölçüm kayıt dönemini "
                          "ayırıyorsa kayıt koşulları (ya da hasta profili) zamanla değişmiştir. Çoklu karşılaştırma: "
                          "hipotez üretir, tek başına sonuç değildir.")
    (args.out_report / "audio_audit.json").write_text(json.dumps(R, indent=2, ensure_ascii=False, default=str))

    # ---------------- Okunabilir özet ----------------
    L = ["# Audio audit (otomatik üretildi)", "",
         f"- Ses dosyası: {R['n_audio_files']} / toplam dosya {R['n_files_total']}; ses olmayan: {R['non_audio_files'][:5]}",
         f"- Formatlar: {json.dumps(R['formats'], ensure_ascii=False)}",
         f"- Bitrate: {R.get('audio_bit_rate')}", f"- Etkin bant genişliği (kHz): {R.get('bandwidth_khz')}",
         f"- Süre (s): {R.get('duration_s')}",
         f"- Adlandırma: {R['naming']}",
         f"- Eşleme: {json.dumps(R['mapping'], ensure_ascii=False, default=str)}",
         f"- Birebir kopyalar: {dups}", f"- Yakın kopya: {R.get('near_duplicates')}",
         f"- Dosya kayıt zamanı vs CSV tarihi: {R.get('creation_time_vs_csv_date')}"]
    if "privacy_warning" in R:
        L.append(f"- **GİZLİLİK:** {R['privacy_warning']}")
    L += ["", "## Meta veri: tekdüze mi, etikete / döneme göre değişiyor mu?", ""]
    for col, e in meta.items():
        L.append(f"- {col}: tekdüze ({e['value']})" if e["uniform"] else
                 f"- **{col}: DEĞİŞKEN** — etiket p={e.get('label_chi2_p', float('nan')):.3g}, "
                 f"hastalarda dönem p={e.get('period_chi2_p', float('nan')):.3g} · etiket: {e['by_label']} · dönem: {e['patients_by_period']}")
    for target in ["label", "period"]:
        top = R[f"nuisance_top_{target}"]
        L += ["", f"## Kayıt-koşulu ölçümleri ↔ {'etiket' if target == 'label' else 'kayıt dönemi (yalnız hastalar)'} (en uç 15)", ""]
        L.append(table_md(pd.DataFrame(top)) if isinstance(top, list) else top)
    L += ["", R["nuisance_note"]]
    (args.out_report / "audio_audit.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
