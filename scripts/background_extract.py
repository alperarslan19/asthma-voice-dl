"""
Adım 0 — Kayıtlardan arka plan (ortam sesi) çıkarma, kalite kapısı ve bilgi testi (EXP-032; D-046)

BİLİMSEL AMAÇ
    Kayıt bağlamı (gün, dönem, saat) etiketi neredeyse belirliyor. Açık kalan yollardan biri ORTAM (G2): kayıt
    odasının akustik arka planı. Bu script üç soruyu ayırır:
      0a  Arka plan çıkarılabiliyor mu? Kişi başına kaç saniye, gruplara göre dengeli mi?
      0b  Çıkarılan şey gerçekten arka plan mı? (konuşma sızıntısı testi: cinsiyet; A1–B spektrum tutarlılığı;
          kör dinleme) — ASTIM ETİKETİNE BAKMAZ; parametreler burada dondurulur
      0c  Arka plan tek başına etiketi / dönemi ayırıyor mu? (gün-gruplu CV) — yalnız D-046 KABUL'den sonra
    Bu, sonraki "arka plan değiştirme" testinin (modeller arka planı KULLANIYOR mu?) ön koşuludur: burada arka
    planda bilgi olup olmadığı ölçülür.

ÜÇ ARKA PLAN TÜRÜ (docs/BACKGROUND_DESIGN.md)
    A1 oda tonu     : konuşmadan uzak (koruma payı GUARD_S) VE taban gürültüsüne ROOMTONE_MARGIN_DB yakın kareler
    A2 tüm sessizlik: SESLİ (periyodik) konuşmadan uzak bütün kareler — nefes ve oda olayları dahil
    Düşük SNR'li kayıtlardan (≤ 20 dB) A1 / A2 ALINMAZ: orada "taban" konuşmanın kendisi olabilir (ör. "aaa")
    B  durağan tahmin: her geçerli kayıtta geniş bant enerjisi (100 Hz–8 kHz log-ortalaması) en düşük LOW_FRAC_B kare (64 ms) → ortalama güç
                      spektrumu; kayıtların eşit ağırlıklı ortalaması; sonra bu spektrumla gürültü sentezi. Kısa
                      aralara da sığar (A1'in 150 ms koruma payı gerekmez); yalnız durağan "oda rengi"ni taşır.
                      (Frekans başına kantil tahmini denendi: konuşma doluluğu > %70'te +3–5 dB sapma verdi — U4.)
    Hepsi MODEL GÖRÜNÜMÜNDE: önbellekle aynı zincir (48 kHz → 32 kHz + 11 kHz alçak geçiren → önbellekteki
    tepe kazancı) ve yalnız önbelleğin kırpma sınırları içinde (modelin gördüğü bölge).

VERİNİN YAPISI (katılımcı düzeyi girdiler / çıktılar Drive'da; git'e GİRMEZ — D-014)
    girdi : recording_map.csv (status, relpath, sr, channels), audio_cache_v1/index.csv (trim_start_s,
            trim_end_s, gain_db_32k), participant_context.csv (label, sex, recording_date, start_hour, days, age),
            splits_day/outer_r{0,1,2}.csv
    çıktı : <out_dir>/intervals.csv, rec_stats.csv, features.csv, tracks_32k.f32 + tracks_index.csv,
            psd_B.npy, listening/ (kör dinleme dosyaları + anahtar), probe_<task>.csv
    rapor : <report_dir>/BG-0a_inventory.{md,json}, BG-0b_qc.{md,json}, BG-0c_info.{md,json} — yalnız agrega

ALT KOMUTLAR
    extract   — 0a + özellikler + dinleme örnekleri (CPU, ~10–20 dk)
    probe     — --tasks sex (0b) | --tasks label period (0c); gün-gruplu dış CV, iç CV ile C, LR
    listening — doldurulmuş kör dinleme formunun özeti (0b)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.signal import resample_poly
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_audio import decode, frame_db  # noqa: E402
from build_audio_cache import render, to_source_sr  # noqa: E402
from extract_mfcc_features import FEATURE_NAMES, features as mfcc44  # noqa: E402

# ----------------------------------------------------------------------------- parametreler (D-046; 0b'den sonra donar)
SR = 32000
WIN_S, HOP_S = 0.025, 0.010                 # kare enerjisi: ses denetimi ve önbellekle aynı
DIGITAL_SILENCE_DB = -100.0                 # sayısal sıfır (kodlayıcı dolgusu) — taban hesabına ve arka plana girmez
SILENCE_DB, SNR_GATE_DB, FLOOR_MARGIN_DB = 35.0, 20.0, 10.0   # konuşma (aktif kare) kuralı: build_audio_cache ile aynı
EDGE_S = 0.05                               # kırpma sınırlarının içinde bırakılan pay
GUARD_S = 0.15                              # konuşma karelerinden uzaklık (kelime kuyruğu, yankı)
ROOMTONE_MARGIN_DB = 3.0                    # A1: taban + 3 dB
VOICING_R, F0_RANGE = 0.45, (60.0, 400.0)   # A2 konuşma maskesi: sesli kare = enerjiyle etkin VE periyodik
MIN_RUN_S = {"A1": 0.06, "A2": 0.10}        # en kısa parça
MIN_TOTAL_S = {"A1": 2.0, "A2": 3.0}        # katılımcı dahil olma eşiği (7 kaydın toplamı)
TRACK_S = 10.0                              # özellik / dinleme / sonraki değiştirme testi için iz uzunluğu
XFADE_S = 0.010                             # eşit güçlü çapraz geçiş
NFFT_B, HOP_B = 2048, 320                   # B ve LTAS için STFT (64 ms / 10 ms, Hann)
LOW_FRAC_B = 0.10                           # B: kayıt başına en sessiz kare oranı
BAND_CENTERS = 125.0 * 2.0 ** (np.arange(19) / 3.0)          # 1/3 oktav, 125 Hz … 8 kHz (11 kHz kesimin altında)
SEED = 0
VARIANTS = ["A1", "A2", "B"]
LTAS_COLS = [f"ltas{i:02d}" for i in range(len(BAND_CENTERS))]
MFCC_COLS = [f"mfcc_{n}" for n in FEATURE_NAMES]
FEATURE_SETS = {"ltas_abs": LTAS_COLS, "ltas_shape": LTAS_COLS, "mfcc44": MFCC_COLS}
C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
GATE = {"sex_auc_max": 0.60, "sex_auc_over_context": 0.05, "a1_b_median_abs_db": 3.0, "inclusion_gap_pp": 10.0,
        "listen_speech_frac": {"A1": 0.10, "A2": 0.20, "B": 0.10}}

HOP = int(round(HOP_S * SR))
WIN = int(round(WIN_S * SR))
GUARD_FRAMES = int(np.ceil((GUARD_S + WIN_S) / HOP_S - 1e-9))   # pencere uzunluğu da hesaba katılır
XF = int(round(XFADE_S * SR))
HANN_B = np.hanning(NFFT_B)


def params() -> dict:
    keys = ["SR", "WIN_S", "HOP_S", "DIGITAL_SILENCE_DB", "SILENCE_DB", "SNR_GATE_DB", "FLOOR_MARGIN_DB", "EDGE_S", "GUARD_S",
            "ROOMTONE_MARGIN_DB", "VOICING_R", "F0_RANGE", "MIN_RUN_S", "MIN_TOTAL_S", "TRACK_S", "XFADE_S", "NFFT_B", "HOP_B", "LOW_FRAC_B",
            "SEED", "GATE"]
    g = globals()
    return {k: g[k] for k in keys} | {"BAND_CENTERS_HZ": [round(float(c), 1) for c in BAND_CENTERS]}


# ----------------------------------------------------------------------------- sinyal işlemleri
def frame_masks(y: np.ndarray, t0: float, t1: float) -> dict:
    """Kare düzeyinde maskeler (32 kHz model görünümü). t0/t1: önbelleğin kırpma sınırları (s)."""
    db = frame_db(y, SR, WIN_S, HOP_S)
    n = len(db)
    starts = np.arange(n) * HOP
    inside = (starts >= (t0 + EDGE_S) * SR) & (starts + WIN <= (t1 - EDGE_S) * SR)
    digital = db < DIGITAL_SILENCE_DB
    ok = inside & ~digital
    assert ok.sum() >= 50, f"kırpma bölgesinde yeterli kare yok ({ok.sum()})"
    floor, speech = float(np.percentile(db[ok], 10)), float(np.percentile(db[ok], 95))
    snr = speech - floor
    thr = float(db[ok].max()) - SILENCE_DB
    if snr > SNR_GATE_DB:
        thr = max(thr, floor + FLOOR_MARGIN_DB)
    active = db > thr
    voiced = active & (voicing(y, n) >= VOICING_R)
    ker = np.ones(2 * GUARD_FRAMES + 1, dtype=int)
    near_active = np.convolve(active.astype(int), ker, mode="same") > 0
    near_voiced = np.convolve(voiced.astype(int), ker, mode="same") > 0
    # A1: enerjiyle etkin HER kareden uzak + tabana yakın (en tutucu). A2: yalnız SESLİ (periyodik) konuşmadan uzak —
    # nefes gibi sessiz-olmayan ama periyodik olmayan olaylar A2'de kalır. Ünsüzler sesli karelerin ≤ GUARD_S yakınında.
    # Düşük SNR'de (≤ SNR_GATE_DB) "taban" (p10) konuşmanın kendisi olabilir (ör. sürdürülen "aaa"): A1 ve A2 boş bırakılır;
    # o katılımcının diğer kayıtları ve B kullanılır.
    valid = snr > SNR_GATE_DB
    a1 = ok & ~near_active & (db <= floor + ROOMTONE_MARGIN_DB) if valid else np.zeros(n, dtype=bool)
    a2 = ok & ~near_voiced if valid else np.zeros(n, dtype=bool)
    return {"db": db, "ok": ok, "active": active & ok, "voiced": voiced & ok, "A1": a1, "A2": a2, "floor_db": floor,
            "speech_db": speech, "snr_db": snr, "n_digital": int((digital & inside).sum()), "bg_valid": bool(valid)}


def voicing(y: np.ndarray, n: int) -> np.ndarray:
    """Kare başına periyodiklik: F0_RANGE içindeki gecikmelerde yansız, normalize özilinti tepe değeri (0–1)."""
    fr = np.lib.stride_tricks.sliding_window_view(y.astype(np.float64), WIN)[::HOP][:n]
    fr = fr - fr.mean(axis=1, keepdims=True)
    X = np.fft.rfft(fr, 2 * WIN, axis=1)
    ac = np.fft.irfft(np.abs(X) ** 2, axis=1)[:, :WIN]
    lo, hi = int(SR / F0_RANGE[1]), int(SR / F0_RANGE[0])
    lags = np.arange(lo, hi + 1)
    r = ac[:, lags] / (WIN - lags) * WIN / (ac[:, :1] + 1e-20)
    out = np.zeros(n)
    out[:len(r)] = r.max(axis=1)
    return out


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Ardışık True karelerinin [başlangıç, bitiş) listesi."""
    m = np.concatenate([[False], mask.astype(bool), [False]])
    d = np.diff(m.astype(int))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def intervals(mask: np.ndarray, min_run_s: float) -> list[tuple[int, int]]:
    """Kare koşuları → örnek aralıkları [a, b). Bir kare [f·hop, f·hop + win) örneklerini kapsar."""
    out = []
    for f0, f1 in runs(mask):
        a, b = f0 * HOP, (f1 - 1) * HOP + WIN
        if b - a >= min_run_s * SR:
            out.append((int(a), int(b)))
    return out


def stft_power(y: np.ndarray) -> np.ndarray:
    if len(y) < NFFT_B:
        return np.zeros((0, NFFT_B // 2 + 1))
    fr = np.lib.stride_tricks.sliding_window_view(y.astype(np.float64), NFFT_B)[::HOP_B] * HANN_B
    return np.abs(np.fft.rfft(fr, axis=1)) ** 2


def b_psd(y: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """Bir kaydın durağan arka plan spektrumu: kırpma bölgesindeki (sayısal sessizlik hariç) STFT karelerinden geniş
    bant log-enerjisi en düşük LOW_FRAC_B oranının ortalama |X|²'si. Seçim sapması saf renkli gürültüde ≈ 0.4 dB (U4)."""
    P = stft_power(y)
    starts = np.arange(len(P)) * HOP_B
    tot = P.sum(axis=1)
    keep = (starts >= (t0 + EDGE_S) * SR) & (starts + NFFT_B <= (t1 - EDGE_S) * SR) & \
        (10 * np.log10(tot / (NFFT_B * np.sum(HANN_B ** 2)) + 1e-30) > DIGITAL_SILENCE_DB)
    P = P[keep]
    assert len(P) >= 50, "B için yeterli kare yok"
    f = np.fft.rfftfreq(NFFT_B, 1 / SR)
    e = np.log(P[:, (f >= 100) & (f <= 8000)] + 1e-30).mean(axis=1)   # log-ortalama: düşük frekans baskınlığını azaltır
    return P[e <= np.quantile(e, LOW_FRAC_B)].mean(axis=0)


def synth_noise(psd: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """psd: STFT(NFFT_B, Hann) karelerinde beklenen |X|². Beyaz gürültü tek büyük FFT ile |H|² = psd / Σw² olacak
    biçimde şekillendirilir; uzun pencereli düzgün spektrumda STFT gücü ≈ psd olur."""
    m = n + NFFT_B
    W = np.fft.rfft(rng.standard_normal(m))
    f_fine, f_b = np.fft.rfftfreq(m, 1 / SR), np.fft.rfftfreq(NFFT_B, 1 / SR)
    H2 = np.interp(f_fine, f_b, psd / np.sum(HANN_B ** 2))
    x = np.fft.irfft(W * np.sqrt(H2), n=m)
    return x[NFFT_B // 2: NFFT_B // 2 + n].astype(np.float32)


def build_track(segments: list[np.ndarray], n: int, rng: np.random.Generator) -> np.ndarray:
    """Parçaları rastgele sırayla, eşit güçlü çapraz geçişle uç uca ekler; gerekirse yeniden karıştırıp döngüler.
    İlişkisiz gürültüde cos² + sin² = 1 → seviye geçişte korunur."""
    segs = [s for s in segments if len(s) >= 3 * XF]
    assert segs, "çapraz geçiş için yeterli uzunlukta parça yok"
    t = np.linspace(0, np.pi / 2, XF, endpoint=False)
    fo, fi = np.cos(t), np.sin(t)
    out, length, order, i = [], 0, rng.permutation(len(segs)), 0
    tail = None
    while length < n:
        s = segs[order[i]].astype(np.float64)
        i += 1
        if i == len(order):
            order, i = rng.permutation(len(segs)), 0
        if tail is None:
            body, tail = s[:-XF], s[-XF:]
        else:
            body = np.concatenate([tail * fo + s[:XF] * fi, s[XF:-XF]])
            tail = s[-XF:]
        out.append(body)
        length += len(body)
    return np.concatenate(out)[:n].astype(np.float32)


def band_db(P_mean: np.ndarray) -> np.ndarray:
    f = np.fft.rfftfreq(NFFT_B, 1 / SR)
    lo, hi = BAND_CENTERS * 2 ** (-1 / 6), BAND_CENTERS * 2 ** (1 / 6)
    out = np.array([P_mean[(f >= a) & (f < b)].sum() for a, b in zip(lo, hi)])
    assert (out > 0).all(), "boş bant"
    return 10 * np.log10(out)


def track_features(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ltas = band_db(stft_power(x).mean(0) + 1e-30)
    m = mfcc44(resample_poly(x.astype(np.float64), 441, 640).astype(np.float32))   # 32 k → 22.05 k (MFCC cache modu)
    return ltas, m


def model_view(path: Path, sr: int, ch: int) -> tuple[np.ndarray, float]:
    """Önbellekle aynı zincir, kırpmasız tam kayıt: decode → mono → 48 kHz → 32 kHz + 11 kHz FIR → tepe kazancı."""
    x = decode(path, sr, ch).mean(axis=1).astype(np.float32)
    x48 = to_source_sr(x, sr)
    y, g = render(x48, {"trim_start_s": 0.0, "trim_end_s": len(x48) / 48000}, SR)
    return y, g


def seed_for(pid: int, variant: str) -> np.random.Generator:
    return np.random.default_rng([SEED, int(pid), VARIANTS.index(variant)])


# ----------------------------------------------------------------------------- yardımcılar
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ids_free(txt: str, ids) -> bool:
    found = {int(m) for m in re.findall(r"(?<![\d.])(\d{6})(?![\d.])", txt)} & {int(i) for i in ids}
    return not found


def q(s: pd.Series) -> str:
    s = pd.Series(s).dropna()
    if not len(s):
        return "—"
    return f"medyan {s.median():.2f} [p10 {s.quantile(.1):.2f}, p90 {s.quantile(.9):.2f}]"


def load_ctx(path: Path) -> pd.DataFrame:
    C = pd.read_csv(path)
    need = {"participant_id", "label", "sex", "recording_date", "start_hour", "days", "age"}
    miss = need - set(C.columns)
    assert not miss, f"participant_context.csv eksik sütun: {sorted(miss)}"
    assert C.participant_id.is_unique
    rd = pd.to_datetime(C["recording_date"])
    last = rd[C.label == 0].max()
    C["period"] = np.select([rd.isna(), rd <= last], ["unknown", "early"], "late")   # analyze_imbalance_selection ile aynı
    C["hour2"] = C["start_hour"] ** 2
    return C.set_index("participant_id")


# ----------------------------------------------------------------------------- 0a: çıkarma
def cmd_extract(a) -> None:
    t_start = time.time()
    rm = pd.read_csv(a.recording_map)
    rm = rm[rm.status == "OK"]
    ci = pd.read_csv(a.cache_index)[["participant_id", "slot", "trim_start_s", "trim_end_s", "gain_db_32k"]]
    items = rm.merge(ci, on=["participant_id", "slot"], how="inner", validate="one_to_one")
    assert len(items) == len(ci), f"önbellek dizini ile kayıt haritası eşleşmiyor ({len(items)} / {len(ci)})"
    items = items.sort_values(["participant_id", "slot"]).reset_index(drop=True)
    if a.limit:
        keep = items.participant_id.drop_duplicates().iloc[:a.limit]
        items = items[items.participant_id.isin(keep)].reset_index(drop=True)
    C = load_ctx(a.context)
    pids = items.participant_id.unique()
    print(f"0a: {len(items)} kayıt, {len(pids)} katılımcı", flush=True)
    a.out_dir.mkdir(parents=True, exist_ok=True)

    iv_rows, st_rows, ft_rows, tr_rows, psd_rows = [], [], [], [], []
    tracks_f = open(a.out_dir / "tracks_32k.f32", "wb")
    offset = 0
    n_track = int(TRACK_S * SR)
    for pi, (pid, g) in enumerate(items.groupby("participant_id", sort=True)):
        segs = {"A1": [], "A2": []}
        bpsd = []
        for r in g.itertuples():
            y, gain = model_view(a.audio_root / r.relpath, int(r.sr), int(r.channels))
            M = frame_masks(y, float(r.trim_start_s), float(r.trim_end_s))
            st = {"participant_id": pid, "slot": int(r.slot), "floor_db": M["floor_db"], "speech_db": M["speech_db"],
                  "snr_db": M["snr_db"], "bg_valid": M["bg_valid"], "n_digital": M["n_digital"],
                  "active_frac": float(M["active"].sum() / max(M["ok"].sum(), 1)),
                  "gain_db": gain, "gain_diff_db": gain - float(r.gain_db_32k)}
            for v in ["A1", "A2"]:
                ivs = intervals(M[v], MIN_RUN_S[v])
                st[f"{v}_s"] = sum(b - a_ for a_, b in ivs) / SR
                st[f"{v}_n"] = len(ivs)
                for a_, b in ivs:
                    iv_rows.append((pid, int(r.slot), v, a_ / SR, b / SR))
                    segs[v].append(y[a_:b])
            if M["bg_valid"]:
                bpsd.append(b_psd(y, float(r.trim_start_s), float(r.trim_end_s)))
            st_rows.append(st)
        psd = np.mean(bpsd, axis=0) if bpsd else np.full(NFFT_B // 2 + 1, np.nan)
        psd_rows.append(psd)
        for v in VARIANTS:
            total = sum(len(s) for s in segs[v]) / SR if v != "B" else float("nan")
            inc = (total >= MIN_TOTAL_S[v]) if v != "B" else len(bpsd) >= 1
            row = {"participant_id": pid, "variant": v, "included": bool(inc), "total_s": total,
                   "n_seg": len(segs[v]) if v != "B" else len(bpsd)}
            if inc:
                rng = seed_for(pid, v)
                x = synth_noise(psd, n_track, rng) if v == "B" else build_track(segs[v], n_track, rng)
                assert len(x) == n_track and np.isfinite(x).all()
                ltas, m = track_features(x)
                row.update(dict(zip(LTAS_COLS, ltas)) | dict(zip(MFCC_COLS, m)))
                row["track_rms_db"] = float(10 * np.log10(np.mean(x.astype(np.float64) ** 2) + 1e-30))
                tracks_f.write(x.astype("<f4").tobytes())
                tr_rows.append({"participant_id": pid, "variant": v, "offset": offset, "n": n_track})
                offset += n_track
            ft_rows.append(row)
        if (pi + 1) % 25 == 0:
            print(f"  {pi + 1}/{len(pids)} katılımcı ({time.time() - t_start:.0f} s)", flush=True)
    tracks_f.close()

    IV = pd.DataFrame(iv_rows, columns=["participant_id", "slot", "variant", "start_s", "end_s"])
    ST, FT, TR = pd.DataFrame(st_rows), pd.DataFrame(ft_rows), pd.DataFrame(tr_rows)
    assert len(ST) == len(items) and FT.groupby("participant_id").size().eq(3).all()
    assert (IV.end_s > IV.start_s).all()
    for name, df in [("intervals.csv", IV), ("rec_stats.csv", ST), ("features.csv", FT), ("tracks_index.csv", TR)]:
        df.to_csv(a.out_dir / name, index=False, float_format="%.8g", lineterminator="\n")
    np.save(a.out_dir / "psd_B.npy", np.array(psd_rows))
    pd.Series(sorted(pids), name="participant_id").to_csv(a.out_dir / "psd_B_index.csv", index=False)

    # A1 ↔ B spektrum tutarlılığı (durağan kısımda örtüşmeli)
    W = FT.pivot(index="participant_id", columns="variant", values=LTAS_COLS[0]).notna()
    both = W.index[W.get("A1", False) & W.get("B", False)]
    A1 = FT[FT.variant == "A1"].set_index("participant_id").loc[both, LTAS_COLS].to_numpy()
    B = FT[FT.variant == "B"].set_index("participant_id").loc[both, LTAS_COLS].to_numpy()
    cons = np.median(np.abs(A1 - B), axis=1) if len(both) else np.array([])

    listen = make_listening(a, FT, TR, n_track)
    rep = inventory_report(items, ST, FT, C, cons, listen, time.time() - t_start)
    rep["outputs_sha256"] = {n: sha256(a.out_dir / n)[:16] for n in ["intervals.csv", "rec_stats.csv", "features.csv",
                                                                  "tracks_32k.f32"]}
    write_report(a.report_dir, "BG-0a_inventory", rep, inventory_md(rep), C.index)
    print(f"0a bitti: {time.time() - t_start:.0f} s", flush=True)


def make_listening(a, FT: pd.DataFrame, TR: pd.DataFrame, n_track: int) -> dict:
    """Kör dinleme: üç türü de dahil olan katılımcılardan rastgele n; dosya adlarında tür harfle karıştırılır
    (a/b/c), anahtar ayrı dosyada. Seviye dinleme için −30 dBFS RMS'e eşitlenir (seviye bilgisi dinlemede yok)."""
    import soundfile as sf
    inc = FT.pivot(index="participant_id", columns="variant", values="included").fillna(False)
    cand = inc.index[inc.all(axis=1)].to_numpy()
    rng = np.random.default_rng(SEED + 7)
    pick = rng.choice(cand, size=min(a.n_listen, len(cand)), replace=False) if len(cand) else []
    d = a.out_dir / "listening"
    d.mkdir(exist_ok=True)
    mm = np.memmap(a.out_dir / "tracks_32k.f32", dtype="<f4", mode="r")
    T = TR.set_index(["participant_id", "variant"])
    key, sheet = [], []
    for i, pid in enumerate(pick, start=1):
        letters = rng.permutation(["a", "b", "c"])
        for v, L in zip(VARIANTS, letters):
            r = T.loc[(pid, v)]
            x = np.asarray(mm[int(r.offset): int(r.offset) + int(r.n)], dtype=np.float64)
            x = x * (10 ** (-30 / 20) / (np.sqrt(np.mean(x ** 2)) + 1e-12))
            x = x * min(1.0, 10 ** (-1 / 20) / (np.abs(x).max() + 1e-12))
            code = f"L{i:02d}{L}"
            sf.write(d / f"{code}.wav", x.astype(np.float32), SR, subtype="PCM_16")
            key.append({"code": code, "participant_id": pid, "variant": v})
            sheet.append({"code": code, "konusma_kalintisi_0_2": "", "nefes_0_2": "", "tik_bozulma_0_2": "",
                          "dongu_hissi_0_2": "", "not": ""})
    pd.DataFrame(key, columns=["code", "participant_id", "variant"]).to_csv(d / "listening_key.csv", index=False)        # ÖZEL: dinleme bitene kadar açılmaz
    cols = ["code", "konusma_kalintisi_0_2", "nefes_0_2", "tik_bozulma_0_2", "dongu_hissi_0_2", "not"]
    pd.DataFrame(sheet, columns=cols).sort_values("code").to_csv(d / "listening_sheet.csv", index=False)
    return {"n_participants": int(len(pick)), "n_files": int(len(key)), "dir": "listening/"}


def inventory_report(items, ST, FT, C, cons, listen, secs) -> dict:
    lab = C.label.reindex(FT.participant_id.unique())
    rep = {"exp": "EXP-032", "step": "0a", "params": params(),
           "n_recordings": int(len(ST)), "n_participants": int(FT.participant_id.nunique()),
           "label_counts": {"astım": int((lab == 1).sum()), "sağlıklı": int((lab == 0).sum())},
           "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                        "pandas": pd.__version__}, "seconds": round(secs, 1)}
    rep["gain_check"] = {"median_abs_diff_db": float(ST.gain_diff_db.abs().median()), "max_abs_diff_db": float(ST.gain_diff_db.abs().max())}
    rep["recordings"] = {"snr_db": q(ST.snr_db), "floor_db": q(ST.floor_db), "active_frac": q(ST.active_frac),
                         "n_low_snr_no_bg": int((~ST.bg_valid).sum()), "n_with_digital_silence": int((ST.n_digital > 0).sum()),
                         "A1_s_per_rec": q(ST.A1_s), "A2_s_per_rec": q(ST.A2_s)}
    by_slot = ST.groupby("slot")[["A1_s", "A2_s"]].median().round(2)
    rep["median_s_by_slot"] = {int(k): v for k, v in by_slot.to_dict("index").items()}
    F = FT.join(C[["label", "period"]], on="participant_id")
    inc = {}
    for v in VARIANTS:
        f = F[F.variant == v]
        r = {}
        for nm, m in {"sağlıklı": f.label == 0, "astım": f.label == 1, "astım-erken": (f.label == 1) & (f.period == "early"),
                      "astım-geç": (f.label == 1) & (f.period == "late")}.items():
            r[nm] = {"n": int(m.sum()), "dahil": int(f[m].included.sum()), "oran": round(float(f[m].included.mean()), 3) if m.sum() else None,
                     "toplam_s": q(f[m].total_s) if v != "B" else "—"}
        gap = 100 * abs((r["sağlıklı"]["oran"] or 0) - (r["astım"]["oran"] or 0))
        r["gap_pp"] = round(gap, 1)
        r["gap_flag"] = bool(gap > GATE["inclusion_gap_pp"])
        inc[v] = r
    rep["inclusion"] = inc
    rep["B_n_valid_recordings"] = q(F[F.variant == "B"].n_seg)
    rep["consistency_A1_B"] = {"n": int(len(cons)), "median_abs_db_per_participant": q(pd.Series(cons)),
                               "overall_median_db": float(np.median(cons)) if len(cons) else None,
                               "pass": bool(len(cons) and np.median(cons) <= GATE["a1_b_median_abs_db"])}
    rep["listening"] = listen
    return rep


def inventory_md(R: dict) -> str:
    L = [f"# BG-0a — Arka plan envanteri (EXP-032, Adım 0a; otomatik üretildi, katılımcı ID'si içermez)", "",
         f"- Kayıt {R['n_recordings']}, katılımcı {R['n_participants']} (astım {R['label_counts']['astım']} / sağlıklı "
         f"{R['label_counts']['sağlıklı']}); süre {R['seconds']} s",
         f"- Önbellek kazancıyla uyum: medyan |fark| {R['gain_check']['median_abs_diff_db']:.3f} dB, en büyük "
         f"{R['gain_check']['max_abs_diff_db']:.3f} dB (0'a yakın olmalı: aynı model görünümü)",
         f"- Kayıt başına: SNR {R['recordings']['snr_db']} dB; taban {R['recordings']['floor_db']} dBFS; konuşma oranı "
         f"{R['recordings']['active_frac']}",
         f"- Düşük SNR (≤ {SNR_GATE_DB:.0f} dB) → A1 / A2 alınmayan kayıt: {R['recordings']['n_low_snr_no_bg']}; sayısal sessizlik "
         f"içeren kayıt: {R['recordings']['n_with_digital_silence']}",
         f"- Kayıt başına arka plan (s): A1 {R['recordings']['A1_s_per_rec']}; A2 {R['recordings']['A2_s_per_rec']}",
         f"- B: katılımcı başına kullanılan (SNR > {SNR_GATE_DB:.0f} dB) kayıt sayısı {R['B_n_valid_recordings']}", "",
         "## Ses birimi başına medyan arka plan (s)", "", "| slot | A1 | A2 |", "|---|---|---|"]
    L += [f"| {k} | {v['A1_s']} | {v['A2_s']} |" for k, v in R["median_s_by_slot"].items()]
    L += ["", f"## Dahil olma (A1 ≥ {MIN_TOTAL_S['A1']} s, A2 ≥ {MIN_TOTAL_S['A2']} s, B ≥ 1 geçerli kayıt)", "",
          "| tür | sağlıklı | astım | astım-erken | astım-geç | fark (puan) |", "|---|---|---|---|---|---|"]
    for v, r in R["inclusion"].items():
        cell = lambda k: f"{r[k]['dahil']}/{r[k]['n']}"
        L.append(f"| {v} | {cell('sağlıklı')} | {cell('astım')} | {cell('astım-erken')} | {cell('astım-geç')} | "
                 f"{r['gap_pp']}{' ⚠' if r['gap_flag'] else ''} |")
    L += ["", "Toplam süre (dahil olsun olmasın, s):", ""]
    for v in ["A1", "A2"]:
        L.append(f"- {v}: " + "; ".join(f"{k} {R['inclusion'][v][k]['toplam_s']}" for k in ["sağlıklı", "astım-erken", "astım-geç"]))
    c = R["consistency_A1_B"]
    L += ["", "## A1 ↔ B spektrum tutarlılığı (kalite kapısı G-c)", "",
          f"- Katılımcı başına bantlar üzerinden medyan |A1 − B|: {c['median_abs_db_per_participant']} dB (n = {c['n']})",
          f"- Genel medyan {c['overall_median_db'] if c['overall_median_db'] is None else round(c['overall_median_db'], 2)} dB → "
          f"**{'GEÇTİ' if c['pass'] else 'KALDI'}** (eşik ≤ {GATE['a1_b_median_abs_db']} dB)", "",
          "## Kör dinleme", "",
          f"- {R['listening']['n_participants']} katılımcı × 3 tür = {R['listening']['n_files']} dosya Drive'da "
          f"`{R['listening']['dir']}`; form `listening_sheet.csv`. Tür adları harfle gizlendi; anahtar ayrı dosyada.",
          "", "Not: Bu adım astım etiketine dayalı bir başarım ölçmez; yalnız envanter ve kalite. Parametreler: "
          "`docs/BACKGROUND_DESIGN.md`, JSON'da `params`."]
    return "\n".join(L) + "\n"


def write_report(report_dir: Path, stem: str, rep: dict, md: str, ids) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    js = json.dumps(rep, indent=2, ensure_ascii=False, default=float)
    assert ids_free(md, ids) and ids_free(js, ids), f"{stem}: rapor katılımcı ID'si içeriyor (D-014)"
    (report_dir / f"{stem}.json").write_text(js)
    (report_dir / f"{stem}.md").write_text(md)
    print(f"Rapor: {report_dir / stem}.md")


# ----------------------------------------------------------------------------- 0b / 0c: gün-gruplu prob
def design(FT: pd.DataFrame, variant: str, fset: str) -> pd.DataFrame:
    f = FT[(FT.variant == variant) & FT.included].set_index("participant_id")
    X = f[FEATURE_SETS[fset]].astype(float)
    if fset == "ltas_shape":
        X = X.sub(X.mean(axis=1), axis=0)
    assert np.isfinite(X.to_numpy()).all()
    return X


def lr_pipe():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(class_weight="balanced", max_iter=5000))])


def task_target(C: pd.DataFrame, task: str) -> pd.Series:
    if task == "sex":
        s = C.sex.map({"M": 1, "F": 0})
    elif task == "label":
        s = C.label.astype(float)
    elif task == "period":                       # yalnız astımlılar: geç (son sağlıklı kayıttan sonra) vs erken
        s = pd.Series(np.where(C.period == "late", 1.0, 0.0), index=C.index)
        s[(C.label != 1) | ~C.period.isin(["early", "late"])] = np.nan
    else:
        raise ValueError(task)
    return s.dropna().astype(int)


def cv_arm(X: pd.DataFrame, y: pd.Series, splits: dict, C: pd.DataFrame, grid: bool = True) -> pd.DataFrame:
    ids = X.index.intersection(y.index)
    rows = []
    for r, S in splits.items():
        S = S[S.participant_id.isin(ids)]
        for k in range(5):
            B = S[S.outer_fold == k]
            tr, te = B[B.role == "train"], B[B.role == "test"]
            assert not set(tr.participant_id) & set(te.participant_id), f"r{r} k{k}: SIZINTI"
            dtr, dte = set(C.loc[tr.participant_id, "recording_date"].dropna()), set(C.loc[te.participant_id, "recording_date"].dropna())
            assert not dtr & dte, f"r{r} k{k}: aynı gün train ve test'te (gün gruplaması bozuk)"
            ytr, yte = y.loc[tr.participant_id].to_numpy(), y.loc[te.participant_id].to_numpy()
            if ytr.min() == ytr.max() or yte.min() == yte.max():
                continue
            Xtr, Xte = X.loc[tr.participant_id].to_numpy(), X.loc[te.participant_id].to_numpy()
            if grid:
                inner = tr.inner_fold.to_numpy()
                ok = all(0 < ytr[inner != j].sum() < (inner != j).sum() and 0 < ytr[inner == j].sum() < (inner == j).sum()
                         for j in np.unique(inner))
                if ok:
                    gs = GridSearchCV(lr_pipe(), {"clf__C": C_GRID}, cv=PredefinedSplit(inner), scoring="neg_log_loss").fit(Xtr, ytr)
                    p, c = gs.predict_proba(Xte)[:, 1], float(gs.best_params_["clf__C"])
                else:                                   # iç fold'da tek sınıf: sabit C (raporlanır)
                    m = clone(lr_pipe()).set_params(clf__C=1.0).fit(Xtr, ytr)
                    p, c = m.predict_proba(Xte)[:, 1], float("nan")
            else:
                m = clone(lr_pipe()).set_params(clf__C=1.0).fit(Xtr, ytr)
                p, c = m.predict_proba(Xte)[:, 1], 1.0
            rows.append(pd.DataFrame({"participant_id": te.participant_id.to_numpy(), "repeat": r, "fold": k, "prob": p,
                                      "y": yte, "C": c}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["participant_id", "repeat", "fold", "prob", "y", "C"])


def boot_auc(y, p, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            v.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def summarize(P: pd.DataFrame) -> dict:
    fa = P.groupby(["repeat", "fold"]).apply(lambda d: roc_auc_score(d.y, d.prob) if d.y.nunique() == 2 else np.nan,
                                              include_groups=False).dropna()
    pooled = [roc_auc_score(d.y, d.prob) for _, d in P.groupby("repeat")]
    r0 = P[P.repeat == P.repeat.min()]
    lo, hi = boot_auc(r0.y, r0.prob)
    return {"n": int(r0.participant_id.nunique()), "n_pos": int(r0.y.sum()), "fold_auc_mean": float(fa.mean()),
            "fold_auc_sd": float(fa.std()), "n_folds": int(len(fa)), "pooled_auc_mean": float(np.mean(pooled)),
            "pooled_auc_r0_ci": [lo, hi], "C_nan_frac": float(P.C.isna().mean())}


def cmd_probe(a) -> None:
    t0 = time.time()
    FT = pd.read_csv(a.features)
    C = load_ctx(a.context)
    splits = {r: pd.read_csv(a.splits_dir / f"outer_r{r}.csv") for r in range(a.n_repeats)}
    cohort = set(splits[0].participant_id)
    assert all(set(s.participant_id) == cohort for s in splits.values())
    FT = FT[FT.participant_id.isin(cohort)]
    if a.tasks != ["sex"]:
        dec = Path(a.decisions).read_text()
        m = re.search(r"## D-046.*?\nDURUM: \*\*(\w+)\*\*", dec, re.S)
        assert m and m.group(1) == "KABUL", "0c yalnız D-046 KABUL (parametreler donduruldu) olduktan sonra çalışır"
    rep = {"exp": "EXP-032", "step": "0b" if a.tasks == ["sex"] else "0c", "params": params(), "tasks": {}}
    all_rows = []
    for task in a.tasks:
        y = task_target(C.loc[C.index.isin(cohort)], task)
        res = {}
        arms = {f"{v}|{fs}": design(FT, v, fs) for v in VARIANTS for fs in FEATURE_SETS}
        if task in ("sex", "label"):
            ok = C.loc[C.index.isin(cohort)].dropna(subset=["start_hour", "days"])
            arms["ref_context"] = ok[["start_hour", "hour2", "days"]]
        if task == "sex" and a.speech_features:
            sp = pd.read_csv(a.speech_features)
            arms["speech_mfcc_pos_control"] = sp.groupby("participant_id")[FEATURE_NAMES].mean()
        for name, X in arms.items():
            P = cv_arm(X, y, splits, C, grid=not name.startswith("ref_"))
            if P.empty or P.groupby("repeat").y.nunique().min() < 2:
                res[name] = {"insufficient": True, "n": int(len(X.index.intersection(y.index)))}
                print(f"  {task:6s} {name:28s} yetersiz (fold'larda tek sınıf)", flush=True)
                continue
            res[name] = summarize(P)
            all_rows.append(P.assign(task=task, arm=name))
            print(f"  {task:6s} {name:28s} AUC {res[name]['fold_auc_mean']:.3f} ± {res[name]['fold_auc_sd']:.3f} "
                  f"(n {res[name]['n']})", flush=True)
        if task == "label":
            try:
                res["_t1"] = t1_block(pd.concat([p for p in all_rows if p.task.iloc[0] == "label"]), C)
            except Exception as ex:                      # ikincil analiz; birincil sonuçları durdurmaz
                res["_t1"] = {"arm": "A1|ltas_abs", "error": f"{type(ex).__name__}: {ex}"}
        rep["tasks"][task] = res
    if a.tasks == ["sex"]:
        rep["gate_leakage"] = leakage_gate(rep["tasks"]["sex"])
    out = pd.concat(all_rows, ignore_index=True)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out_dir / f"probe_{'_'.join(a.tasks)}.csv", index=False)       # katılımcı düzeyi → Drive
    rep["seconds"] = round(time.time() - t0, 1)
    stem = "BG-0b_qc" if a.tasks == ["sex"] else "BG-0c_info"
    write_report(a.report_dir, stem, rep, probe_md(rep), C.index)


def leakage_gate(res: dict) -> dict:
    ref = res.get("ref_context", {}).get("fold_auc_mean", 0.5)
    thr = max(GATE["sex_auc_max"], ref + GATE["sex_auc_over_context"])
    out = {"threshold": thr, "ref_context_auc": ref}
    for v in VARIANTS:
        aucs = {fs: res[f"{v}|{fs}"].get("fold_auc_mean", float("nan")) for fs in FEATURE_SETS}
        out[v] = {"aucs": aucs, "pass": bool(all(np.isfinite(list(aucs.values()))) and max(aucs.values()) <= thr)}
    return out


def t1_block(P: pd.DataFrame, C: pd.DataFrame) -> dict:
    """İkincil (betimsel): A1|ltas_abs skorunun bağlamın (saat, tarih, yaş) ötesinde etiketle ilişkisi."""
    import evaluate_context as E
    s = P[P.arm == "A1|ltas_abs"].groupby("participant_id").prob.mean()
    D = C.loc[s.index].dropna(subset=["start_hour", "days", "age", "recording_date"])
    Xc = E.context_matrix(D, 4, hour=True, date=True, age=True)
    g = D.recording_date.astype("category").cat.codes.to_numpy()
    t = E.t1(D.label.to_numpy(float), Xc, E.logit(s.loc[D.index].to_numpy()), groups=g, ml=False, jackknife=True)
    return {"arm": "A1|ltas_abs", "n": int(len(D)), "or_sd": t["or_sd"], "p_plr": t["p_lrt"], "p_cv3": t.get("p_cv3")}


def probe_md(R: dict) -> str:
    title = "BG-0b — Kalite kapısı: konuşma sızıntısı (cinsiyet)" if R["step"] == "0b" else "BG-0c — Bilgi testi"
    L = [f"# {title} (EXP-032; otomatik üretildi, katılımcı ID'si içermez)", "",
         "Gün-gruplu dış CV (3 tekrar × 5 fold, `splits_day`), iç CV ile C (neg log-loss), dengeli LR. "
         "AUC: fold ort ± SD; havuz AUC (3 tekrar ort.) ve tekrar-0 havuzunun katılımcı bootstrap %95 GA'sı.", ""]
    for task, res in R["tasks"].items():
        L += [f"## Görev: {task}", "", "| kol | n (pozitif) | fold AUC ort ± SD | havuz AUC | %95 GA (r0) | C sabit pay |",
              "|---|---|---|---|---|---|"]
        for name, s in res.items():
            if name.startswith("_"):
                continue
            if s.get("insufficient"):
                L.append(f"| {name} | {s['n']} | yetersiz (fold'larda tek sınıf) | — | — | — |")
                continue
            L.append(f"| {name} | {s['n']} ({s['n_pos']}) | {s['fold_auc_mean']:.3f} ± {s['fold_auc_sd']:.3f} | "
                     f"{s['pooled_auc_mean']:.3f} | [{s['pooled_auc_r0_ci'][0]:.3f}, {s['pooled_auc_r0_ci'][1]:.3f}] | "
                     f"{s['C_nan_frac']:.2f} |")
        if "_t1" in res:
            t = res["_t1"]
            if "error" in t:
                L += ["", f"İkincil T1 hesaplanamadı: {t['error']}"]
            else:
                L += ["", f"İkincil T1 ({t['arm']}; bağlam saat + tarih + yaş; n {t['n']}): OR/SD {t['or_sd']:.2f}, PLR p "
                      f"{t['p_plr']:.3g}, CV3 p {t['p_cv3'] if t['p_cv3'] is None else format(t['p_cv3'], '.3g')}"]
        L.append("")
    if "gate_leakage" in R:
        g = R["gate_leakage"]
        L += ["## Kapı G-b (konuşma sızıntısı)", "",
              f"Eşik: her özellik setinde cinsiyet AUC ≤ max({GATE['sex_auc_max']}, bağlam referansı + "
              f"{GATE['sex_auc_over_context']}) = **{g['threshold']:.3f}** (bağlam referansı {g['ref_context_auc']:.3f}).", "",
              "| tür | en yüksek AUC | sonuç |", "|---|---|---|"]
        for v in VARIANTS:
            L.append(f"| {v} | {max(g[v]['aucs'].values()):.3f} | **{'GEÇTİ' if g[v]['pass'] else 'KALDI'}** |")
        L += ["", "Pozitif kontrol (konuşma MFCC'si, 7 birimin ortalaması) cinsiyeti yüksek AUC ile ayırmalı; ayırmıyorsa test "
                  "güçsüzdür ve kapı yorumlanmaz."]
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------- 0b: dinleme özeti
def cmd_listening(a) -> None:
    S, K = pd.read_csv(a.sheet), pd.read_csv(a.key)
    cols = ["konusma_kalintisi_0_2", "nefes_0_2", "tik_bozulma_0_2", "dongu_hissi_0_2"]
    D = S.merge(K[["code", "variant"]], on="code", validate="one_to_one")
    assert D[cols].notna().all().all(), "formda boş puan var"
    rep = {"exp": "EXP-032", "step": "0b-listening", "n_files": int(len(D)), "by_variant": {}}
    for v, g in D.groupby("variant"):
        rep["by_variant"][v] = {c: {"ort": float(g[c].mean()), "≥1 oranı": float((g[c] >= 1).mean())} for c in cols}
        rep["by_variant"][v]["gate_speech_pass"] = bool((g["konusma_kalintisi_0_2"] >= 1).mean() <= GATE["listen_speech_frac"][v])
    L = ["# BG-0b — Kör dinleme özeti (EXP-032; katılımcı ID'si içermez)", "", "| tür | " + " | ".join(cols) + " | konuşma kapısı |",
         "|---|" + "---|" * (len(cols) + 1)]
    for v, r in rep["by_variant"].items():
        L.append(f"| {v} | " + " | ".join(f"{r[c]['ort']:.2f} (≥1: {r[c]['≥1 oranı']:.0%})" for c in cols) +
                 f" | {'GEÇTİ' if r['gate_speech_pass'] else 'KALDI'} (eşik ≤ {GATE['listen_speech_frac'][v]:.0%}) |")
    write_report(a.report_dir, "BG-0b_listening", rep, "\n".join(L) + "\n", K.participant_id)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--audio-root", type=Path, required=True)
    e.add_argument("--recording-map", type=Path, required=True)
    e.add_argument("--cache-index", type=Path, required=True)
    e.add_argument("--context", type=Path, required=True)
    e.add_argument("--out-dir", type=Path, required=True)
    e.add_argument("--report-dir", type=Path, required=True)
    e.add_argument("--n-listen", type=int, default=30)
    e.add_argument("--limit", type=int, default=0, help="yalnız ilk N katılımcı (smoke test)")
    p = sub.add_parser("probe")
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--context", type=Path, required=True)
    p.add_argument("--splits-dir", type=Path, required=True)
    p.add_argument("--tasks", nargs="+", required=True, choices=["sex", "label", "period"])
    p.add_argument("--speech-features", type=Path, default=None)
    p.add_argument("--decisions", type=Path, default=Path(__file__).resolve().parents[1] / "DECISIONS.md")
    p.add_argument("--n-repeats", type=int, default=3)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--report-dir", type=Path, required=True)
    li = sub.add_parser("listening")
    li.add_argument("--sheet", type=Path, required=True)
    li.add_argument("--key", type=Path, required=True)
    li.add_argument("--report-dir", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "probe" and "sex" in a.tasks and len(a.tasks) > 1:
        ap.error("sex (0b) ile label/period (0c) aynı çağrıda çalıştırılmaz")
    {"extract": cmd_extract, "probe": cmd_probe, "listening": cmd_listening}[a.cmd](a)


if __name__ == "__main__":
    main()
