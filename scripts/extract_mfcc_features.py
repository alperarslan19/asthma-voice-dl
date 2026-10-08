"""
Faz 1 — MFCC + spektral özet özellikleri (EXP-010 / EXP-011; D-031)

BİLİMSEL AMAÇ
    Yayınlanmış çalışmanın (Alagöz ve ark., BMC Pulm Med 2026) 44 boyutlu özellik tarifini iki girdiden çıkarmak:
      --mode paper : ORİJİNAL .m4a → 22.05 kHz → librosa enerji tabanlı kırpma (top_db=60) → 44 özellik
                     (EXP-010, sadık yeniden üretim; makaledeki ön işleme)
      --mode cache : HARMONİZE önbellek (D-030, 32 kHz, kırpılmış, −1 dBFS) → 22.05 kHz → 44 özellik
                     (EXP-011; derin modellerle AYNI girdi → RQ1 karşılaştırması adil olur)

    44 özellik [FROM PAPER]: 12 MFCC + Δ + ΔΔ (çerçeve 2048, adım 512, Hamming) → zaman ortalaması = 36;
    sıfır geçiş oranı, spektral merkez, bant genişliği, roll-off → ortalama ve SD = 8.
    Makalede belirtilmeyen ayrıntılar [DECISION, D-031]: MFCC'lerin zaman üzerinden ORTALAMASI alınır
    (makale "36 descriptor" diyor → yalnız ortalama); librosa varsayılanı c0'ı içerir; spektral özetlerde
    librosa varsayılan penceresi (Hann); kırpma = librosa.effects.trim(top_db=60), makalede eşik yok.

VERİNİN YAPISI
    recording_map.csv (status == OK, relpath, sr, channels)   — paper modu
    audio_cache_v1/index.csv + audio_32k.f32                   — cache modu
ÇIKTI (Drive; katılımcı düzeyi → git'e GİRMEZ)
    <out_csv>: participant_id, slot, f00 … f43 (+ n_frames, kept_s)

ÇALIŞTIRMA
    python scripts/extract_mfcc_features.py --mode paper --audio-root /content/audio \
        --recording-map .../recording_map.csv --out-csv .../mfcc_features_paper.csv
    python scripts/extract_mfcc_features.py --mode cache --cache-dir .../audio_cache_v1 \
        --out-csv .../mfcc_features_cache.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_audio import decode  # noqa: E402

FEAT_SR = 22050
N_MFCC, N_FFT, HOP = 12, 2048, 512
TRIM_TOP_DB = 60
FEATURE_NAMES = ([f"mfcc{i:02d}_mean" for i in range(N_MFCC)] + [f"d1_mfcc{i:02d}_mean" for i in range(N_MFCC)]
                 + [f"d2_mfcc{i:02d}_mean" for i in range(N_MFCC)]
                 + [f"{n}_{s}" for n in ["zcr", "centroid", "bandwidth", "rolloff"] for s in ["mean", "sd"]])
assert len(FEATURE_NAMES) == 44


def features(y: np.ndarray, sr: int = FEAT_SR) -> np.ndarray:
    """44 boyutlu vektör. Girdi: mono float32, 22.05 kHz."""
    assert y.ndim == 1 and sr == FEAT_SR
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP, window="hamming")
    assert m.shape[0] == N_MFCC and m.shape[1] >= 9, f"çok kısa sinyal ({m.shape[1]} çerçeve; Δ için ≥ 9 gerekir)"
    d1, d2 = librosa.feature.delta(m), librosa.feature.delta(m, order=2)
    spec = [librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP),
            librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP),
            librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP),
            librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP)]
    v = np.concatenate([m.mean(1), d1.mean(1), d2.mean(1)] + [np.array([s.mean(), s.std()]) for s in spec])
    assert v.shape == (44,) and np.isfinite(v).all()
    return v.astype(np.float64)


def paper_signal(path: Path, sr: int, channels: int) -> np.ndarray:
    """Makaledeki ön işleme: 22.05 kHz'e yeniden örnekleme (librosa varsayılanı soxr_hq) + enerji tabanlı kırpma.
    Çözme ffmpeg ile (librosa.load'un m4a için yaptığıyla aynı yol; 16 bit nicemleme hariç) [INFERENCE]."""
    x = decode(path, sr, channels).mean(axis=1).astype(np.float32)
    y = librosa.resample(x, orig_sr=sr, target_sr=FEAT_SR, res_type="soxr_hq")
    yt, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB, frame_length=N_FFT, hop_length=HOP)
    return yt


def cache_signal(a32: np.memmap, offset: int, n: int) -> np.ndarray:
    """Harmonize önbellek (zaten kırpılmış ve normalize) → 22.05 kHz (32000 · 441/640 = 22050)."""
    x = np.asarray(a32[offset:offset + n], dtype=np.float64)
    return resample_poly(x, 441, 640).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["paper", "cache"])
    ap.add_argument("--audio-root", type=Path)
    ap.add_argument("--recording-map", type=Path)
    ap.add_argument("--cache-dir", type=Path)
    ap.add_argument("--out-csv", required=True, type=Path)
    args = ap.parse_args()
    t0 = time.time()

    if args.mode == "paper":
        assert args.audio_root and args.recording_map, "--audio-root ve --recording-map gerekli"
        rm = pd.read_csv(args.recording_map)
        items = rm[rm["status"] == "OK"].sort_values(["participant_id", "slot"]).reset_index(drop=True)
    else:
        assert args.cache_dir, "--cache-dir gerekli"
        items = pd.read_csv(args.cache_dir / "index.csv").sort_values(["participant_id", "slot"]).reset_index(drop=True)
        a32 = np.memmap(args.cache_dir / "audio_32k.f32", dtype="<f4", mode="r")
        assert len(a32) == int(items["n_32k"].sum()), "önbellek dosyası dizinle uyuşmuyor"
    assert not items.duplicated(["participant_id", "slot"]).any()
    print(f"{args.mode}: {len(items)} kayıt, {items.participant_id.nunique()} katılımcı")

    rows = []
    for i, r in enumerate(items.itertuples()):
        y = paper_signal(args.audio_root / r.relpath, int(r.sr), int(r.channels)) if args.mode == "paper" \
            else cache_signal(a32, int(r.offset_32k), int(r.n_32k))
        rows.append([r.participant_id, r.slot, len(y) / FEAT_SR, *features(y)])
        if (i + 1) % 300 == 0:
            print(f"  {i + 1}/{len(items)} ({time.time() - t0:.0f} s)", flush=True)
    df = pd.DataFrame(rows, columns=["participant_id", "slot", "kept_s", *FEATURE_NAMES])
    assert df[FEATURE_NAMES].notna().all().all() and np.isfinite(df[FEATURE_NAMES].to_numpy()).all()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False, float_format="%.8g", lineterminator="\n")
    print(f"Yazıldı: {args.out_csv.name} — {df.shape[0]} satır × 44 özellik; slot başına "
          f"{df.slot.value_counts().sort_index().to_dict()}; kalan süre medyan {df.kept_s.median():.2f} s "
          f"(min {df.kept_s.min():.2f}); {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
