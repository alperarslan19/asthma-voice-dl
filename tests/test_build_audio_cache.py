"""
build_audio_cache.py için kendi kendine yeten test (gerçek veri GEREKMEZ; ffmpeg gerekir).

Bilinen özellikleri olan sahte kayıtlar üretir (AAC/.m4a), script'i çalıştırır ve her kuralı assert eder:

    python tests/test_build_audio_cache.py

Kontroller ve tuzaklar:
    K1 MISSING satırı işlenmez; dizinde yalnız OK kayıtlar
    K2 1.0 s baş / 2.0 s son sessizlik → ~0.9 s / ~1.9 s kırpılır (0.10 s pay kalır)
    K3 her parçanın tepe değeri −1 dBFS
    K4 32 kHz yolunda etkin bant genişliği ≤ 11.2 kHz (geniş bantlı kaynakta önce > 14 kHz)
    K5 44.1 kHz kaynak doğru süreyle işlenir
    K6 iki ayrı çalıştırma birebir aynı dosyaları üretir (sha256)
    K7 32k ve 16k sürümleri aynı kırpma sınırlarını kullanır (süreler eşit)
    K8 rapor katılımcı ID'si içermez
    T1 dosyanın sha256'ı recording_map'tekinden farklı → script hata verir
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "build_audio_cache.py"
PEAK = 10 ** (-1 / 20)


def voice(sr: int, dur: float, seed: int, wideband: bool) -> np.ndarray:
    r = np.random.default_rng(seed)
    t = np.arange(int(dur * sr)) / sr
    f0 = r.uniform(110, 220) * (1 + 0.03 * np.sin(2 * np.pi * 4 * t))
    x = sum(np.sin(2 * np.pi * np.cumsum(f0 * k) / sr) / k for k in range(1, 12))
    if wideband:  # nefes/türbülans benzeri geniş bant bileşen → yüksek frekans içeriği
        x = x + 0.3 * r.standard_normal(len(t))
    return 0.4 * x / np.abs(x).max()


def make_file(path: Path, sr: int, seed: int, lead: float, trail: float, wideband: bool, bitrate: str) -> None:
    r = np.random.default_rng(seed + 100)
    x = np.concatenate([np.zeros(int(lead * sr)), voice(sr, 6.0, seed, wideband), np.zeros(int(trail * sr))])
    x = x + 1e-4 * r.standard_normal(len(x))          # ~−80 dB oda gürültüsü
    tmp = path.with_suffix(".wav")
    sf.write(tmp, x.astype(np.float32), sr, subtype="PCM_24")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-c:a", "aac", "-b:a", bitrate, str(path)], check=True)
    tmp.unlink()


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(root: Path, rm: pd.DataFrame, work: Path, tag: str) -> subprocess.CompletedProcess:
    d = work / tag
    d.mkdir()
    rm.to_csv(d / "recording_map.csv", index=False)
    return subprocess.run([sys.executable, str(SCRIPT), "--audio-root", str(root), "--recording-map", str(d / "recording_map.csv"),
                           "--out-dir", str(d / "cache"), "--report-dir", str(d / "report")], capture_output=True, text=True)


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="test_audio_cache_"))
    try:
        root = work / "audio"
        (root / "soundData").mkdir(parents=True)
        specs = [  # (pid, slot, sr, lead, trail, wideband, bitrate, encoder_tag)
            (900001, 1, 48000, 1.0, 2.0, True, "192k", "Lavf59.16.100"),
            (900001, 2, 48000, 0.3, 0.3, False, "96k", "Core Media Audio"),
            (900002, 1, 44100, 0.5, 0.5, True, "128k", "Core Media Audio"),
            (900002, 2, 48000, 0.2, 1.0, True, "160k", "Lavf59.16.100"),
        ]
        rows = []
        for i, (pid, slot, sr, lead, trail, wb, br, enc) in enumerate(specs):
            rel = f"soundData/{pid}_{slot}.m4a"
            make_file(root / rel, sr, i, lead, trail, wb, br)
            rows.append({"participant_id": pid, "slot": slot, "status": "OK", "relpath": rel,
                         "file_sha256": sha(root / rel), "sr": float(sr), "channels": 1, "tag_encoder": enc})
        rows.append({"participant_id": 900003, "slot": 1, "status": "MISSING", "relpath": np.nan,
                     "file_sha256": np.nan, "sr": np.nan, "channels": np.nan, "tag_encoder": np.nan})   # K1
        rm = pd.DataFrame(rows)

        a = run(root, rm, work, "a")
        assert a.returncode == 0, a.stdout[-2000:] + a.stderr[-3000:]
        b = run(root, rm, work, "b")
        assert b.returncode == 0, b.stderr[-3000:]

        ca, cb = work / "a" / "cache", work / "b" / "cache"
        idx = pd.read_csv(ca / "index.csv")
        assert len(idx) == 4 and 900003 not in set(idx.participant_id), "K1"
        r0 = idx.iloc[0]
        assert abs(r0.lead_trimmed_s - 0.9) < 0.06 and abs(r0.trail_trimmed_s - 1.9) < 0.06, \
            f"K2: baş {r0.lead_trimmed_s:.3f} s, son {r0.trail_trimmed_s:.3f} s"
        for sr, k in [(32000, "32k"), (16000, "16k")]:
            mm = np.memmap(ca / f"audio_{k}.f32", dtype="<f4", mode="r")
            for o, n in zip(idx[f"offset_{k}"], idx[f"n_{k}"]):
                assert abs(np.abs(mm[o:o + n]).max() - PEAK) < 1e-6, "K3"
        wide = idx[idx.relpath.isin(["soundData/900001_1.m4a", "soundData/900002_2.m4a"])]
        assert (wide.bw_before_khz > 14).all(), f"K4 ön koşul: geniş bant kaynak {wide.bw_before_khz.tolist()}"
        assert (idx.bw_after_32k_khz <= 11.2).all(), f"K4: sonra {idx.bw_after_32k_khz.tolist()}"
        r2 = idx[idx.src_sr == 44100].iloc[0]
        assert abs(r2.src_duration_s - 7.0) < 0.1, f"K5: {r2.src_duration_s}"
        assert np.allclose(idx.n_32k / 32000, idx.n_16k / 16000, atol=1e-4), "K7"
        for f in ["audio_32k.f32", "audio_16k.f32", "index.csv"]:
            assert sha(ca / f) == sha(cb / f), f"K6: {f} belirlenimci değil"
        rep = (work / "a" / "report" / "audio_cache_report.md").read_text() + \
              (work / "a" / "report" / "audio_cache_report.json").read_text()
        assert not any(str(p) in rep for p in [900001, 900002, 900003]), "K8"
        json.loads((ca / "cache_info.json").read_text())

        # T1: dosya değişmiş (sha256 farklı)
        rm2 = rm.copy()
        rm2.loc[0, "file_sha256"] = "0" * 64
        t1 = run(root, rm2, work, "t1")
        assert t1.returncode != 0 and "sha256" in t1.stderr, "T1 yakalanmadı"
        print("Tüm kontroller geçti (K1–K8, T1).")
        print(f"  kırpma: baş {r0.lead_trimmed_s:.3f} s / son {r0.trail_trimmed_s:.3f} s; "
              f"bant genişliği önce {wide.bw_before_khz.round(1).tolist()} → sonra {idx.bw_after_32k_khz.round(2).tolist()} kHz")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
