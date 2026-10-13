"""
my_voice_demo.py testi (gerçek veri GEREKMEZ; ffmpeg gerekir):  python tests/test_my_voice.py

    M1 dosya adı → slot (aaaaaa, ana, araba, gelecek, ordu, titiz, ünlem — NFC ve NFD)
    M2 döngüleme: tam uzunluk, seviye korunur (≤ 0.5 dB)
    M3 karıştırma: model görünümünde arka planın konuşma tepesine göre seviyesi, bağışçınınkiyle aynı (≤ 0.5 dB)
    M4 uçtan uca: sahte bağlam / anahtar / izler / split / MFCC özellikleri → mix → score-mfcc → load_all;
       orijinalin kayması 0, dışlama listesi bağışçıları ve bağışçı günlerini içeriyor
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import my_voice_demo as MV  # noqa: E402
from build_audio_cache import render, trim_bounds  # noqa: E402
from extract_mfcc_features import FEATURE_NAMES  # noqa: E402


def voice(sr, dur, seed, silent_tail=0.0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * sr)) / sr
    x = sum(np.sin(2 * np.pi * 130 * h * t) / h for h in range(1, 12)) * (0.5 + 0.5 * np.sin(2 * np.pi * 1.0 * t) ** 2)
    x = 0.3 * x / np.abs(x).max() + rng.standard_normal(len(t)) * 1e-5
    if silent_tail:
        x[int((dur - silent_tail) * sr):] = rng.standard_normal(int(silent_tail * sr)) * 1e-5
    return x.astype(np.float32)


def test_names():
    names = ["aaaaaa.m4a", "ana.m4a", "araba.m4a", "gelecek.m4a", "ordu.m4a", "titiz.m4a", "ünlem.m4a",
             unicodedata.normalize("NFD", "ünlem.m4a"), "Ünlem.M4A"]
    got = [MV.slot_of(n) for n in names]
    assert got == [1, 3, 2, 5, 4, 6, 7, 7, 7], got
    print("M1 ✓", got)


def test_loop():
    x = np.random.default_rng(1).standard_normal(32000 * 3) * 0.01
    y = MV.loop_to(x, 32000 * 11)
    lvl = 10 * np.log10(np.mean(y ** 2) / np.mean(x ** 2))
    assert len(y) == 32000 * 11 and abs(lvl) < 0.5, lvl
    print(f"M2 ✓ seviye farkı {lvl:+.2f} dB")


def test_ratio():
    rng = np.random.default_rng(2)
    from build_audio_cache import lowpass
    w = lowpass(rng.standard_normal(320000), 32000, 11000.0)                     # izler zaten 11 kHz alçak geçirenden geçti
    bg32 = (w / np.sqrt(np.mean(w ** 2)) * 10 ** (-50 / 20)).astype(np.float32)   # bağışçı ölçeğinde: −50 dBFS RMS
    v = voice(48000, 10.0, 3, silent_tail=3.0)
    v48, _ = MV.voice_model_view(v)
    mix = MV.mix_one(v48, bg32, 0.0)
    y32, _ = render(mix, {"trim_start_s": 0.0, "trim_end_s": len(mix) / 48000}, 32000)   # kırpmasız: kuyruk kalsın
    tail = y32[-int(2.0 * 32000):]                                    # sesin sustuğu bölüm: yalnız arka plan
    lvl = 10 * np.log10(np.mean(tail.astype(float) ** 2))
    assert abs(lvl - (-50)) < 0.5, f"arka plan seviyesi {lvl:.2f} dBFS (beklenen −50)"
    print(f"M3 ✓ arka plan model görünümünde {lvl:.2f} dBFS (bağışçı ölçeğinde −50)")


def test_e2e():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rng = np.random.default_rng(4)
        n = 60
        pids = np.arange(101001, 101001 + n)
        labels = (np.arange(n) % 4 != 0).astype(int)
        days = np.array([f"2024-0{1 + i // 20}-{1 + (i % 20) // 2:02d}" for i in range(n)])
        pd.DataFrame({"participant_id": pids, "label": labels, "recording_date": days}).to_csv(td / "ctx.csv", index=False)
        (td / "splits").mkdir()
        pd.DataFrame({"participant_id": pids, "repeat": 0, "outer_fold": 0, "role": "train", "inner_fold": 0}).to_csv(
            td / "splits" / "outer_r0.csv", index=False)
        F = [{"participant_id": p, "slot": s, "kept_s": 10.0, **dict(zip(FEATURE_NAMES, rng.standard_normal(44) + 0.5 * l))}
             for p, l in zip(pids, labels) for s in (2, 3)]
        pd.DataFrame(F).to_csv(td / "features.csv", index=False)
        # bağışçı izleri (seviye korunmuş) ve anahtar
        (td / "bgsel").mkdir(); (td / "tracks").mkdir()
        donors = [(pids[0], "A1", "L01a"), (pids[5], "B", "L02c")]
        tr, off = [], 0
        with open(td / "tracks" / "tracks_32k.f32", "wb") as f:
            for pid, var, code in donors:
                x = (rng.standard_normal(320000) * 0.003).astype("<f4")
                f.write(x.tobytes()); tr.append({"participant_id": pid, "variant": var, "offset": off, "n": 320000}); off += 320000
                sf.write(td / "bgsel" / f"{code}.wav", x, 32000)
        pd.DataFrame(tr).to_csv(td / "tracks" / "tracks_index.csv", index=False)
        pd.DataFrame([{"code": c, "participant_id": p, "variant": v} for p, v, c in donors]).to_csv(td / "key.csv", index=False)
        (td / "voices").mkdir()
        for name, s in [("araba.wav", 5), ("ana.wav", 6)]:
            sf.write(td / "voices" / name, voice(48000, 6.0, s), 48000)
        py, sc = sys.executable, str(REPO / "scripts" / "my_voice_demo.py")
        run = lambda *a: subprocess.run([py, sc, *map(str, a)], capture_output=True, text=True)
        r = run("mix", "--voice-dir", td / "voices", "--bg-dir", td / "bgsel", "--key", td / "key.csv", "--tracks-dir", td / "tracks",
                "--context", td / "ctx.csv", "--out-dir", td / "out")
        assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-2500:]
        excl = set(pd.read_csv(td / "out" / "excluded_PRIVATE.csv").participant_id)
        exp = set(pids[np.isin(days, days[[0, 5]])])
        assert excl == exp, "dışlama = bağışçılar + bağışçı günleri"
        idx = pd.read_csv(td / "out" / "mixcache" / "index.csv")
        assert len(idx) == 2 * 3 and len(list((td / "out" / "mixes_wav").glob("*.wav"))) == 6
        a32 = np.memmap(td / "out" / "mixcache" / "audio_32k.f32", dtype="<f4", mode="r")
        seg = lambda r: np.asarray(a32[int(r.offset_32k): int(r.offset_32k) + int(r.n_32k)])
        g = idx[idx.slot == 2]
        o, b = seg(g[g.bg == "orig"].iloc[0]), seg(g[g.bg == "L01a"].iloc[0])
        assert len(o) != len(b) or not np.allclose(o, b), "arka plan karışıma girmedi"
        r = run("score-mfcc", "--context", td / "ctx.csv", "--splits-dir", td / "splits", "--out-dir", td / "out",
                "--features", td / "features.csv")
        assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-2500:]
        S = MV.load_all(td / "out")
        assert len(S) == 6 and np.allclose(S[S.bg == "orig"].delta_logit, 0) and S.bg_group.isin(["orijinal", "astım", "sağlıklı"]).all()
        print(f"M4 ✓ mix → score-mfcc → load_all; dışlanan {len(excl)} katılımcı; kaymalar {S.delta_logit.round(2).tolist()}")


if __name__ == "__main__":
    test_names()
    test_loop()
    test_ratio()
    test_e2e()
    print("Hepsi geçti.")
