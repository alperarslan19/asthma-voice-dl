"""
background_extract.py için kendi kendine yeten test (gerçek veri GEREKMEZ; ffmpeg gerekir).

    python tests/test_background.py

Birim testleri (bilinen yapıda sentetik sinyaller):
    U1 A1 hiçbir konuşma patlamasına GUARD_S'den yakın değil; nefes A1'de yok, A2'de var; A2 ⊇ A1
    U2 sayısal sessizlik (sıfırlar) tabanı bozmaz ve sayılır
    U3 düşük SNR'de A1 / A2 alınmaz
    U4 B (en sessiz kareler), %60 konuşma doluluğunda renkli gürültünün bant düzeylerini ≤ 1 dB (medyan) bulur
    U5 synth_noise verilen spektrumu ≤ 1 dB (medyan) üretir
    U6 build_track: tam uzunluk, aralık dışından örnek yok, seviye korunur (≤ 0.5 dB), deterministik
    U7 cv_arm: sıfır sinyalde AUC ≈ 0.5, güçlü sinyalde > 0.9; train/test gün kesişimi assert'i çalışır
Uçtan uca (E): 20 sahte katılımcı (.wav), kayıt haritası, önbellek dizini, bağlam, gün-gruplu split'ler →
    extract (envanter, özellikler, izler, kör dinleme) → probe sex (0b) → probe label: D-046 ÖNERİLDİ iken REDDEDİLİR,
    KABUL iken çalışır; raporlar katılımcı ID'si içermez.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import background_extract as BG  # noqa: E402
from make_splits import grouped_folds  # noqa: E402

SR48 = 48000


def db2a(db):
    return 10 ** (db / 20)


def colored(n, rng, level_db):
    w = rng.standard_normal(n + 64)
    x = np.convolve(w, np.ones(8) / 8, mode="same")[:n] + 0.3 * w[:n]       # alçak frekans ağırlıklı
    return x / np.sqrt(np.mean(x ** 2)) * db2a(level_db)


def voice_burst(n, sr, f0=140.0):
    t = np.arange(n) / sr
    v = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 15))
    env = np.hanning(n) ** 0.3
    return v * env / np.sqrt(np.mean(v ** 2))


def synth_recording(sr, seed, floor_db=-60, speech_db=-14, dur=10.0, breath=True, zeros_s=0.0):
    """0.4 s baş sessizliği; 0.45 s konuşma / 0.55 s ara × 9; ara ortalarında 0.12 s nefes (−38 dBFS)."""
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    x = colored(n, rng, floor_db)
    bursts, br = [], []
    t = 0.4
    for _ in range(9):
        a, b = int(t * sr), int((t + 0.45) * sr)
        x[a:b] += voice_burst(b - a, sr) * db2a(speech_db)
        bursts.append((t, t + 0.45))
        if breath:
            c = t + 0.45 + 0.275
            ba, bb = int((c - 0.06) * sr), int((c + 0.06) * sr)
            x[ba:bb] += rng.standard_normal(bb - ba) * db2a(-38)
            br.append((c - 0.06, c + 0.06))
        t += 1.0
    if zeros_s:
        x[int(0.05 * sr): int((0.05 + zeros_s) * sr)] = 0.0
    return x.astype(np.float32), bursts, br


def to32(x48):
    y, _ = BG.render(BG.to_source_sr(x48, SR48), {"trim_start_s": 0.0, "trim_end_s": len(x48) / SR48}, BG.SR)
    return y


def sec_intervals(mask, v):
    return [(a / BG.SR, b / BG.SR) for a, b in BG.intervals(mask, BG.MIN_RUN_S[v])]


def overlap(iv, spans, margin=0.0):
    return any(a < e + margin and s - margin < b for a, b in iv for s, e in spans)


def test_masks():
    x, bursts, br = synth_recording(SR48, 1)
    y = to32(x)
    M = BG.frame_masks(y, 0.0, len(y) / BG.SR)
    a1, a2 = sec_intervals(M["A1"], "A1"), sec_intervals(M["A2"], "A2")
    assert a1 and a2, "arka plan bulunamadı"
    assert not overlap(a1, bursts, BG.GUARD_S - 0.011), "U1: A1 konuşmaya koruma payından yakın"
    assert not overlap(a2, bursts, BG.GUARD_S - 0.011), "U1: A2 konuşmaya koruma payından yakın"
    assert not overlap(a1, br), "U1: nefes A1'e girdi"
    assert overlap(a2, br), "U1: nefes A2'de yok"
    assert (M["A1"] <= M["A2"]).all(), "U1: A1 ⊄ A2"
    assert M["bg_valid"] and M["snr_db"] > 30
    print(f"U1 ✓ A1 {sum(b - a for a, b in a1):.2f} s, A2 {sum(b - a for a, b in a2):.2f} s, SNR {M['snr_db']:.1f} dB")


def test_digital():
    x0, _, _ = synth_recording(SR48, 2)
    x1, _, _ = synth_recording(SR48, 2, zeros_s=0.3)
    M0 = BG.frame_masks(to32(x0), 0.0, 10.0)
    M1 = BG.frame_masks(to32(x1), 0.0, 10.0)
    assert M1["n_digital"] > 0 and M0["n_digital"] == 0, "U2: sayısal sessizlik sayılmadı"
    assert abs(M1["floor_db"] - M0["floor_db"]) < 1.0, "U2: sıfırlar tabanı bozdu"
    print(f"U2 ✓ sayısal sessizlik karesi {M1['n_digital']}, taban farkı {abs(M1['floor_db'] - M0['floor_db']):.2f} dB")


def test_low_snr():
    x, _, _ = synth_recording(SR48, 3, floor_db=-50, speech_db=-38, breath=False)
    M = BG.frame_masks(to32(x), 0.0, 10.0)
    assert not M["bg_valid"] and not M["A2"].any() and not M["A1"].any(), "U3: düşük SNR'de arka plan alındı"
    print(f"U3 ✓ SNR {M['snr_db']:.1f} dB → A1 / A2 alınmadı")


def test_psd_b():
    rng = np.random.default_rng(4)
    n = BG.SR * 12
    noise = colored(n, rng, -55)
    x = noise.copy()
    for t in np.arange(0.2, 11.5, 1.0):                      # %60 doluluk
        a, b = int(t * BG.SR), int((t + 0.6) * BG.SR)
        x[a:b] += voice_burst(b - a, BG.SR) * db2a(-15)
    truth = BG.band_db(BG.stft_power(noise).mean(0))
    est = BG.band_db(BG.b_psd(x.astype(np.float32), 0.0, 12.0))
    est0 = BG.band_db(BG.b_psd(noise.astype(np.float32), 0.0, 12.0))
    err, err0 = np.median(np.abs(est - truth)), np.median(np.abs(est0 - truth))
    assert err <= 1.0 and err0 <= 0.6, f"U4: B bant hatası {err:.2f} dB (saf gürültüde {err0:.2f})"
    print(f"U4 ✓ B medyan bant hatası %60 konuşmada {err:.2f} dB, saf gürültüde {err0:.2f} dB")
    return BG.stft_power(noise).mean(0)


def test_synth(psd):
    x = BG.synth_noise(psd, BG.SR * 10, np.random.default_rng(5))
    err = np.median(np.abs(BG.band_db(BG.stft_power(x).mean(0)) - BG.band_db(psd)))
    assert err <= 1.0, f"U5: sentez bant hatası {err:.2f} dB"
    print(f"U5 ✓ sentez medyan bant hatası {err:.2f} dB")


def test_track():
    rng = np.random.default_rng(6)
    src = np.full(BG.SR * 5, 0.9, dtype=np.float32)
    segs = []
    for a in range(0, BG.SR * 5 - 4000, 8000):
        src[a:a + 3000] = rng.standard_normal(3000) * 0.01
        segs.append(src[a:a + 3000])
    n = int(BG.TRACK_S * BG.SR)
    t1 = BG.build_track(segs, n, np.random.default_rng(1))
    t2 = BG.build_track(segs, n, np.random.default_rng(1))
    assert len(t1) == n and np.array_equal(t1, t2), "U6: uzunluk / determinizm"
    assert np.abs(t1).max() < 0.5, "U6: aralık dışından örnek girdi"
    lvl = 10 * np.log10(np.mean(t1.astype(float) ** 2) / np.mean(np.concatenate(segs).astype(float) ** 2))
    assert abs(lvl) < 0.5, f"U6: seviye değişti ({lvl:.2f} dB)"
    print(f"U6 ✓ iz uzunluğu {n}, seviye farkı {lvl:+.2f} dB, deterministik")


def fake_splits(pids, labels, days, n_rep=3):
    out = {}
    for r in range(n_rep):
        f = grouped_folds(labels, days, 5, 100 + r)
        rows = []
        for k in range(5):
            tr = f != k
            inner = np.full(len(pids), -1)
            inner[tr] = grouped_folds(labels[tr], days[tr], 5, 200 + r * 10 + k)
            rows.append(pd.DataFrame({"participant_id": pids, "repeat": r, "outer_fold": k,
                                      "role": np.where(tr, "train", "test"), "inner_fold": inner}))
        out[r] = pd.concat(rows, ignore_index=True)
    return out


def test_cv():
    rng = np.random.default_rng(7)
    n = 200
    pids = np.arange(100001, 100001 + n)
    y = (rng.random(n) < 0.3).astype(int)
    days = np.array([f"2024-01-{1 + i // 4:02d}" if i < 112 else f"2024-02-{1 + (i - 112) // 4:02d}" for i in range(n)])
    C = pd.DataFrame({"recording_date": days}, index=pids)
    S = fake_splits(pids, y, days, 1)
    Xn = pd.DataFrame(rng.standard_normal((n, 5)), index=pids)
    Xs = Xn.copy()
    Xs[0] += 3 * y
    ys = pd.Series(y, index=pids)
    an = BG.summarize(BG.cv_arm(Xn, ys, S, C))["pooled_auc_mean"]
    as_ = BG.summarize(BG.cv_arm(Xs, ys, S, C))["pooled_auc_mean"]
    assert 0.35 < an < 0.65 and as_ > 0.9, f"U7: AUC null {an:.3f}, sinyal {as_:.3f}"
    bad = {0: S[0].copy()}
    C2 = C.copy()
    C2.loc[:, "recording_date"] = "2024-01-01"                        # her gün hem train hem test'te
    try:
        BG.cv_arm(Xn, ys, bad, C2)
        raise AssertionError("U7: gün sızıntısı yakalanmadı")
    except AssertionError as e:
        assert "gün" in str(e), e
    print(f"U7 ✓ null AUC {an:.3f}, sinyal AUC {as_:.3f}, gün sızıntısı assert'i çalışıyor")


def test_e2e():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "audio").mkdir()
        n = 20
        pids = np.arange(101001, 101001 + n)
        labels = np.array([0 if i % 3 == 0 else 1 for i in range(n)])
        sexes = np.array(["M" if i % 2 else "F" for i in range(n)])
        dates = np.array([f"2024-0{1 + i // 10}-{1 + i % 10:02d}" for i in range(n)])
        rm, ci = [], []
        for i, pid in enumerate(pids):
            for slot in (2, 3, 5):
                x, _, _ = synth_recording(SR48, 1000 + 10 * i + slot, floor_db=-60 + 3 * (i % 4), breath=slot == 2)
                rel = f"{pid}_{slot}.wav"
                sf.write(td / "audio" / rel, x, SR48, subtype="FLOAT")
                rm.append({"participant_id": pid, "slot": slot, "status": "OK", "relpath": rel, "sr": SR48, "channels": 1})
                _, g = BG.model_view(td / "audio" / rel, SR48, 1)
                ci.append({"participant_id": pid, "slot": slot, "trim_start_s": 0.3, "trim_end_s": 9.7, "gain_db_32k": g})
        pd.DataFrame(rm).to_csv(td / "rm.csv", index=False)
        pd.DataFrame(ci).to_csv(td / "index.csv", index=False)
        pd.DataFrame({"participant_id": pids, "label": labels, "sex": sexes, "recording_date": dates,
                      "start_hour": 9 + (np.arange(n) % 8), "days": np.arange(n), "age": np.random.default_rng(9).integers(20, 65, n)}).to_csv(td / "ctx.csv", index=False)
        (td / "splits").mkdir()
        for r, s in fake_splits(pids, labels, dates).items():
            s.to_csv(td / "splits" / f"outer_r{r}.csv", index=False)
        py, sc = sys.executable, str(REPO / "scripts" / "background_extract.py")
        run = lambda *a: subprocess.run([py, sc, *map(str, a)], capture_output=True, text=True)
        r = run("extract", "--audio-root", td / "audio", "--recording-map", td / "rm.csv", "--cache-index", td / "index.csv",
                "--context", td / "ctx.csv", "--out-dir", td / "out", "--report-dir", td / "rep", "--n-listen", 3)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
        FT = pd.read_csv(td / "out" / "features.csv")
        assert FT.groupby("participant_id").size().eq(3).all() and FT.included.all(), "E: özellik tablosu"
        TR = pd.read_csv(td / "out" / "tracks_index.csv")
        assert (td / "out" / "tracks_32k.f32").stat().st_size == 4 * TR.n.sum()
        assert len(list((td / "out" / "listening").glob("L*.wav"))) == 9
        md = (td / "rep" / "BG-0a_inventory.md").read_text()
        assert BG.ids_free(md, pids) and "GEÇTİ" in md, "E: 0a raporu"
        r = run("probe", "--features", td / "out" / "features.csv", "--context", td / "ctx.csv", "--splits-dir", td / "splits",
                "--tasks", "sex", "--out-dir", td / "out", "--report-dir", td / "rep")
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
        assert "Kapı G-b" in (td / "rep" / "BG-0b_qc.md").read_text()
        dec = td / "DECISIONS.md"
        dec.write_text("## D-046 — x\nDURUM: **ÖNERİLDİ** (y)\n")
        args = ["probe", "--features", td / "out" / "features.csv", "--context", td / "ctx.csv", "--splits-dir", td / "splits",
                "--tasks", "label", "period", "--decisions", dec, "--out-dir", td / "out", "--report-dir", td / "rep"]
        r = run(*args)
        assert r.returncode != 0 and "KABUL" in r.stderr, "E: 0c kapısı ÖNERİLDİ iken açıldı"
        dec.write_text("## D-046 — x\nDURUM: **KABUL** (y)\n")
        r = run(*args)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
        js = (td / "rep" / "BG-0c_info.json").read_text()
        assert BG.ids_free(js, pids) and "_t1" in js
        print("E ✓ extract → probe sex → 0c kapısı (ÖNERİLDİ reddedildi, KABUL çalıştı); raporlar ID'siz")


if __name__ == "__main__":
    test_masks()
    test_digital()
    test_low_snr()
    test_synth(test_psd_b())
    test_track()
    test_cv()
    test_e2e()
    print("Hepsi geçti.")
