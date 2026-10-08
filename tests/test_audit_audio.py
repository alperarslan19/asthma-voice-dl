"""
audit_audio.py için kendi kendine yeten test (gerçek veri GEREKMEZ).

Sahte bir katılımcı tablosu ve bilerek hatalar konmuş sahte ses dosyaları üretir, script'i çalıştırır
ve her hatanın yakalandığını assert eder. Script değiştiğinde yeniden çalıştır:

    python tests/test_audit_audio.py

Konulan tuzaklar ve beklenen sonuç:
    T1 eksik dosya (P05 slot 4)                         -> MISSING
    T2 aynı slot için hem .mp4 hem .m4a (P07 slot 2)     -> MULTIPLE
    T3 video akışlı .mp4 (P10)                          -> çözülür, n_video_streams=1
    T4 kalıba uymayan adlar (slot 8, IMG_0001, notes.txt) -> unmatched / non-audio
    T5 dosyası hiç olmayan katılımcı                     -> participants_with_zero_files
    T6 dosya tarihi CSV'den 2 gün farklı (P03)           -> frac_same_day < 1
    T7 CSV tarihi yok, dosyada var (P11)                 -> sayılır
    T8 konum etiketi (P01 slot 1)                        -> has_location_tag + gizlilik uyarısı
    T9 geç dönemde farklı yazılım sürümü + yüksek gürültü -> meta veri ve gürültü tabanı dönemle ilişkili
    T10 geç dönemde düşük bitrate (daha dar AAC bant genişliği) -> bitrate ve bant genişliği dönemle ilişkili
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
SR = 48000


def make_signal(seed: int, kind: str, noise: float) -> np.ndarray:
    r = np.random.default_rng(seed)
    t = np.arange(int(10 * SR)) / SR
    f0 = r.uniform(100, 250) * (1 + 0.03 * np.sin(2 * np.pi * r.uniform(3, 6) * t))
    x = sum(np.sin(2 * np.pi * np.cumsum(f0 * k) / SR) / k * r.uniform(0.3, 1) for k in range(1, 8))
    if kind == "word":
        x *= np.sin(2 * np.pi * r.uniform(0.8, 1.2) * t + r.uniform(0, 6.28)) > 0
    x = 0.3 * x / np.abs(x).max()
    x[: int(0.5 * SR)] = 0
    return (x + noise * r.standard_normal(len(t))).astype(np.float32)


def encode(x: np.ndarray, out: Path, meta: dict, video: bool = False, bitrate: str = "96k") -> None:
    tmp = out.with_suffix(".tmp.wav")
    sf.write(tmp, x, SR, subtype="PCM_16")
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if video:
        cmd += ["-f", "lavfi", "-i", "color=c=black:s=64x64:d=10"]
    cmd += ["-i", str(tmp)]
    if video:
        cmd += ["-map", "0:v", "-map", "1:a", "-c:v", "mpeg4"]
    cmd += ["-c:a", "aac", "-b:a", bitrate, "-movflags", "use_metadata_tags"]
    for k, v in meta.items():
        cmd += ["-metadata", f"{k}={v}"]
    subprocess.run(cmd + [str(out)], check=True)
    tmp.unlink()


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="audit_test_"))
    audio = work / "soundData"
    audio.mkdir()
    # Sahte katılımcılar: 1..30 erken dönem (karışık etiket), 31..44 geç dönem hastalar, 45 dosyasız
    rows = []
    for i in range(1, 46):
        pid = 900000 + i
        early = i <= 30
        label = (i % 3 != 0) if early else True
        day = (pd.Timestamp("2024-02-01") + pd.Timedelta(days=i % 20)) if early else (pd.Timestamp("2024-06-01") + pd.Timedelta(days=i % 20))
        rows.append({"participant_id": pid, "label": int(label), "in_paper_cohort": True,
                     "period": "early" if early else "late", "collection_day": str(day.date())})
    P = pd.DataFrame(rows)
    P.loc[P.participant_id == 900011, ["collection_day", "period"]] = [np.nan, "unknown"]   # T7
    P.to_csv(work / "participants.csv", index=False)

    for _, r in P.iterrows():
        pid, i = int(r.participant_id), int(r.participant_id) - 900000
        if i == 45:                                   # T5: hiç dosya yok
            continue
        late = r.period == "late"
        day = pd.Timestamp(r.collection_day) if isinstance(r.collection_day, str) else pd.Timestamp("2024-02-03")
        if i == 3:
            day += pd.Timedelta(days=2)               # T6
        for s in range(1, 8):
            if i == 5 and s == 4:                     # T1
                continue
            meta = {"creation_time": f"{day.date()}T07:{s:02d}:00Z",
                    "com.apple.quicktime.make": "Apple", "com.apple.quicktime.model": "iPhone 14",
                    "com.apple.quicktime.software": "17.5" if late else "17.3"}  # T9
            if i == 1 and s == 1:
                meta["location"] = "+41.0000+028.9000/"   # T8
            x = make_signal(pid * 10 + s, "vowel" if s == 1 else "word", noise=0.01 if late else 0.002)  # T9
            ext = ".mp4" if i % 2 else ".m4a"
            encode(x, audio / f"{pid}_{s}{ext}", meta, video=(i == 10), bitrate="48k" if late else "96k")  # T3, T10
            if i == 7 and s == 2:                     # T2
                encode(x, audio / f"{pid}_{s}.m4a" if ext == ".mp4" else audio / f"{pid}_{s}.mp4", meta)
    encode(make_signal(1, "word", 0.002), audio / "900003_8.mp4", {})   # T4
    encode(make_signal(2, "word", 0.002), audio / "IMG_0001.m4a", {})   # T4
    (audio / "notes.txt").write_text("not audio")                        # T4

    r = subprocess.run([sys.executable, str(REPO / "scripts" / "audit_audio.py"), "--audio-root", str(audio),
                        "--participants", str(work / "participants.csv"), "--out-private", str(work / "priv"),
                        "--out-report", str(work / "rep")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    R = json.loads((work / "rep" / "audio_audit.json").read_text())
    M = pd.read_csv(work / "priv" / "recording_map.csv")
    st = M.set_index(["participant_id", "slot"])["status"]
    mp = R["mapping"]

    assert st[(900005, 4)] == "MISSING", "T1"
    assert st[(900007, 2)] == "MULTIPLE", "T2"
    assert R["formats"]["n_video_streams"].get("1", 0) == 7 and R["formats"]["n_decode_errors"] == 0, "T3"
    assert mp["n_unmatched_audio_files"] == 2 and R["naming"]["n_slot_out_of_range"] == 1, "T4"
    assert R["non_audio_files"] == ["notes.txt"], "T4b"
    assert mp["participants_with_zero_files"] == [900045], "T5"
    assert mp["participants_with_all_7_ok"] == 42, mp  # 44 dosyalı - P05 (eksik) - P07 (çoklu)
    ct = R["creation_time_vs_csv_date"]
    assert 0.9 < ct["frac_same_day"] < 1.0, "T6"
    assert ct["n_participants_csv_date_missing_but_file_has_date"] == 1, "T7"
    assert R["formats"]["has_location_tag"].get("True") == 1 and "privacy_warning" in R, "T8"
    sw = R["metadata_vs_label_and_period"]["tag_apple_software"]
    assert not sw["uniform"] and sw["period_chi2_p"] < 1e-6, "T9 meta"
    top_period = pd.DataFrame(R["nuisance_top_period"])
    nf = top_period[top_period.feature == "noise_floor_db"]
    assert len(nf) and (nf["auc_period_within_patients"] - 0.5).abs().min() > 0.45, "T9 gürültü"
    assert R["metadata_vs_label_and_period"]["tag_apple_model"]["uniform"], "model tekdüze olmalı"
    inv = pd.read_csv(work / "priv" / "audio_inventory.csv")
    bw = inv.assign(late=inv["participant_id"].isin(P.loc[P.period == "late", "participant_id"]))
    bw_e, bw_l = bw[~bw.late]["bandwidth_khz"].median(), bw[bw.late]["bandwidth_khz"].median()
    print(f"T10 bant genişliği medyanı: erken {bw_e:.1f} kHz, geç {bw_l:.1f} kHz")
    assert bw_l < bw_e - 1, "T10: düşük bitrate daha dar bant genişliği vermeli"
    # Konum değeri HİÇBİR çıktıda olmamalı
    for f in list((work / "priv").glob("*")) + list((work / "rep").glob("*")):
        assert "028.9" not in f.read_text(errors="ignore"), f"konum değeri sızdı: {f.name}"
    print("TÜM TESTLER GEÇTİ ✔  (geçici klasör:", work, ")")
    if not os.environ.get("KEEP_TEST_OUTPUT"):  # incelemek için: KEEP_TEST_OUTPUT=1 python tests/test_audit_audio.py
        shutil.rmtree(work)


if __name__ == "__main__":
    main()
