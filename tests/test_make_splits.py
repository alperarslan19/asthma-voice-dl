"""
make_splits.py için kendi kendine yeten test (gerçek veri GEREKMEZ).

Sahte bir katılımcı tablosu ve kayıt eşlemesi üretir, script'i çalıştırır ve her kuralın tuttuğunu
assert eder. Script değiştiğinde yeniden çalıştır:

    python tests/test_make_splits.py

Kontroller ve konulan tuzaklar:
    K1 sesi olmayan katılımcı (P_NOAUDIO)               -> kohortta yok
    K2 her tekrarda her katılımcı tam bir kez testte     -> evet
    K3 train ∩ test = ∅, iç fold'lar train'i bölüyor     -> evet
    K4 aynı girdi iki kez çalıştırılınca                  -> aynı sha256 (belirlenimci)
    K5 yaşı / tarihi bilinmeyen tek kişilik tabakalar     -> birleştirilir, manifestte görünür
    K6 bir görevde kaydı eksik katılımcı                   -> kohortta kalır, o görevin sayısı 1 eksik
    K7 manifest katılımcı ID'si içermiyor                  -> evet
    T1 tekrarlanan katılımcı ID'si                         -> script hata verir
    T2 has_audio ile recording_map uyuşmuyor               -> script hata verir
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "make_splits.py"


def fake_data(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    pid = 900001
    # 120 astım (erken/geç), 30 sağlıklı (yalnız erken) — gerçek verinin oranlarına benzer
    for label, n in [(1, 120), (0, 30)]:
        for _ in range(n):
            late = label == 1 and rng.random() < 0.6
            day = pd.Timestamp("2024-05-15") + pd.Timedelta(days=int(rng.integers(0, 60))) if late else \
                pd.Timestamp("2024-01-10") + pd.Timedelta(days=int(rng.integers(0, 80)))
            rows.append({"participant_id": pid, "label": label, "age": float(rng.integers(20, 75)),
                         "recording_date": str(day.date()), "start_hour": float(rng.uniform(9, 17)), "has_audio": True})
            pid += 1
    ctx = pd.DataFrame(rows)
    healthy = ctx.index[ctx["label"] == 0]
    ctx.loc[healthy[0], "age"] = np.nan                                   # K5: yaşı bilinmeyen sağlıklı
    ctx.loc[healthy[1], ["age", "recording_date"]] = [np.nan, np.nan]     # K5: tarihi ve yaşı bilinmeyen sağlıklı
    noaudio = {"participant_id": 999999, "label": 0, "age": 30.0, "recording_date": "2024-02-01",
               "start_hour": np.nan, "has_audio": False}                  # K1
    ctx = pd.concat([ctx, pd.DataFrame([noaudio])], ignore_index=True)
    rm = [{"participant_id": p, "slot": s, "status": "OK"} for p in ctx.loc[ctx.has_audio, "participant_id"] for s in range(1, 8)]
    rm += [{"participant_id": 999999, "slot": s, "status": "MISSING"} for s in range(1, 8)]
    rm = pd.DataFrame(rm)
    rm.loc[(rm.participant_id == 900005) & (rm.slot == 4), "status"] = "MISSING"   # K6
    return ctx, rm


def run(ctx: pd.DataFrame, rm: pd.DataFrame, work: Path, tag: str) -> subprocess.CompletedProcess:
    d = work / tag
    d.mkdir()
    ctx.to_csv(d / "participant_context.csv", index=False)
    rm.to_csv(d / "recording_map.csv", index=False)
    return subprocess.run([sys.executable, str(SCRIPT), "--context", str(d / "participant_context.csv"),
                           "--recording-map", str(d / "recording_map.csv"), "--out-dir", str(d / "splits"),
                           "--report-dir", str(d / "report"), "--n-repeats", "3"], capture_output=True, text=True)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="test_make_splits_"))
    try:
        ctx, rm = fake_data(np.random.default_rng(0))
        a = run(ctx, rm, work, "a")
        assert a.returncode == 0, a.stdout + a.stderr
        b = run(ctx, rm, work, "b")
        assert b.returncode == 0, b.stdout + b.stderr
        cohort = set(ctx.loc[ctx.has_audio, "participant_id"])

        for r in range(3):
            fa, fb = work / "a" / "splits" / f"outer_r{r}.csv", work / "b" / "splits" / f"outer_r{r}.csv"
            assert sha(fa) == sha(fb), f"K4: r={r} belirlenimci değil"
            s = pd.read_csv(fa)
            assert 999999 not in set(s.participant_id), "K1: sesi olmayan katılımcı kohortta"
            assert set(s.participant_id) == cohort
            t = s[s.role == "test"]
            assert t.participant_id.is_unique and set(t.participant_id) == cohort, "K2"
            for k, blk in s.groupby("outer_fold"):
                tr, te = set(blk[blk.role == "train"].participant_id), set(blk[blk.role == "test"].participant_id)
                assert not tr & te and tr | te == cohort, f"K3: r={r} k={k}"
                assert set(blk[blk.role == "train"].inner_fold) == set(range(5)), f"K3: r={r} k={k} iç fold eksik"
                assert (blk[blk.role == "test"].inner_fold == -1).all()
                assert not blk[blk.role == "test"].is_inner_val.any()
            # Tekrarlar birbirinden farklı olmalı (aynı tohum hatası)
            if r > 0:
                prev = pd.read_csv(work / "a" / "splits" / f"outer_r{r-1}.csv")
                assert not t.set_index("participant_id").outer_fold.sort_index().equals(
                    prev[prev.role == "test"].set_index("participant_id").outer_fold.sort_index()), "tekrarlar aynı"

        man = (work / "a" / "report" / "splits_manifest.json").read_text()
        import json
        m = json.loads(man)
        assert m["cohort"]["n"] == len(cohort)
        assert m["cohort"]["per_task_n"]["slot4"] == len(cohort) - 1, "K6"
        merged = {x["stratum"] for x in m["merged_sparse_strata"]}
        assert "0|early|unknown" in merged and "0|unknown|unknown" in merged, f"K5: {merged}"
        for p in cohort:
            assert str(p) not in man, "K7: manifestte katılımcı ID'si var"

        # T1: tekrarlanan ID
        bad = pd.concat([ctx, ctx.iloc[[0]]], ignore_index=True)
        r1 = run(bad, rm, work, "t1")
        assert r1.returncode != 0 and "tekrarlanan" in r1.stderr, "T1 yakalanmadı"
        # T2: has_audio True ama hiç OK kaydı yok
        rm2 = rm.copy()
        rm2.loc[rm2.participant_id == 900010, "status"] = "MISSING"
        r2 = run(ctx, rm2, work, "t2")
        assert r2.returncode != 0 and "uyuşmuyor" in r2.stderr, "T2 yakalanmadı"
        print("Tüm kontroller geçti (K1–K7, T1–T2).")
        print(a.stdout.strip().splitlines()[0])
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
