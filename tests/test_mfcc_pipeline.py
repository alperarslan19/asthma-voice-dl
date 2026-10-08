"""
MFCC hattı için kendi kendine yeten test (gerçek veri GEREKMEZ; ffmpeg gerekir):
extract_mfcc_features.py (iki mod) + run_mfcc_baselines.py (EXP-010 ve EXP-011).

    python tests/test_mfcc_pipeline.py

Kontroller:
    K1 özellik çıkarımı iki modda da 44 sonlu değer üretir (sahte .m4a ve sahte önbellek)
    K2 EXP-011: yerleştirilmiş güçlü sinyalde AUC > 0.8, saf gürültüde AUC ≈ 0.5 (0.35–0.65)
    K3 EXP-011: referans çizgileri (yaş, bağlam) aynı fold'larda hesaplanır; yaş sinyali yerleştirildi → yaş AUC > 0.6
    K4 EXP-011: füzyon satırları var; her tekrarda her katılımcı tam bir kez test tahmini alır
    K5 EXP-010: makale kuralıyla "en iyi" model seçilir ve rapora yazılır
    K6 kaldığı yerden devam: ikinci çalıştırma partial/ sonuçlarını kullanır, aynı özet
    K7 raporlar katılımcı ID'si içermez; registry'ye satır eklenir
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
import test_build_audio_cache as tbc  # noqa: E402  (sahte .m4a üretici)

FEATS = None


def run(*cmd) -> subprocess.CompletedProcess:
    r = subprocess.run([sys.executable, *map(str, cmd)], capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout[-3000:] + r.stderr[-3000:])
    return r


def test_extraction(work: Path) -> None:
    root = work / "audio"
    (root / "soundData").mkdir(parents=True)
    rows = []
    for i, (pid, slot) in enumerate([(900001, 1), (900001, 2)]):
        rel = f"soundData/{pid}_{slot}.m4a"
        tbc.make_file(root / rel, 48000, i, 0.5, 0.5, i == 0, "128k")
        rows.append({"participant_id": pid, "slot": slot, "status": "OK", "relpath": rel, "file_sha256": tbc.sha(root / rel),
                     "sr": 48000.0, "channels": 1, "tag_encoder": "Lavf"})
    pd.DataFrame(rows).to_csv(work / "recording_map.csv", index=False)
    run(REPO / "scripts/build_audio_cache.py", "--audio-root", root, "--recording-map", work / "recording_map.csv",
        "--out-dir", work / "cache", "--report-dir", work / "cache_report")
    for mode, extra in [("paper", ["--audio-root", root, "--recording-map", work / "recording_map.csv"]),
                        ("cache", ["--cache-dir", work / "cache"])]:
        out = work / f"feat_{mode}.csv"
        run(REPO / "scripts/extract_mfcc_features.py", "--mode", mode, *extra, "--out-csv", out)
        f = pd.read_csv(out)
        cols = [c for c in f.columns if c not in {"participant_id", "slot", "kept_s"}]
        assert f.shape[0] == 2 and len(cols) == 44 and np.isfinite(f[cols].to_numpy()).all(), f"K1 {mode}"
        if mode == "paper":
            assert (f.kept_s < 7.0).all(), "K1: paper modunda kırpma çalışmadı"


def synthetic_cohort(work: Path, rng) -> tuple[Path, Path, Path]:
    n1, n0 = 120, 40
    pids = np.arange(800001, 800001 + n1 + n0)
    label = np.r_[np.ones(n1, int), np.zeros(n0, int)]
    age = np.where(label == 1, rng.normal(48, 9, len(pids)), rng.normal(38, 9, len(pids)))   # yaş sinyali
    days = rng.integers(0, 80, len(pids))
    ctx = pd.DataFrame({"participant_id": pids, "label": label, "age": age.round(),
                        "recording_date": [str((pd.Timestamp("2024-01-10") + pd.Timedelta(days=int(d))).date()) for d in days],
                        "start_hour": rng.uniform(9, 17, len(pids)), "has_audio": True})
    ctx["days"] = days
    rm = pd.DataFrame([{"participant_id": p, "slot": s, "status": "OK"} for p in pids for s in range(1, 8)])
    feats = []
    for p, y in zip(pids, label):
        for s in range(1, 8):
            v = rng.normal(0, 1, 44)
            v[0] += 1.6 * y                       # güçlü sinyal: f00 (her görevde)
            feats.append([p, s, 10.0, *v])
    cols = ["participant_id", "slot", "kept_s"] + [f"f{i:02d}" for i in range(44)]
    F = pd.DataFrame(feats, columns=cols)
    Fn = F.copy()
    Fn[cols[3:]] = rng.normal(0, 1, (len(F), 44))   # saf gürültü
    paths = work / "ctx.csv", work / "feat_signal.csv", work / "feat_noise.csv"
    ctx.to_csv(paths[0], index=False)
    F.to_csv(paths[1], index=False)
    Fn.to_csv(paths[2], index=False)
    rm.to_csv(work / "rm_syn.csv", index=False)
    run(REPO / "scripts/make_splits.py", "--context", paths[0], "--recording-map", work / "rm_syn.csv",
        "--out-dir", work / "splits", "--report-dir", work / "splits_report", "--n-repeats", "2")
    return paths


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="test_mfcc_"))
    try:
        test_extraction(work)
        ctx, fsig, fnoise = synthetic_cohort(work, np.random.default_rng(0))
        reg = work / "registry.csv"
        pd.read_csv(REPO / "results/registry.csv", nrows=0).to_csv(reg, index=False)

        common = ["--context", ctx, "--splits-dir", work / "splits", "--n-repeats", "2", "--models", "LR", "--registry", reg]
        run(REPO / "scripts/run_mfcc_baselines.py", "--exp", "EXP-011", "--features", fsig,
            "--out-dir", work / "e11", "--report-dir", work / "rep_sig", *common)
        s = pd.DataFrame(json.loads((work / "rep_sig" / "EXP-011_mfcc.json").read_text())["summary"])
        lr = s[(s.model == "LR") & (s.task == "aaa")].iloc[0]
        assert lr.auc_mean > 0.8, f"K2: sinyal AUC {lr.auc_mean:.3f}"
        age = s[(s.task == "reference") & (s.model == "age")].iloc[0]
        assert age.auc_mean > 0.6, f"K3: yaş AUC {age.auc_mean:.3f}"
        assert (s.task == "fusion (7 görev)").any(), "K4: füzyon yok"
        oof = pd.read_csv(work / "e11" / "oof_predictions.csv")
        for (r, t), g in oof[(oof.model == "LR") & (oof.task == "aaa")].groupby(["repeat", "task"]):
            assert g.participant_id.is_unique and len(g) == 160, f"K4: r{r} {len(g)} test tahmini"
        # K6: kaldığı yerden devam
        r2 = run(REPO / "scripts/run_mfcc_baselines.py", "--exp", "EXP-011", "--features", fsig,
                 "--out-dir", work / "e11", "--report-dir", work / "rep_sig2", *common)
        assert "önceki sonuç kullanıldı" in r2.stdout, "K6"
        s2 = pd.DataFrame(json.loads((work / "rep_sig2" / "EXP-011_mfcc.json").read_text())["summary"])
        assert np.allclose(s.auc_mean, s2.auc_mean), "K6: devam eden çalıştırma farklı sonuç verdi"

        run(REPO / "scripts/run_mfcc_baselines.py", "--exp", "EXP-011", "--features", fnoise,
            "--out-dir", work / "e11n", "--report-dir", work / "rep_noise", *common)
        sn = pd.DataFrame(json.loads((work / "rep_noise" / "EXP-011_mfcc.json").read_text())["summary"])
        nz = sn[(sn.model == "LR") & (sn.task == "aaa")].iloc[0]
        assert 0.35 < nz.auc_mean < 0.65, f"K2: gürültü AUC {nz.auc_mean:.3f}"

        run(REPO / "scripts/run_mfcc_baselines.py", "--exp", "EXP-010", "--features", fsig, "--context", ctx,
            "--out-dir", work / "e10", "--report-dir", work / "rep10", "--models", "LogisticRegression,NaiveBayes,KNN",
            "--registry", reg)
        r10 = json.loads((work / "rep10" / "EXP-010_mfcc.json").read_text())
        assert len(r10["paper_rule_best_per_task"]) == 7, "K5"

        txt = "".join(p.read_text() for p in work.glob("rep*/*"))
        assert not any(str(p) in txt for p in range(800001, 800161)), "K7: raporda katılımcı ID'si"
        assert len(pd.read_csv(reg)) > 0, "K7: registry boş"
        print("Tüm kontroller geçti (K1–K7).")
        print(f"  sinyal AUC {lr.auc_mean:.3f} | gürültü AUC {nz.auc_mean:.3f} | yaş referansı {age.auc_mean:.3f}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
