"""
EVAL-016 post hoc (scripts/posthoc_eval016.py) için testler. Gerçek veri GEREKMEZ.

    python tests/test_posthoc_eval016.py

    E18 PH3'ün koşullu logit fit'i (clogit_firth_v2) bağımsız R referansıyla (survival::coxph "exact" + optim; E13 ile
        aynı referanslar) aynı ve "yakınsadı" diyor; biri tabaka içinde tam ayrışmalı
    E19 PH1 / PH2 bağlam matrisleri: tam ranklı; birincil bağlam matrisinin sütun uzayını kapsıyor (PH2 yalnız ekliyor);
        PH2'de cinsiyet 1, sigara 2 sütun
    E20 PH0 bütünlük kapısı: skor dosyası Aşama 1 raporuyla uyuşmazsa script durur
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tests"))
import posthoc_eval016 as PH  # noqa: E402
import test_eval016 as T  # noqa: E402


def test_clogit_v2_reference() -> None:
    for (seed, strong), ref in T.CL_REF.items():
        d = T._cl_data(seed, strong)
        r = PH.clogit_firth_v2(d.y.to_numpy(), d[["s", "x2", "x3", "x4"]].to_numpy(float), d.g.to_numpy())
        assert r["converged"] and np.allclose(r["coef"], ref, atol=1e-4), (seed, r["coef"], ref)
    print("  E18 geçti (PH3 koşullu logit = R survival referansı; yakınsadı)", flush=True)


def _ctx(rng) -> pd.DataFrame:
    C = T.cohort(rng).dropna(subset=["start_hour"]).copy()
    C["sex"] = np.where(rng.random(len(C)) < 0.68, "F", "M")
    C["smoking"] = rng.choice(["current", "former", "never"], len(C), p=[0.35, 0.13, 0.52])
    return C


def test_designs() -> None:
    C = _ctx(np.random.default_rng(5))
    X0, X1, X2 = (PH.design(C, k) for k in ("primary", "age_spline", "demog"))
    for X in (X0, X1, X2):
        assert np.linalg.matrix_rank(X) == X.shape[1]
    assert X2.shape[1] == X0.shape[1] + 3, (X0.shape, X2.shape)
    proj = X2 @ np.linalg.lstsq(X2, X0, rcond=None)[0]
    assert np.allclose(proj, X0, atol=1e-8), "PH2 birincil bağlamı kapsamıyor"
    assert X1.shape[1] > X0.shape[1]
    print(f"  E19 geçti (sütun: birincil {X0.shape[1]}, PH1 {X1.shape[1]}, PH2 {X2.shape[1]})", flush=True)


def test_ph0_gate() -> None:
    """Uyuşmayan skor dosyası (bir kolun skoru karıştırılmış) → PH0 assert'i çalışmayı durdurur."""
    import evaluate_context as E
    rng = np.random.default_rng(7)
    C = _ctx(rng)
    C.loc[:, "in_time_overlap"] = C.days.between(C.days[C.label == 0].min(), C.days[C.label == 0].max())
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        C.to_csv(td / "ctx.csv", index=False)
        _, Ct, _ = E.load_context_table(td / "ctx.csv", set(C.participant_id))
        y = Ct.label.to_numpy()
        day = Ct.recording_date.astype("category").cat.codes.to_numpy()
        s = pd.DataFrame({"cnn10": y + rng.normal(0, 1, len(y))}, index=Ct.index)
        s["label"] = y
        X = E.context_matrix(Ct, 4)
        orig = {"arms": ["cnn10"], "backbones": ["cnn10"], "t1": {"cnn10": E.t1(y, X, s.cnn10.to_numpy(), groups=day, ml=False)},
                "verdicts": {"cnn10": {"stage1_all_met": True}}, "e2h": {"cnn10": {"or_sd": 2.0, "converged": True}}}
        (td / "orig.json").write_text(json.dumps(orig, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
        bad = s.copy()
        bad["cnn10"] = rng.permutation(bad.cnn10.to_numpy())
        bad.to_csv(td / "scores.csv")
        r = subprocess.run([sys.executable, str(REPO / "scripts/posthoc_eval016.py"), "--scores", str(td / "scores.csv"),
                            "--context", str(td / "ctx.csv"), "--orig-report", str(td / "orig.json"), "--report-dir", str(td / "rep")],
                           capture_output=True, text=True)
        assert r.returncode != 0 and "PH0" in (r.stdout + r.stderr), (r.stdout[-1500:], r.stderr[-1500:])
        assert not (td / "rep").exists()
    print("  E20 geçti (uyuşmayan skor dosyası reddedildi)", flush=True)


if __name__ == "__main__":
    test_designs()
    test_ph0_gate()
    test_clogit_v2_reference()
    print("Tüm post hoc kontrolleri geçti (E18–E20). Not: sentetik veri; gerçek veride sonuç üretilmedi.")
