"""
Faz 3 değerlendirmesi (docs/PHASE3_DESIGN.md Bölüm 5). Şu an: `frozen-dg` (G1). Fine-tune değerlendirmesi (E1, T1, C3)
eğitim bitmeden eklenecek; tanımları tasarım belgesinde sonuçlardan önce sabit.

frozen-dg — G1: "gün parmak izi" kontrolü (D-041)
    EXP-016'nın AYNI probu gün-gruplu split'lerde (EXP-016DG; 3 tekrar) → katılımcı ŝ_DG (tekrarların dış-test füzyon
    logit ortalaması) → EVAL-016 ile AYNI T1 (Firth PLR; cr(saat, 4) + cr(tarih, 4) + yaş), gün-kümeli CV3, katılımcı
    bootstrap OR (B 2000). Okuma kuralı (önceden yazıldı, PHASE3_DESIGN 5.1): iddiadaki backbone'lar (cnn10, beats,
    wavlm_large; ayrıca sınırda cnn14) için ŝ_DG ile T1-PLR p < 0.05 ve OR > 1 ise "gün parmak izi yoluyla açıklanmıyor";
    değilse o backbone için iddia "gün / seans artefaktı dışlanamıyor"a düşer. Holm yalnız betimsel.
    E1 (fold AUC ort ± SD) betimsel; katılımcı-gruplu Faz 2 değerleriyle yan yana (farklı split → doğrudan test yok).

ÇIKTI  reports/phase3/G1_frozen_daygrouped.{md,json} (agrega) · Drive: g1_participant_scores.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_context as E  # noqa: E402
import run_probes as RP  # noqa: E402

CLAIM = ["cnn10", "beats", "wavlm_large"]       # docs/EVAL016_CONCLUSION.md: desteklenen iddia
BORDER = ["cnn14"]


def fold_aucs(rec: pd.DataFrame, arm: str, labels: pd.Series) -> np.ndarray:
    d = rec[rec.arm == arm]
    out = []
    for (r, k), g in d.groupby(["repeat", "fold"]):
        f = g.groupby("participant_id").prob.mean()
        out.append(roc_auc_score(labels[f.index], f))
    return np.asarray(out)


def cmd_frozen_dg(a) -> None:
    t0 = time.time()
    splits = {r: pd.read_csv(a.splits_dir / f"outer_r{r}.csv") for r in range(a.n_repeats)}
    cohort = set(splits[0].participant_id)
    bbs = [b for b in a.backbones.split(",") if b]
    assert set(bbs) <= set(RP.BACKBONES)
    arms = bbs + ["mfcc_lr", "mfcc_mlp"]
    S_hat, per_rep, rec, _ = E.load_scores(a.probe_dir, arms, splits, a.splits_dir, a.context)
    C, Ct, cinfo = E.load_context_table(a.context, cohort)
    y = Ct.label.to_numpy()
    day = Ct.recording_date.astype("category").cat.codes.to_numpy()
    X = E.context_matrix(Ct, 4)
    S = S_hat.reindex(Ct.index)
    assert S.notna().all().all()
    stage1 = json.loads(a.stage1_report.read_text()) if a.stage1_report and a.stage1_report.exists() else None
    labels = C.label
    R = {"exp": "G1 (EXP-016DG → T1)", "date": time.strftime("%Y-%m-%d"), "cohort": cinfo, "n_repeats": a.n_repeats,
         "n_boot": a.n_boot, "rule": "ŝ_DG ile T1-PLR p < 0.05 ve OR > 1 → gün parmak izi yoluyla açıklanmıyor (PHASE3_DESIGN 5.1)",
         "t1": {}, "boot": {}, "e1": {}}
    for arm in arms:
        R["t1"][arm] = E.t1(y, X, S[arm].to_numpy(), groups=day, ml=False, jackknife=True)
        R["boot"][arm] = E.boot_or(y, X, S[arm].to_numpy(), a.n_boot, a.seed)
        fa = fold_aucs(rec, arm, labels)
        R["e1"][arm] = {"fold_auc_mean": float(fa.mean()), "fold_auc_sd": float(fa.std(ddof=1)), "n_folds": int(len(fa))}
        print(f"  {arm}: p {R['t1'][arm]['p_lrt']:.4g} OR {R['t1'][arm]['or_sd']:.2f} CV3 {R['t1'][arm]['p_cv3']:.3g} "
              f"E1 {fa.mean():.3f}", flush=True)
    R["holm_descriptive"] = RP.holm({b: R["t1"][b]["p_lrt"] for b in bbs})
    R["verdict"] = {}
    for b in [x for x in CLAIM + BORDER if x in bbs]:
        t = R["t1"][b]
        held = bool(t["p_lrt"] < 0.05 and t["or_sd"] > 1)
        R["verdict"][b] = ("gün parmak izi yoluyla açıklanmıyor" if held else "gün / seans artefaktı dışlanamıyor")
    R["elapsed_s"] = round(time.time() - t0, 1)
    f3, fp = E.f3, E.fp
    L = ["# G1 — dondurulmuş problar gün-gruplu split'lerde (EXP-016DG) ve T1 (otomatik üretildi; katılımcı ID'si içermez)", "",
         f"**Okuma kuralı (önceden yazıldı):** {R['rule']}. CV3 ve Holm betimsel. Kohort {cinfo['n_t1']} "
         f"({cinfo['n_t1_asthma']} / {cinfo['n_t1_healthy']}); {a.n_repeats} tekrar × 5 fold, gün-gruplu.", "",
         "| kol | E1 fold AUC (gün-gruplu) | T1 PLR p | OR / SD [bootstrap %95] | CV3 p | Holm (betimsel) | Aşama 1 (katılımcı-gruplu): p · OR | G1 okuması |",
         "|---|---|---|---|---|---|---|---|"]
    for arm in arms:
        t, bo, e = R["t1"][arm], R["boot"][arm], R["e1"][arm]
        s1 = f"{fp(stage1['t1'][arm]['p_lrt'])} · {f3(stage1['t1'][arm]['or_sd'])}" if stage1 else "—"
        L.append(f"| {arm} | {e['fold_auc_mean']:.3f} ± {e['fold_auc_sd']:.3f} | {fp(t['p_lrt'])} | {f3(t['or_sd'])} [{f3(bo['lo'])}, "
                 f"{f3(bo['hi'])}] | {fp(t['p_cv3'])} | {fp(R['holm_descriptive'].get(arm)) if arm in R['holm_descriptive'] else 'ref.'} | "
                 f"{s1} | {R['verdict'].get(arm, '—')} |")
    L += ["", "- E1 değerleri katılımcı-gruplu Faz 2 sayılarıyla doğrudan karşılaştırılamaz: split'ler ve tekrar sayısı farklı; "
          "gün-gruplu split'te eğitim günleri daha az çeşitli.",
          "- Bu tablo Faz 3'ün dondurulmuş baz çizgisidir (C3 eşleştirmesi aynı split'lerde yapılır).", ""]
    md = "\n".join(L)
    js = json.dumps(R, indent=1, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    assert E.ids_free(md + js, C.index), "raporda katılımcı ID'si var (D-014)"
    a.report_dir.mkdir(parents=True, exist_ok=True)
    (a.report_dir / "G1_frozen_daygrouped.md").write_text(md)
    (a.report_dir / "G1_frozen_daygrouped.json").write_text(js)
    if a.out_dir:
        a.out_dir.mkdir(parents=True, exist_ok=True)
        S.assign(label=Ct.label).to_csv(a.out_dir / "g1_participant_scores.csv")
    print(md)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("frozen-dg")
    p.add_argument("--probe-dir", required=True, type=Path)
    p.add_argument("--splits-dir", required=True, type=Path)
    p.add_argument("--context", required=True, type=Path)
    p.add_argument("--report-dir", required=True, type=Path)
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--stage1-report", type=Path, help="reports/context/EVAL-016_context.json")
    p.add_argument("--n-repeats", type=int, default=3)
    p.add_argument("--backbones", default=",".join(RP.BACKBONES))
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cmd_frozen_dg(a)


if __name__ == "__main__":
    main()
