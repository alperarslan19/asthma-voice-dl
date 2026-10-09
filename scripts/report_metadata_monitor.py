"""
META-016 — Faz 2 meta veri / confounder izleme raporu (AYRI rapor; ana deney akışına girmez)
(D-005, D-028, D-034 madde 1 ve 8; docs/PHASE2_DESIGN.md Bölüm 4.2, 5.2)

BİLİMSEL AMAÇ
    EXP-016'nın ana sonuçlarını DEĞİŞTİRMEDEN, iki tür riski görünür kılmak:
      (1) Kayıt bağlamı (D-028): etiket sabitken skorun kayıt dönemini / saatini ne kadar ayırdığı
          (EXP-014'teki "bağlam bağımlılığı" ölçüleri). Yalnız betimsel; test, düzeltme ya da seçim YOK.
      (2) Kayıt süresi (D-034 madde 1): modeller süreyi dolaylı biçimde kullanabilir (kısa kayıtlarda dolgu ya da kısa
          bağlam, kayıt içi sessizlik oranı). Bunun için: süre dağılımı etikete göre, 4 s'den kısa kayıtların etikete ve
          göreve göre sayısı, yalnız-süre referans çizgisi (D-005 kayıt-koşulu tabanı; aynı fold'lar) ve önceden
          belirlenmiş duyarlılık: kısa kaydı olan katılımcılar değerlendirmeden çıkarıldığında füzyon AUC'leri.
    Bu raporun hiçbir sayısı EXP-016'nın onaylayıcı kararına, D-035 seçimine ya da katman seçimine girmez.

GİRDİLER
    --probe-dir   run_probes.py --out-dir (partial/<kol>_r<r>_test.csv)
    --cache-dir   audio_cache_v1 (index.csv: n_16k → kırpma sonrası süre)
    --context     participant_context.csv     --splits-dir  outer_r*.csv
ÇIKTI
    <report_dir>/META-016_metadata_monitor.{md,json}  (yalnız agrega; katılımcı ID'si içermez)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_probes as RP  # noqa: E402

TASKS = RP.TASKS
SHORT_S = 4.0


def load_tests(probe_dir: Path, splits: dict, C) -> pd.DataFrame:
    """Kaydedilmiş dış test tahminleri (birincil kollar; adında '@' olanlar hariç) + referans çizgileri (aynı fold'larda
    yeniden hesaplanır; run_probes bunları ara dosyaya yazmaz)."""
    files = sorted((probe_dir / "partial").glob("*_test.csv"))
    keep = [f for f in files if re.fullmatch(r"[a-z0-9_]+_r\d+_test\.csv", f.name) and int(f.name.rsplit("_r", 1)[1].split("_")[0]) in splits]
    # Tutarlılık: her kolun bütün tekrarları AYNI imzayla (aynı gömme / özellik / bağlam) üretilmiş olmalı
    arms = sorted({f.name.rsplit("_r", 1)[0] for f in keep})
    for a in arms:
        sigs = []
        for r in splits:
            j = probe_dir / "partial" / f"{a}_r{r}.json"
            assert j.exists() and (probe_dir / "partial" / f"{a}_r{r}_test.csv").exists(), f"{a}: tekrar {r} eksik"
            sig = json.loads(j.read_text())
            sig.pop("split_sha256", None)
            sigs.append(json.dumps(sig, sort_keys=True))
        assert len(set(sigs)) == 1, f"{a}: tekrarlar farklı imzalarla üretilmiş (eski ara sonuç karışmış olabilir)"
    df = pd.concat([pd.read_csv(f) for f in keep], ignore_index=True)
    refs = pd.concat([RP.run_references(C, S, r) for r, S in splits.items()], ignore_index=True)
    return RP.add_fusion(pd.concat([df, refs], ignore_index=True))


def context_monitors(test, C, arm) -> dict:
    s = RP.pooled(test, arm, RP.FUSION, C).rename("score").to_frame().join(C[["label", "period", "ampm"]])
    known = s[s.period != "unknown"]
    pat, hlt = known[known.label == 1], known[known.label == 0]
    pa, ha = pat[pat.ampm != "unknown"], hlt[hlt.ampm != "unknown"]
    f = lambda y, x: float(roc_auc_score(y, x)) if pd.Series(y).nunique() > 1 else float("nan")
    return {"patients_late_vs_early": f(pat.period == "late", pat.score), "patients_AM_vs_PM": f(pa.ampm == "AM", pa.score),
            "healthy_AM_vs_PM": f(ha.ampm == "AM", ha.score), "n_patients": int(len(pat)), "n_healthy": int(len(hlt))}


def duration_table(cache_dir: Path, C) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.read_csv(cache_dir / "index.csv")
    idx = idx[idx.participant_id.isin(C.index)].copy()
    idx["dur_s"] = idx.n_16k / 16000.0
    idx["label"] = C.loc[idx.participant_id, "label"].to_numpy()
    idx["short"] = idx.dur_s < SHORT_S
    rows = []
    for slot, task in TASKS.items():
        for lab, g in idx[idx.slot == slot].groupby("label"):
            q = g.dur_s.quantile([0.25, 0.5, 0.75])
            rows.append({"task": task, "label": "astım" if lab == 1 else "sağlıklı", "n": len(g),
                         "dur_median": q[0.5], "dur_q1": q[0.25], "dur_q3": q[0.75], "n_short_lt4s": int(g.short.sum())})
    part = idx.groupby("participant_id").agg(mean_dur=("dur_s", "mean"), n_short=("short", "sum"))
    return pd.DataFrame(rows), part


def duration_reference(part: pd.DataFrame, C, splits: dict) -> dict:
    """Yalnız-süre referans çizgisi (D-005 kayıt-koşulu tabanı): katılımcının ortalama kayıt süresi + kısa kayıt sayısı,
    LR (aynı ayarlarla, C=1, class_weight=balanced), aynı dış fold'lar."""
    X = part.join(C[["label"]])
    aucs, oof = [], []
    for r, S in splits.items():
        for k in range(5):
            Bk = S[S.outer_fold == k].set_index("participant_id")
            tr = [i for i in X.index if Bk.role.get(i) == "train"]
            te = [i for i in X.index if Bk.role.get(i) == "test"]
            m = RP.lr_pipe().set_params(clf__C=1.0).fit(X.loc[tr, ["mean_dur", "n_short"]], X.loc[tr, "label"])
            p = m.predict_proba(X.loc[te, ["mean_dur", "n_short"]])[:, 1]
            aucs.append(roc_auc_score(X.loc[te, "label"], p))
            oof.append(pd.Series(p, index=te))
    pm = pd.concat(oof).groupby(level=0).mean()
    y = C.loc[pm.index, "label"].to_numpy()
    lo, hi = RP.boot_ci_fast(y, pm.to_numpy())
    return {"auc_fold_mean": float(np.mean(aucs)), "auc_fold_sd": float(np.std(aucs, ddof=1)),
            "pooled_auc": float(roc_auc_score(y, pm)), "ci": [lo, hi], "n_folds": len(aucs)}


def exclusion_sensitivity(test, C, part, arms) -> dict:
    """Önceden belirlenmiş duyarlılık: 4 s'den kısa kaydı olan katılımcılar DEĞERLENDİRMEDEN çıkarılır (modeller aynı)."""
    excl = set(part.index[part.n_short > 0])
    out = {"n_excluded_participants": len(excl),
           "n_excluded_by_label": {("astım" if k == 1 else "sağlıklı"): int(v)
                                   for k, v in C.loc[sorted(excl), "label"].value_counts().items()} if excl else {}}
    res = {}
    for a in arms:
        f = test[(test.arm == a) & (test.task == RP.FUSION)]
        rows = {}
        for name, g in [("tümü", f), ("kısa kaydı olanlar hariç", f[~f.participant_id.isin(excl)])]:
            fold = [roc_auc_score(C.loc[gg.participant_id, "label"], gg.prob) for _, gg in g.groupby(["repeat", "fold"])]
            pm = g.groupby("participant_id").prob.mean()
            rows[name] = {"auc_fold_mean": float(np.mean(fold)), "pooled_auc": float(roc_auc_score(C.loc[pm.index, "label"], pm))}
        res[a] = rows
    out["by_arm"] = res
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", required=True, type=Path)
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--context", required=True, type=Path)
    ap.add_argument("--splits-dir", required=True, type=Path)
    ap.add_argument("--report-dir", required=True, type=Path)
    ap.add_argument("--n-repeats", type=int, default=5)
    args = ap.parse_args()
    splits = {r: pd.read_csv(args.splits_dir / f"outer_r{r}.csv") for r in range(args.n_repeats)}
    C = RP.load_context(args.context, set(splits[0].participant_id))
    test = load_tests(args.probe_dir, splits, C)
    arms = [a for a in RP.BACKBONES + ["mfcc_lr", "mfcc_mlp"] if a in set(test.arm)]
    refs = [a for a in ["ref_context", "ref_age"] if a in set(test.arm)]
    dur, part = duration_table(args.cache_dir, C)
    rep = {"exp": "META-016", "date": time.strftime("%Y-%m-%d"), "role": "ayrı meta veri / confounder izleme raporu (betimsel)",
           "context_monitors": {a: context_monitors(test, C, a) for a in arms + refs},
           "duration_by_task_label": dur.round(3).to_dict(orient="records"),
           "n_short_recordings_total": int(dur.n_short_lt4s.sum()),
           "duration_reference": duration_reference(part, C, splits),
           "short_exclusion_sensitivity": exclusion_sensitivity(test, C, part, arms)}
    L = ["# META-016 — meta veri / confounder izleme (AYRI rapor; ana deney akışına girmez; katılımcı ID'si içermez)", "",
         "Bu rapordaki hiçbir sayı EXP-016'nın onaylayıcı kararına, D-035 seçimine ya da katman seçimine girmez. "
         "Kayıt bağlamının etkisinin asıl değerlendirmesi model geliştirme bitince yapılacak (D-028).", "",
         "## 1. Bağlam bağımlılığı (EXP-014 ölçüleri; etiket sabitken skorun bağlamı ayırma gücü, 0.5 = bağımlılık yok)", "",
         "| kol | hastalarda geç vs erken | hastalarda sabah vs öğleden sonra | sağlıklılarda sabah vs öğleden sonra |", "|---|---|---|---|"]
    for a, m in rep["context_monitors"].items():
        L.append(f"| {a} | {m['patients_late_vs_early']:.3f} | {m['patients_AM_vs_PM']:.3f} | {m['healthy_AM_vs_PM']:.3f} |")
    L += ["", "## 2. Kayıt süresi (kırpma sonrası, önbellekten)", "",
          "| görev | etiket | n | medyan [Q1–Q3] s | < 4 s kayıt |", "|---|---|---|---|---|"]
    for r in dur.itertuples():
        L.append(f"| {r.task} | {r.label} | {r.n} | {r.dur_median:.2f} [{r.dur_q1:.2f}–{r.dur_q3:.2f}] | {r.n_short_lt4s} |")
    d = rep["duration_reference"]
    L += ["", f"Toplam < 4 s kayıt: {rep['n_short_recordings_total']}.", "",
          "## 3. Yalnız-süre referans çizgisi (D-005; katılımcının ortalama süresi + kısa kayıt sayısı; aynı dış fold'lar)", "",
          f"AUC fold ort {d['auc_fold_mean']:.3f} ± {d['auc_fold_sd']:.3f} · havuzlanmış {d['pooled_auc']:.3f} "
          f"[{d['ci'][0]:.3f}–{d['ci'][1]:.3f}] ({d['n_folds']} fold). 0.5'ten belirgin büyükse süre etiketle ilişkilidir ve "
          "modellerin süreyi dolaylı kullanması bir risk olur; yorum değerlendirme aşamasında.", ""]
    e = rep["short_exclusion_sensitivity"]
    L += ["## 4. Önceden belirlenmiş duyarlılık: kısa kaydı olan katılımcılar değerlendirmeden çıkarıldığında (modeller aynı)", "",
          f"Çıkarılan katılımcı: {e['n_excluded_participants']} {e['n_excluded_by_label']}", "",
          "| kol | füzyon AUC fold ort (tümü → hariç) | havuzlanmış AUC (tümü → hariç) |", "|---|---|---|"]
    for a, v in e["by_arm"].items():
        t, x = v["tümü"], v["kısa kaydı olanlar hariç"]
        L.append(f"| {a} | {t['auc_fold_mean']:.3f} → {x['auc_fold_mean']:.3f} | {t['pooled_auc']:.3f} → {x['pooled_auc']:.3f} |")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "META-016_metadata_monitor.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=float))
    md = "\n".join(L) + "\n"
    (args.report_dir / "META-016_metadata_monitor.md").write_text(md)
    print(md)
    txt = md + (args.report_dir / "META-016_metadata_monitor.json").read_text()
    leaked = {int(m) for m in re.findall(r"(?<![\d.])(\d{6})(?![\d.])", txt)} & set(C.index)
    assert not leaked, "raporda katılımcı ID'si var"


if __name__ == "__main__":
    main()
