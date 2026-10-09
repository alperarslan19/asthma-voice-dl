"""
EXP-012 (KEŞİFSEL, sonuçlar görüldükten SONRA tasarlandı) — MFCC sonuçlarını anlamak

BİLİMSEL AMAÇ
    EXP-010 (makale protokolü) ile EXP-011 (bizim protokol) iki şeyi birden değiştiriyor: girdiyi (orijinal ses vs
    harmonize önbellek) ve değerlendirme protokolünü (tek split + 14 model arasından seçim vs 5×5 split + önceden
    sabit model). Farkın nereden geldiğini ve makalenin sayısının ne kadarının şans/seçim olduğunu ayırmak:
      A. Yeniden üretilebilirlik: EXP-011 LR bu makinede yeniden çalıştırılınca Colab'la aynı sayıyı veriyor mu?
      B. Ayrıştırma (yalnız LR): {makale, önbellek özellikleri} × {makale, bizim protokol} 2×2 tablosu.
      C. Split şansı: makale protokolü + LR, 50 farklı tohumla → tek bir 5-fold'un ortalaması ne kadar oynuyor?
      D. Kazananın laneti (winner's curse): EXP-010'da tohum 42'de "en iyi" seçilen model, aynı makale
         pipeline'ıyla bizim 25 fold'umuzda ne alıyor? Kazanan hâlâ kazanan mı?
      E. Eşleştirilmiş karşılaştırmalar (D-011): LR füzyonu vs tek görevler; LR füzyonu vs yaş referansı.
      F. Özellik uyumu: aynı kayıt için makale ve önbellek özellikleri ne kadar benzer?
    Bu analiz ÖNCEDEN KAYDEDİLMEDİ; bulguları hipotez üretir, kesin sonuç değildir. Kayıt bağlamı değerlendirmesi
    (T1 vb.) bu analizin kapsamında DEĞİL (D-028).

GİRDİLER (katılımcı düzeyi; Drive / yerel, git'e girmez): iki özellik dosyası, participant_context.csv, split dosyaları
ÇIKTI: reports/mfcc/EXP-012_mfcc_analysis.{json,md} (yalnız agrega)
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy import stats
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_mfcc_baselines as rb  # noqa: E402  (EXP-010/011 ile AYNI kod)

warnings.filterwarnings("ignore")
TASKS = rb.TASKS
FUSION = "fusion (7 görev)"


def paper_protocol(F, C, feat, est, seed, folds_from=None, only_task=None) -> pd.DataFrame:
    """Makale pipeline'ı (StandardScaler → SMOTE(42) → model). folds_from verilirse bizim 25 fold'umuzda."""
    rows = []
    for slot, task in TASKS.items():
        if only_task and task != only_task:
            continue
        d = F[F.slot == slot].sort_values("participant_id")
        X, y, ids = d[feat].to_numpy(), C.loc[d.participant_id, "label"].to_numpy(), d.participant_id.to_numpy()
        splits = []
        if folds_from is None:
            splits = [(0, k, tr, te) for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y))]
        else:
            for r, S in folds_from.items():
                for k in range(5):
                    test = set(S[(S.outer_fold == k) & (S.role == "test")].participant_id)
                    te = np.flatnonzero(np.isin(ids, list(test)))
                    tr = np.flatnonzero(~np.isin(ids, list(test)))
                    splits.append((r, k, tr, te))
        for r, k, tr, te in splits:
            assert not set(ids[tr]) & set(ids[te])
            pipe = ImbPipeline([("sc", StandardScaler()), ("smote", SMOTE(random_state=rb.SEED)), ("clf", clone(est))])
            pipe.fit(X[tr], y[tr])
            rows.append({"task": task, "repeat": r, "fold": k, "auc": roc_auc_score(y[te], pipe.predict_proba(X[te])[:, 1])})
    return pd.DataFrame(rows)


def paired_boot(y, p1, p2, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            d.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i]))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def nadeau_bengio(d: np.ndarray, test_train_ratio: float) -> tuple[float, float]:
    """Düzeltilmiş tekrarlı-CV t-testi (Nadeau & Bengio 2003): fold'lar bağımsız değil → varyans şişirilir."""
    J = len(d)
    se = np.sqrt((1 / J + test_train_ratio) * d.var(ddof=1))
    t = d.mean() / se
    return float(t), float(2 * stats.t.sf(abs(t), J - 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-paper", required=True, type=Path)
    ap.add_argument("--features-cache", required=True, type=Path)
    ap.add_argument("--context", required=True, type=Path)
    ap.add_argument("--splits-dir", required=True, type=Path)
    ap.add_argument("--exp010-json", required=True, type=Path)
    ap.add_argument("--exp011-json", required=True, type=Path)
    ap.add_argument("--report-dir", required=True, type=Path)
    ap.add_argument("--n-seeds", type=int, default=50)
    args = ap.parse_args()
    t0 = time.time()
    Fp, C, feat = rb.load(args.features_paper, args.context)
    Fc, _, feat_c = rb.load(args.features_cache, args.context)
    assert feat == feat_c
    e10 = pd.DataFrame(json.loads(args.exp010_json.read_text())["summary"])
    e11 = pd.DataFrame(json.loads(args.exp011_json.read_text())["summary"])
    S = {r: pd.read_csv(args.splits_dir / f"outer_r{r}.csv") for r in range(5)}
    R: dict = {"note": "KEŞİFSEL (post hoc) analiz; önceden kaydedilmedi"}

    # A + B: bizim protokol, LR, iki özellik setiyle (run_mfcc_baselines ile aynı kod)
    tmp = Path(tempfile.mkdtemp())
    fm_c, oof_c = rb.run_exp011(Fc, C, feat, args.splits_dir, tmp / "c", ["LR"], 5)
    fm_p, oof_p = rb.run_exp011(Fp, C, feat, args.splits_dir, tmp / "p", ["LR"], 5)
    sc, sp = rb.summarize(fm_c, oof_c, C), rb.summarize(fm_p, oof_p, C)
    lr11 = e11[e11.model == "LR"].set_index("task").auc_mean
    rep = sc[sc.model == "LR"].set_index("task").auc_mean
    R["A_reproduction_max_abs_diff"] = float((rep - lr11).abs().max())
    print(f"A: Colab ile en büyük fark {R['A_reproduction_max_abs_diff']:.2e}")

    p4 = paper_protocol(Fc, C, feat, LogisticRegression(max_iter=1000, random_state=rb.SEED), seed=rb.SEED)
    B = pd.DataFrame({
        "P1 makale özellik + makale protokol (EXP-010 LR)": e10[e10.model == "LogisticRegression"].set_index("task").auc_mean,
        "P4 önbellek özellik + makale protokol": p4.groupby("task").auc.mean(),
        "P2 makale özellik + bizim protokol": sp[sp.model == "LR"].set_index("task").auc_mean,
        "P3 önbellek özellik + bizim protokol (EXP-011 LR)": rep,
    }).reindex(list(TASKS.values()) + [FUSION])
    R["B_decomposition_lr_auc"] = B.round(4).to_dict()

    # C: split şansı (makale protokolü + LR, makale özellikleri)
    seeds = []
    for s in range(args.n_seeds):
        seeds.append(paper_protocol(Fp, C, feat, LogisticRegression(max_iter=1000, random_state=rb.SEED), seed=s)
                     .groupby("task").auc.mean().rename(s))
    Cs = pd.concat(seeds, axis=1)
    s42 = paper_protocol(Fp, C, feat, LogisticRegression(max_iter=1000, random_state=rb.SEED), seed=42).groupby("task").auc.mean()
    R["C_split_luck_lr"] = {t: {"seed42": round(float(s42[t]), 4), "mean": round(float(Cs.loc[t].mean()), 4),
                                "sd": round(float(Cs.loc[t].std()), 4), "min": round(float(Cs.loc[t].min()), 4),
                                "max": round(float(Cs.loc[t].max()), 4),
                                "seed42_percentile": round(float((Cs.loc[t] < s42[t]).mean() * 100), 1)} for t in TASKS.values()}
    print(f"C: bitti ({time.time() - t0:.0f} s)")

    # D: kazananın laneti
    models = rb.paper_models()
    cheap = [m for m in models if m not in {"CatBoost", "VotingEnsemble", "StackingEnsemble", "WeightedVotingEnsemble"}]
    winners = json.loads(args.exp010_json.read_text())["paper_rule_best_per_task"]
    D_rows = []
    for name in cheap:
        res = paper_protocol(Fp, C, feat, models[name], seed=None, folds_from=S).groupby("task").auc.mean()
        for t, v in res.items():
            D_rows.append({"task": t, "model": name, "auc25": v})
        print(f"  D: {name} ({time.time() - t0:.0f} s)", flush=True)
    for w in winners:
        if w["model"] not in cheap:
            res = paper_protocol(Fp, C, feat, models[w["model"]], seed=None, folds_from=S, only_task=w["task"])
            D_rows.append({"task": w["task"], "model": w["model"], "auc25": float(res.auc.mean())})
            print(f"  D: {w['model']} / {w['task']} ({time.time() - t0:.0f} s)", flush=True)
    D = pd.DataFrame(D_rows)
    Dout = []
    for w in winners:
        g = D[D.task == w["task"]].sort_values("auc25", ascending=False).reset_index(drop=True)
        win25 = float(g[g.model == w["model"]].auc25.iloc[0])
        Dout.append({"task": w["task"], "winner_seed42": w["model"], "auc_seed42": w["auc_mean"], "auc_25folds": round(win25, 4),
                     "drop": round(w["auc_mean"] - win25, 4), "rank_among_evaluated_on_25folds": int(g.index[g.model == w["model"]][0]) + 1,
                     "n_models_evaluated": len(g), "best_on_25folds": g.model.iloc[0], "best_auc_25folds": round(float(g.auc25.iloc[0]), 4)})
    R["D_winners_curse"] = Dout

    # E: eşleştirilmiş karşılaştırmalar (EXP-011 LR, bu makinedeki yeniden üretim)
    ratio = 68 / 274
    E = []
    fz = fm_c[(fm_c.task == FUSION) & (fm_c.model == "LR")].set_index(["repeat", "fold"]).auc
    m_f = oof_c[(oof_c.task == FUSION) & (oof_c.model == "LR")].groupby("participant_id").prob.mean()
    for t in list(TASKS.values()) + ["age", "context (hour+hour²+date)"]:
        is_ref = t in {"age", "context (hour+hour²+date)"}
        sel = (fm_c.task == "reference") & (fm_c.model == t) if is_ref else (fm_c.task == t) & (fm_c.model == "LR")
        other = fm_c[sel].set_index(["repeat", "fold"]).auc
        d = (fz - other).dropna().to_numpy()
        tt, pp = nadeau_bengio(d, ratio)
        m_o = oof_c[(oof_c.task == ("reference" if is_ref else t)) & (oof_c.model == (t if is_ref else "LR"))].groupby("participant_id").prob.mean()
        ids = m_f.index.intersection(m_o.index)
        y = C.loc[ids, "label"].to_numpy()
        lo, hi = paired_boot(y, m_f[ids].to_numpy(), m_o[ids].to_numpy())
        E.append({"comparison": f"LR füzyon − {t}", "delta_fold_mean": round(float(d.mean()), 4), "nb_t": round(tt, 2),
                  "nb_p": round(pp, 4), "delta_pooled": round(float(roc_auc_score(y, m_f[ids]) - roc_auc_score(y, m_o[ids])), 4),
                  "boot_ci": [round(lo, 4), round(hi, 4)], "n": int(len(ids))})
    R["E_paired_comparisons"] = E

    # F: özellik uyumu
    M = Fp.merge(Fc, on=["participant_id", "slot"], suffixes=("_p", "_c"))
    corr = {f: float(np.corrcoef(M[f + "_p"], M[f + "_c"])[0, 1]) for f in feat}
    cs = pd.Series(corr)
    R["F_feature_agreement"] = {"n_recordings": int(len(M)), "median_r": round(float(cs.median()), 3),
                                "min_r": round(float(cs.min()), 3), "features_r_below_0.8": cs[cs < 0.8].round(3).to_dict(),
                                "kept_s_paper_median": round(float(M.kept_s_p.median()), 2),
                                "kept_s_cache_median": round(float(M.kept_s_c.median()), 2)}
    R["runtime_s"] = round(time.time() - t0, 1)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "EXP-012_mfcc_analysis.json").write_text(json.dumps(R, indent=2, ensure_ascii=False, default=float))

    L = ["# EXP-012 — MFCC sonuçlarının keşifsel ayrıştırması (post hoc; katılımcı ID'si içermez)", "",
         "**Bu analiz sonuçlar görüldükten sonra tasarlandı:** bulgular hipotez üretir. Bütün AUC'ler üst sınırdır (D-028).", "",
         f"## A. Yeniden üretilebilirlik\n\nEXP-011 LR bu makinede (farklı scikit-learn sürümü) yeniden çalıştırıldı; görev başına en büyük fark: {R['A_reproduction_max_abs_diff']:.1e}.", "",
         "## B. Girdi mi, protokol mü? (yalnız LR, fold AUC ortalaması)", "",
         "| görev | " + " | ".join(B.columns) + " |", "|---|" + "---|" * len(B.columns)]
    for t, row in B.iterrows():
        L.append(f"| {t} | " + " | ".join("—" if pd.isna(v) else f"{v:.3f}" for v in row) + " |")
    L += ["", "## C. Split şansı (makale protokolü + LR; aynı veri, 50 farklı 5-fold bölünmesi)", "",
          "| görev | tohum 42 | 50 tohum ort ± SD | min–max | tohum 42'nin yüzdeliği |", "|---|---|---|---|---|"]
    for t, v in R["C_split_luck_lr"].items():
        L.append(f"| {t} | {v['seed42']:.3f} | {v['mean']:.3f} ± {v['sd']:.3f} | {v['min']:.3f}–{v['max']:.3f} | {v['seed42_percentile']:.0f} |")
    L += ["", "## D. Kazananın laneti (makale pipeline'ı; makale özellikleri)", "",
          "| görev | tohum 42'de kazanan | AUC (tohum 42, 5 fold) | aynı model, bizim 25 fold | düşüş | 25 fold'da sırası | 25 fold'da en iyi |",
          "|---|---|---|---|---|---|---|"]
    for v in Dout:
        L.append(f"| {v['task']} | {v['winner_seed42']} | {v['auc_seed42']:.3f} | {v['auc_25folds']:.3f} | {v['drop']:+.3f} | "
                 f"{v['rank_among_evaluated_on_25folds']}/{v['n_models_evaluated']} | {v['best_on_25folds']} ({v['best_auc_25folds']:.3f}) |")
    L += ["", "## E. Eşleştirilmiş karşılaştırmalar (EXP-011 LR; D-011)", "",
          "| karşılaştırma | ΔAUC fold ort. | Nadeau–Bengio t, p | ΔAUC havuzlanmış [%95 bootstrap CI] | n |", "|---|---|---|---|---|"]
    for v in E:
        L.append(f"| {v['comparison']} | {v['delta_fold_mean']:+.3f} | {v['nb_t']:.2f}, {v['nb_p']:.3f} | "
                 f"{v['delta_pooled']:+.3f} [{v['boot_ci'][0]:+.3f}, {v['boot_ci'][1]:+.3f}] | {v['n']} |")
    Fa = R["F_feature_agreement"]
    L += ["", "## F. Aynı kaydın makale ve önbellek özellikleri ne kadar benzer?", "",
          f"- {Fa['n_recordings']} kayıt; özellik başına Pearson r medyanı {Fa['median_r']}, en düşük {Fa['min_r']}",
          f"- r < 0.8 olan özellikler: {Fa['features_r_below_0.8'] if Fa['features_r_below_0.8'] else 'yok'}",
          f"- Kırpma sonrası süre medyanı: makale {Fa['kept_s_paper_median']} s, önbellek {Fa['kept_s_cache_median']} s",
          "", f"Süre: {R['runtime_s']} s"]
    (args.report_dir / "EXP-012_mfcc_analysis.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
