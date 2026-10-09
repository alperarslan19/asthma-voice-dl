"""
EXP-013 / EXP-014 / EXP-015 (KEŞİFSEL; Alper'in 11. turdaki metodoloji soruları üzerine) — MFCC özellikleriyle

BİLİMSEL AMAÇ
  EXP-013  Etiket dengesizliği (283 astım / 59 sağlıklı) nasıl ele alınmalı?
           Yöntemler: hiçbiri / sınıf ağırlığı / SMOTE / rastgele aşırı örnekleme. Modeller: LR (C iç CV'de) ve MLP.
           Ölçülenler:
             - ayırma (AUC);
             - kalibrasyon (Brier, ortalama tahmin − gerçek oran, kalibrasyon eğimi);
             - eşik metrikleri: 0.5 eşiğinde ve iç CV'den seçilen eşikte dengeli doğruluk.
           Soru: SMOTE'u kullanmak ya da kullanmamak sonucu neresinden değiştirir?
  EXP-014  Kayıt bağlamı dengesizliği SMOTE / yeniden ağırlıklandırma ile "düzeltilebilir" mi?
           Bağlam hücreleri = dönem (erken / geç, D-024) × sabah / öğleden sonra.
           Varyantlar (yalnız LR):
             - V0 sınıf ağırlığı, tüm veri (= EXP-011 LR);
             - V2 hücre-içi SMOTE, tüm veri;
             - V3 hücre-ters-eğilim ağırlığı, tüm veri;
             - V4 yalnız örtüşme bölgesi (erken dönem) + hücre ağırlığı;
             - V5 yalnız örtüşme bölgesi + hücre-içi SMOTE.
           Ölçülenler: tüm testte AUC, yalnız erken dönemde AUC, hücre-tabakalı AUC (aynı hücredeki astım–sağlıklı
           çiftleri), aynı ölçütlerle yaş-only ve bağlam-only referansları, ve "bağlam bağımlılığı":
           hastalar içinde skorun geç / erken ve sabah / öğleden sonra ayrımı.
  EXP-015  İç içe (nested) CV: makalenin "14 model arasından en iyisini seç" prosedürünün dürüst performansı.
           Seçim iç CV'de yapılır, dış test fold'u seçime hiç karışmaz. Ayrıca hangi modelin ne sıklıkla seçildiği raporlanır.
           Kapsam: makale pipeline'ı (StandardScaler → SMOTE → model), makale özellikleri, 10 model.
           Hesap maliyeti nedeniyle CatBoost ve 3 ensemble dışarıda.
  Hepsi sonuçlar görüldükten sonra, soruya cevap olarak tasarlandı → keşifsel. Bütün AUC'ler üst sınırdır (D-028).

GİRDİLER (katılımcı düzeyi; git'e girmez): özellik dosyaları, participant_context.csv, split dosyaları outer_r0–4
ÇIKTI: reports/mfcc/EXP-01{3,4,5}_*.{md,json} (yalnız agrega)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy import stats
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_mfcc_baselines as rb  # noqa: E402

warnings.filterwarnings("ignore")
TASKS, SEED = rb.TASKS, rb.SEED
C_GRID = [0.001, 0.01, 0.1, 1, 10, 100]


# ----------------------------------------------------------------------------- ortak yardımcılar
def folds(S: dict):
    for r, Sr in S.items():
        for k in range(5):
            B = Sr[Sr.outer_fold == k].set_index("participant_id")
            yield r, k, set(B.index[B.role == "test"]), B.loc[B.role == "train", "inner_fold"]


def best_threshold(y, p) -> float:
    """İç CV OOF tahminlerinde dengeli doğruluğu en yükselten eşik (eşitlikte ortanca)."""
    cand = np.unique(p)
    ba = np.array([0.5 * (((p >= t) & (y == 1)).sum() / max(1, (y == 1).sum()) +
                          ((p < t) & (y == 0)).sum() / max(1, (y == 0).sum())) for t in cand])
    return float(np.median(cand[ba == ba.max()]))


def bal_acc(y, p, t) -> float:
    yh = (p >= t).astype(int)
    return 0.5 * (((yh == 1) & (y == 1)).sum() / max(1, (y == 1).sum()) + ((yh == 0) & (y == 0)).sum() / max(1, (y == 0).sum()))


def calib(y, p) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lg = np.log(p / (1 - p)).reshape(-1, 1)
    m = LogisticRegression(penalty=None).fit(lg, y)
    return {"brier": float(np.mean((p - y) ** 2)), "mean_pred_minus_prev": float(p.mean() - y.mean()),
            "calib_slope": float(m.coef_[0, 0])}


def paired(y, p1, p2, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            d.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i]))
    return float(roc_auc_score(y, p1) - roc_auc_score(y, p2)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def strat_auc(df: pd.DataFrame, score: str, key: str) -> tuple[float, int]:
    """Yalnız aynı tabakadaki astım–sağlıklı çiftlerini karşılaştıran AUC."""
    num = den = 0.0
    for _, g in df.groupby(key):
        a, b = g.loc[g.label == 1, score].to_numpy(), g.loc[g.label == 0, score].to_numpy()
        if len(a) and len(b):
            num += (a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()
            den += len(a) * len(b)
    return (num / den if den else float("nan")), int(den)


def add_cells(C: pd.DataFrame) -> pd.DataFrame:
    rd = pd.to_datetime(C["recording_date"])
    last = rd[C.label == 0].max()
    C = C.copy()
    C["period"] = np.select([rd.isna(), rd <= last], ["unknown", "early"], "late")
    C["ampm"] = np.select([C.start_hour.isna(), C.start_hour < 12], ["unknown", "AM"], "PM")
    C["cell"] = C["period"] + "-" + C["ampm"]
    return C


def fit_predict(model, Xtr, ytr, Xte, inner, sw=None, grid=None):
    """İç CV: (varsa) ızgara seçimi + eşik için OOF; sonra tüm train'de yeniden fit, test tahmini."""
    fit_kw = {"clf__sample_weight": sw} if sw is not None else {}
    if grid:
        gs = GridSearchCV(clone(model), grid, cv=inner, scoring="neg_log_loss", refit=True).fit(Xtr, ytr, **fit_kw)
        best = gs.best_estimator_
        est = clone(model).set_params(**gs.best_params_)
    else:
        est = clone(model)
        best = clone(model).fit(Xtr, ytr, **fit_kw)
    inner_p = cross_val_predict(est, Xtr, ytr, cv=inner, method="predict_proba", params=fit_kw)[:, 1]
    return best.predict_proba(Xte)[:, 1], inner_p


# ----------------------------------------------------------------------------- EXP-013
def exp013(F, C, feat, S) -> tuple[pd.DataFrame, pd.DataFrame]:
    lr = LogisticRegression(max_iter=5000)
    mlp = MLPClassifier(random_state=SEED)
    variants = {
        ("LR", "none"): (ImbPipeline([("sc", StandardScaler()), ("clf", lr)]), {"clf__C": C_GRID}, False),
        ("LR", "class_weight"): (ImbPipeline([("sc", StandardScaler()), ("clf", clone(lr).set_params(class_weight="balanced"))]), {"clf__C": C_GRID}, False),
        ("LR", "SMOTE"): (ImbPipeline([("sc", StandardScaler()), ("rs", SMOTE(random_state=SEED)), ("clf", lr)]), {"clf__C": C_GRID}, False),
        ("LR", "random_oversampling"): (ImbPipeline([("sc", StandardScaler()), ("rs", RandomOverSampler(random_state=SEED)), ("clf", lr)]), {"clf__C": C_GRID}, False),
        ("MLP", "none"): (ImbPipeline([("sc", StandardScaler()), ("clf", mlp)]), None, False),
        ("MLP", "class_weight"): (ImbPipeline([("sc", StandardScaler()), ("clf", mlp)]), None, True),
        ("MLP", "SMOTE"): (ImbPipeline([("sc", StandardScaler()), ("rs", SMOTE(random_state=SEED)), ("clf", mlp)]), None, False),
        ("MLP", "random_oversampling"): (ImbPipeline([("sc", StandardScaler()), ("rs", RandomOverSampler(random_state=SEED)), ("clf", mlp)]), None, False),
    }
    rows, inner_rows = [], []
    t0 = time.time()
    for r, k, test_ids, inner_map in folds(S):
        for slot, task in TASKS.items():
            d = F[F.slot == slot]
            tr, te = d[d.participant_id.isin(inner_map.index)], d[d.participant_id.isin(test_ids)]
            ytr = C.loc[tr.participant_id, "label"].to_numpy()
            inner = PredefinedSplit(inner_map.loc[tr.participant_id].to_numpy())
            for (m, strat), (pipe, grid, needs_sw) in variants.items():
                sw = compute_sample_weight("balanced", ytr) if needs_sw else None
                p, ip = fit_predict(pipe, tr[feat].to_numpy(), ytr, te[feat].to_numpy(), inner, sw, grid)
                rows.append(pd.DataFrame({"participant_id": te.participant_id.to_numpy(), "task": task, "model": m,
                                          "strategy": strat, "repeat": r, "fold": k, "prob": p}))
                inner_rows.append(pd.DataFrame({"participant_id": tr.participant_id.to_numpy(), "task": task, "model": m,
                                                "strategy": strat, "repeat": r, "fold": k, "prob": ip}))
        print(f"  EXP-013 r{r} k{k} ({time.time() - t0:.0f} s)", flush=True)
    return pd.concat(rows, ignore_index=True), pd.concat(inner_rows, ignore_index=True)


def summarize013(oof, inner, C) -> pd.DataFrame:
    keys = ["model", "strategy"]
    # füzyon: görev olasılıklarının ortalaması (test ve iç OOF için ayrı ayrı)
    fus = oof.groupby(["participant_id", *keys, "repeat", "fold"], as_index=False).prob.mean().assign(task="fusion")
    ifus = inner.groupby(["participant_id", *keys, "repeat", "fold"], as_index=False).prob.mean().assign(task="fusion")
    oof, inner = pd.concat([oof, fus]), pd.concat([inner, ifus])
    out = []
    for (task, m, s), g in oof.groupby(["task", *keys]):
        aucs, ba05, bat = [], [], []
        for (r, k), gg in g.groupby(["repeat", "fold"]):
            y = C.loc[gg.participant_id, "label"].to_numpy()
            gi = inner[(inner.task == task) & (inner.model == m) & (inner.strategy == s) & (inner.repeat == r) & (inner.fold == k)]
            t = best_threshold(C.loc[gi.participant_id, "label"].to_numpy(), gi.prob.to_numpy())
            aucs.append(roc_auc_score(y, gg.prob))
            ba05.append(bal_acc(y, gg.prob.to_numpy(), 0.5))
            bat.append(bal_acc(y, gg.prob.to_numpy(), t))
        pooled = g.groupby("participant_id").prob.mean()
        y = C.loc[pooled.index, "label"].to_numpy()
        allp = g.prob.to_numpy()
        ally = C.loc[g.participant_id, "label"].to_numpy()
        lo, hi = rb.boot_ci(y, pooled.to_numpy())
        out.append({"task": task, "model": m, "strategy": s, "auc_mean": np.mean(aucs), "auc_sd": np.std(aucs, ddof=1),
                    "pooled_auc": roc_auc_score(y, pooled), "ci_low": lo, "ci_high": hi,
                    "balacc_05": np.mean(ba05), "balacc_thr": np.mean(bat), **calib(ally, allp)})
    return pd.DataFrame(out), oof


# ----------------------------------------------------------------------------- EXP-014
def cell_smote(X, y, cells, only_cells=None):
    """Her hücrede azınlık etiketini o hücrenin çoğunluğuna kadar SMOTE'la çoğalt (hücrede ≥2 azınlık varsa).
    Azınlığı hiç olmayan hücre (geç dönem: sağlıklı yok) DEĞİŞMEZ — pozitiflik sınırı."""
    Xs, ys, cs = [X], [y], [cells]
    for c in np.unique(cells):
        if only_cells is not None and c not in only_cells:
            continue
        m = cells == c
        n1, n0 = int((y[m] == 1).sum()), int((y[m] == 0).sum())
        if min(n1, n0) < 2 or n1 == n0:
            continue
        sm = SMOTE(random_state=SEED, k_neighbors=min(5, min(n1, n0) - 1))
        Xr, yr = sm.fit_resample(X[m], y[m])
        new = len(Xr) - int(m.sum())
        Xs.append(Xr[-new:]); ys.append(yr[-new:]); cs.append(np.full(new, c))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(cs)


def ipw(y, cells) -> np.ndarray:
    """Ters eğilim ağırlığı w = 1 / p̂(y | hücre) (train'deki frekanslardan); ortalaması 1'e normalize."""
    df = pd.DataFrame({"y": y, "c": cells})
    p = df.groupby("c").y.transform("mean").to_numpy()
    w = np.where(y == 1, 1 / p, 1 / (1 - p))
    w = np.where(np.isfinite(w), w, 0.0)
    return w / w.mean()


def exp014(F, C, feat, S) -> pd.DataFrame:
    rows = []
    t0 = time.time()
    for r, k, test_ids, inner_map in folds(S):
        for slot, task in TASKS.items():
            d = F[F.slot == slot]
            tr, te = d[d.participant_id.isin(inner_map.index)], d[d.participant_id.isin(test_ids)]
            ytr, ctr = C.loc[tr.participant_id, "label"].to_numpy(), C.loc[tr.participant_id, "cell"].to_numpy()
            sc = StandardScaler().fit(tr[feat].to_numpy())
            Xtr, Xte = sc.transform(tr[feat].to_numpy()), sc.transform(te[feat].to_numpy())
            early = np.array([c.startswith("early") for c in ctr])
            inner = inner_map.loc[tr.participant_id].to_numpy()

            def lr_fit(X, y, w, inner_f):
                # C iç fold'larda log-loss ile (ağırlıklı); örneklenmiş veride iç fold'lar sentetik örnek içermez
                best, bl = None, np.inf
                for Cv in C_GRID:
                    ll = 0.0
                    for j in np.unique(inner_f):
                        a, b = inner_f != j, inner_f == j
                        m = LogisticRegression(C=Cv, max_iter=5000).fit(X[a], y[a], sample_weight=None if w is None else w[a])
                        p = np.clip(m.predict_proba(X[b])[:, 1], 1e-9, 1 - 1e-9)
                        ll += -np.mean(y[b] * np.log(p) + (1 - y[b]) * np.log(1 - p))
                    if ll < bl:
                        best, bl = Cv, ll
                return LogisticRegression(C=best, max_iter=5000).fit(X, y, sample_weight=w)

            variants = {}
            variants["V0 sınıf ağırlığı (tüm veri)"] = lr_fit(Xtr, ytr, compute_sample_weight("balanced", ytr), inner)
            Xs, ys, cs = cell_smote(Xtr, ytr, ctr)
            # sentetik örnekler iç fold'a girmesin: C, gerçek örneklerle (V0'da) seçilir, sonra hepsiyle fit
            m0 = variants["V0 sınıf ağırlığı (tüm veri)"]
            variants["V2 hücre-içi SMOTE (tüm veri)"] = LogisticRegression(C=m0.C, max_iter=5000).fit(Xs, ys, sample_weight=compute_sample_weight("balanced", ys))
            variants["V3 hücre-ters-eğilim ağırlığı (tüm veri)"] = lr_fit(Xtr, ytr, ipw(ytr, ctr), inner)
            Xe, ye, ce, ie = Xtr[early], ytr[early], ctr[early], inner[early]
            variants["V4 yalnız erken dönem + hücre ağırlığı"] = lr_fit(Xe, ye, ipw(ye, ce), ie)
            m4 = variants["V4 yalnız erken dönem + hücre ağırlığı"]
            Xs5, ys5, _ = cell_smote(Xe, ye, ce)
            variants["V5 yalnız erken dönem + hücre-içi SMOTE"] = LogisticRegression(C=m4.C, max_iter=5000).fit(Xs5, ys5)
            for name, m in variants.items():
                rows.append(pd.DataFrame({"participant_id": te.participant_id.to_numpy(), "task": task, "variant": name,
                                          "repeat": r, "fold": k, "prob": m.predict_proba(Xte)[:, 1]}))
        # referanslar (sesi kullanmayan LR), aynı fold'larda
        ok = C.dropna(subset=["age", "start_hour", "days"])
        trn = [i for i in ok.index if i in inner_map.index]
        tst = [i for i in ok.index if i in test_ids]
        for ref, cols in {"ref: yaş": ["age"], "ref: bağlam (saat+saat²+tarih)": ["start_hour", "hour2", "days"]}.items():
            m = rb.Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(class_weight="balanced", max_iter=5000))])
            m.fit(ok.loc[trn, cols], ok.loc[trn, "label"])
            rows.append(pd.DataFrame({"participant_id": tst, "task": "reference", "variant": ref, "repeat": r, "fold": k,
                                      "prob": m.predict_proba(ok.loc[tst, cols])[:, 1]}))
        print(f"  EXP-014 r{r} k{k} ({time.time() - t0:.0f} s)", flush=True)
    return pd.concat(rows, ignore_index=True)


def summarize014(oof, C) -> pd.DataFrame:
    fus = oof[oof.task != "reference"].groupby(["participant_id", "variant", "repeat", "fold"], as_index=False).prob.mean()
    fus["task"] = "fusion"
    ref = oof[oof.task == "reference"].assign(task="fusion")
    allf = pd.concat([fus, ref])
    out = []
    for v, g in allf.groupby("variant"):
        s = g.groupby("participant_id").prob.mean().rename("score").to_frame().join(C[["label", "period", "ampm", "cell"]])
        known = s[s.period != "unknown"]
        e = known[known.period == "early"]
        cells = known[known.cell.isin(["early-AM", "early-PM"])]
        pat = known[known.label == 1]
        hlt = known[known.label == 0]
        lo, hi = rb.boot_ci(s.label.to_numpy(), s.score.to_numpy())
        out.append({
            "variant": v, "auc_all": roc_auc_score(s.label, s.score), "auc_all_ci": [lo, hi],
            "auc_early_period_only": roc_auc_score(e.label, e.score),
            "auc_within_cells_early_AM_PM": strat_auc(cells, "score", "cell")[0],
            "auc_early_AM_cell": roc_auc_score(cells[cells.cell == "early-AM"].label, cells[cells.cell == "early-AM"].score),
            "auc_early_PM_cell": roc_auc_score(cells[cells.cell == "early-PM"].label, cells[cells.cell == "early-PM"].score),
            "context_reliance_patients_late_vs_early": roc_auc_score(pat.period == "late", pat.score),
            "context_reliance_patients_AM_vs_PM": roc_auc_score(pat[pat.ampm != "unknown"].ampm == "AM", pat[pat.ampm != "unknown"].score),
            "context_reliance_healthy_AM_vs_PM": roc_auc_score(hlt[hlt.ampm != "unknown"].ampm == "AM", hlt[hlt.ampm != "unknown"].score),
            "n_all": int(len(s)), "n_early": int(len(e)),
            "n_cells": {c: [int((cells[cells.cell == c].label == 1).sum()), int((cells[cells.cell == c].label == 0).sum())] for c in ["early-AM", "early-PM"]},
        })
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- EXP-015
def exp015(F, C, feat, S) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = {k: v for k, v in rb.paper_models().items()
              if k not in {"CatBoost", "VotingEnsemble", "StackingEnsemble", "WeightedVotingEnsemble"}}
    rows, sel = [], []
    t0 = time.time()
    for r, k, test_ids, inner_map in folds(S):
        for slot, task in TASKS.items():
            d = F[F.slot == slot]
            tr, te = d[d.participant_id.isin(inner_map.index)], d[d.participant_id.isin(test_ids)]
            Xtr, ytr = tr[feat].to_numpy(), C.loc[tr.participant_id, "label"].to_numpy()
            Xte, yte = te[feat].to_numpy(), C.loc[te.participant_id, "label"].to_numpy()
            inner = inner_map.loc[tr.participant_id].to_numpy()
            inner_auc, test_auc = {}, {}
            for name, est in models.items():
                pipe = ImbPipeline([("sc", StandardScaler()), ("smote", SMOTE(random_state=SEED)), ("clf", clone(est))])
                ia = []
                for j in range(5):
                    a, b = inner != j, inner == j
                    ia.append(roc_auc_score(ytr[b], clone(pipe).fit(Xtr[a], ytr[a]).predict_proba(Xtr[b])[:, 1]))
                inner_auc[name] = float(np.mean(ia))
                p = clone(pipe).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
                test_auc[name] = roc_auc_score(yte, p)
                rows.append(pd.DataFrame({"participant_id": te.participant_id.to_numpy(), "task": task, "model": name,
                                          "repeat": r, "fold": k, "prob": p}))
            chosen = max(inner_auc, key=inner_auc.get)
            sel.append({"task": task, "repeat": r, "fold": k, "chosen": chosen, "chosen_inner_auc": inner_auc[chosen],
                        "nested_test_auc": test_auc[chosen], "oracle_test_auc": max(test_auc.values()),
                        **{f"m_{n}": v for n, v in test_auc.items()}})
        print(f"  EXP-015 r{r} k{k} ({time.time() - t0:.0f} s)", flush=True)
    return pd.DataFrame(sel), pd.concat(rows, ignore_index=True)


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["imbalance", "context", "nested"])
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--context", required=True, type=Path)
    ap.add_argument("--splits-dir", required=True, type=Path)
    ap.add_argument("--report-dir", required=True, type=Path)
    ap.add_argument("--n-repeats", type=int, default=5)
    args = ap.parse_args()
    F, C, feat = rb.load(args.features, args.context)
    C = add_cells(C)
    S = {r: pd.read_csv(args.splits_dir / f"outer_r{r}.csv") for r in range(args.n_repeats)}
    args.report_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    feats_sha = rb.sha256_file(args.features)[:16]

    if args.part == "imbalance":
        oof, inner = exp013(F, C, feat, S)
        summ, oofall = summarize013(oof, inner, C)
        comp = []
        for m in ["LR", "MLP"]:
            base = oofall[(oofall.task == "fusion") & (oofall.model == m) & (oofall.strategy == "none")].groupby("participant_id").prob.mean()
            for s in ["class_weight", "SMOTE", "random_oversampling"]:
                o = oofall[(oofall.task == "fusion") & (oofall.model == m) & (oofall.strategy == s)].groupby("participant_id").prob.mean()
                ids = base.index.intersection(o.index)
                d, lo, hi = paired(C.loc[ids, "label"].to_numpy(), o[ids].to_numpy(), base[ids].to_numpy())
                comp.append({"model": m, "comparison": f"{s} − none (füzyon)", "delta_auc": d, "ci": [lo, hi]})
        rep = {"exp": "EXP-013", "features_sha256": feats_sha, "summary": summ.round(4).to_dict(orient="records"),
               "paired_fusion": comp, "runtime_s": round(time.time() - t0, 1)}
        (args.report_dir / "EXP-013_imbalance.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=float))
        L = ["# EXP-013 — Etiket dengesizliğiyle başa çıkma yöntemleri (keşifsel; katılımcı ID'si içermez)", "",
             "Önbellek özellikleri (EXP-011 ile aynı), split dosyaları 5×5. Eşik: iç 5-fold OOF tahminlerinde dengeli doğruluğu en yükselten değer. "
             "Kalibrasyon: Brier (düşük iyi), ortalama tahmin − gerçek astım oranı (0 iyi), kalibrasyon eğimi (1 iyi). Bütün AUC'ler üst sınırdır (D-028).", "",
             "## Füzyon (7 görev)", "",
             "| model | yöntem | AUC ort ± SD | havuzlanmış AUC [%95 CI] | dengeli doğr. @0.5 | dengeli doğr. @iç-CV eşiği | Brier | ort. tahmin − oran | kalibrasyon eğimi |",
             "|---|---|---|---|---|---|---|---|---|"]
        for r_ in summ[summ.task == "fusion"].itertuples():
            L.append(f"| {r_.model} | {r_.strategy} | {r_.auc_mean:.3f} ± {r_.auc_sd:.3f} | {r_.pooled_auc:.3f} [{r_.ci_low:.3f}–{r_.ci_high:.3f}] | "
                     f"{r_.balacc_05:.3f} | {r_.balacc_thr:.3f} | {r_.brier:.3f} | {r_.mean_pred_minus_prev:+.3f} | {r_.calib_slope:.2f} |")
        L += ["", "## Eşleştirilmiş fark (füzyon, havuzlanmış AUC; katılımcı bootstrap)", "", "| model | karşılaştırma | ΔAUC [%95 CI] |", "|---|---|---|"]
        for c_ in comp:
            L.append(f"| {c_['model']} | {c_['comparison']} | {c_['delta_auc']:+.3f} [{c_['ci'][0]:+.3f}, {c_['ci'][1]:+.3f}] |")
        L += ["", "## Görev başına AUC (fold ortalaması)", "", "| görev | " + " | ".join(f"{m} {s}" for m in ["LR", "MLP"] for s in ["none", "class_weight", "SMOTE", "random_oversampling"]) + " |",
              "|---|" + "---|" * 8]
        for t in list(TASKS.values()):
            vals = [summ[(summ.task == t) & (summ.model == m) & (summ.strategy == s)].auc_mean.iloc[0]
                    for m in ["LR", "MLP"] for s in ["none", "class_weight", "SMOTE", "random_oversampling"]]
            L.append(f"| {t} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
        (args.report_dir / "EXP-013_imbalance.md").write_text("\n".join(L) + "\n")
        print("\n".join(L))

    elif args.part == "context":
        oof = exp014(F, C, feat, S)
        summ = summarize014(oof, C)
        fus = oof[oof.task != "reference"].groupby(["participant_id", "variant", "repeat", "fold"], as_index=False).prob.mean()
        piv = fus.groupby(["participant_id", "variant"]).prob.mean().unstack()
        comp = []
        base = "V0 sınıf ağırlığı (tüm veri)"
        e_ids = C.index[(C.period == "early")].intersection(piv.index)
        for v in piv.columns:
            if v == base:
                continue
            for name, ids in [("tüm test", piv.index), ("yalnız erken dönem", e_ids)]:
                d, lo, hi = paired(C.loc[ids, "label"].to_numpy(), piv.loc[ids, v].to_numpy(), piv.loc[ids, base].to_numpy())
                comp.append({"comparison": f"{v} − V0", "set": name, "delta_auc": d, "ci": [lo, hi]})
        cells = pd.crosstab(C.cell, C.label).rename(columns={0: "healthy", 1: "asthma"})
        rep = {"exp": "EXP-014", "features_sha256": feats_sha, "cells": cells.to_dict(orient="index"),
               "summary": summ.round(4).to_dict(orient="records"), "paired_fusion": comp, "runtime_s": round(time.time() - t0, 1)}
        (args.report_dir / "EXP-014_context_balancing.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=float))
        L = ["# EXP-014 — Kayıt bağlamı dengesizliği SMOTE / ağırlıkla düzeltilebilir mi? (keşifsel; katılımcı ID'si içermez)", "",
             "Önbellek özellikleri, LR, füzyon (7 görev), split dosyaları 5×5. Hücre = dönem (erken/geç; D-024) × seans başlangıcı (sabah/öğleden sonra).", "",
             "## Veri yapısı: hücrelerde katılımcı sayısı (pozitiflik)", "", "| hücre | astım | sağlıklı |", "|---|---|---|"]
        for c_, v in cells.iterrows():
            L.append(f"| {c_} | {v.get('asthma', 0)} | {v.get('healthy', 0)} |")
        L += ["", "## Sonuçlar (füzyon; tekrar-ortalamalı OOF)", "",
              "| varyant / referans | AUC tüm test | AUC yalnız erken dönem | hücre-içi AUC (erken-AM + erken-PM) | erken-AM | erken-PM | hastalarda skor: geç vs erken | hastalarda skor: sabah vs öğleden sonra | sağlıklılarda skor: sabah vs öğleden sonra |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r_ in summ.itertuples():
            L.append(f"| {r_.variant} | {r_.auc_all:.3f} | {r_.auc_early_period_only:.3f} | {r_.auc_within_cells_early_AM_PM:.3f} | {r_.auc_early_AM_cell:.3f} | "
                     f"{r_.auc_early_PM_cell:.3f} | {r_.context_reliance_patients_late_vs_early:.3f} | {r_.context_reliance_patients_AM_vs_PM:.3f} | {r_.context_reliance_healthy_AM_vs_PM:.3f} |")
        L += ["", "Son üç sütun \"bağlam bağımlılığı\": etiket sabitken skorun bağlamı ayırma gücü (0.5 = bağımlılık yok). "
              "Hastalarda geç > 0.5 → geç dönem hastalarına daha yüksek 'astım' skoru.", "",
              "## Eşleştirilmiş fark (V0'a göre; katılımcı bootstrap)", "", "| karşılaştırma | küme | ΔAUC [%95 CI] |", "|---|---|---|"]
        for c_ in comp:
            L.append(f"| {c_['comparison']} | {c_['set']} | {c_['delta_auc']:+.3f} [{c_['ci'][0]:+.3f}, {c_['ci'][1]:+.3f}] |")
        (args.report_dir / "EXP-014_context_balancing.md").write_text("\n".join(L) + "\n")
        print("\n".join(L))

    else:
        sel, oof = exp015(F, C, feat, S)
        out = []
        for task, g in sel.groupby("task", sort=False):
            freq = g.chosen.value_counts().to_dict()
            mcols = [c for c in g.columns if c.startswith("m_")]
            best_fixed = max(mcols, key=lambda c: g[c].mean())
            out.append({"task": task, "nested_auc_mean": g.nested_test_auc.mean(), "nested_auc_sd": g.nested_test_auc.std(ddof=1),
                        "best_fixed_model_25folds": best_fixed[2:], "best_fixed_auc_mean": g[best_fixed].mean(),
                        "oracle_per_fold_mean": g.oracle_test_auc.mean(), "selection_frequency": freq,
                        "median_model_auc_mean": float(np.median([g[c].mean() for c in mcols]))})
        # füzyon: her görevde o fold'da seçilen modelin olasılığı
        pick = sel[["task", "repeat", "fold", "chosen"]].rename(columns={"chosen": "model"})
        nf = oof.merge(pick, on=["task", "repeat", "fold", "model"]).groupby(["participant_id", "repeat", "fold"], as_index=False).prob.mean()
        fa = [roc_auc_score(C.loc[g.participant_id, "label"], g.prob) for _, g in nf.groupby(["repeat", "fold"])]
        pooled = nf.groupby("participant_id").prob.mean()
        lo, hi = rb.boot_ci(C.loc[pooled.index, "label"].to_numpy(), pooled.to_numpy())
        res = pd.DataFrame(out)
        rep = {"exp": "EXP-015", "features_sha256": feats_sha, "models": sorted(c[2:] for c in sel.columns if c.startswith("m_")),
               "per_task": res.round(4).to_dict(orient="records"),
               "nested_fusion": {"auc_mean": float(np.mean(fa)), "auc_sd": float(np.std(fa, ddof=1)),
                                 "pooled_auc": float(roc_auc_score(C.loc[pooled.index, "label"], pooled)), "ci": [lo, hi]},
               "runtime_s": round(time.time() - t0, 1)}
        (args.report_dir / "EXP-015_nested_cv.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=float))
        L = ["# EXP-015 — Makalenin model seçim prosedürü için iç içe (nested) CV (keşifsel; katılımcı ID'si içermez)", "",
             f"Makale özellikleri, makale pipeline'ı (StandardScaler → SMOTE → model), {len(rep['models'])} model: {', '.join(rep['models'])}. "
             "Dış: split dosyaları 5×5 (25 fold). İç: her dış train içinde 5-fold; model iç ortalama AUC'ye göre seçilir, dış test seçime karışmaz.", "",
             "| görev | iç içe CV AUC (seçim dahil) | 25 fold'da sabit en iyi model (sonradan bakarak) | her fold'da en iyiyi seçmek (kâhin, üst sınır) | 10 modelin medyanı | seçim sıklığı (25 fold) |",
             "|---|---|---|---|---|---|"]
        for r_ in res.itertuples():
            L.append(f"| {r_.task} | {r_.nested_auc_mean:.3f} ± {r_.nested_auc_sd:.3f} | {r_.best_fixed_model_25folds} {r_.best_fixed_auc_mean:.3f} | "
                     f"{r_.oracle_per_fold_mean:.3f} | {r_.median_model_auc_mean:.3f} | {r_.selection_frequency} |")
        nfz = rep["nested_fusion"]
        L += ["", f"**Füzyon (her görevde iç CV'nin seçtiği model):** {nfz['auc_mean']:.3f} ± {nfz['auc_sd']:.3f}; "
                  f"havuzlanmış {nfz['pooled_auc']:.3f} [{nfz['ci'][0]:.3f}–{nfz['ci'][1]:.3f}]", "", f"Süre: {rep['runtime_s']} s"]
        (args.report_dir / "EXP-015_nested_cv.md").write_text("\n".join(L) + "\n")
        print("\n".join(L))


if __name__ == "__main__":
    main()
