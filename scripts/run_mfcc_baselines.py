"""
Faz 1 — MFCC baseline'ları: EXP-010 (sadık yeniden üretim) ve EXP-011 (bizim protokol)  (D-012, D-028, D-031)

BİLİMSEL AMAÇ
    EXP-010  Yayınlanmış çalışmanın değerlendirme protokolünü olduğu gibi uygulamak: görev başına tek bir
             StratifiedKFold(5, shuffle, random_state=42); fold içinde StandardScaler → SMOTE → sınıflandırıcı;
             makaledeki 14 model (varsayılan ayarlar); her görevde "en iyi" = en yüksek ortalama fold AUC'si
             (makalenin kuralı — test fold'larına bakarak seçim, iyimser). Soru: makaledeki AUC aralığına
             (≈0.65–0.77) ulaşıyor muyuz? Bütün modeller raporlanır; "en iyi"nin iyimserliği görünür olur.
    EXP-011  Aynı 44 özellik (harmonize önbellekten), BİZİM protokolümüz: ortak split dosyaları (5 tekrar × 5 fold,
             D-029), önceden belirlenmiş model ailesi (LR = birincil, SVM-RBF, gradient boosting), düzenlileştirme
             iç 5-fold'da seçilir, test fold'larına bakarak seçim YOK, görev başına + katılımcı düzeyinde füzyon
             (7 görevin olasılık ortalaması), aynı fold'larda referans çizgileri (yalnız yaş, yalnız bağlam).
             Bu, derin modellerin karşılaştırılacağı RQ1 tabanıdır.
    Her iki deneyde sonuçlar "üst sınır" olarak raporlanır (D-028): kayıt bağlamı tek başına AUC ~0.93.

SIZINTI KONTROLLERİ
    Fold'lar katılımcı düzeyinde; her fold'da train ∩ test = ∅ assert'i; ölçekleme / SMOTE / ayar yalnız train'de
    (Pipeline içinde). Füzyon, her katılımcının yalnız kendisini görmemiş modellerden gelen OOF olasılıklarını kullanır.

GİRDİLER
    --features   extract_mfcc_features.py çıktısı (participant_id, slot, 44 özellik)
    --context    participant_context.csv (label, age, start_hour, days)
    --splits-dir outer_r{r}.csv (yalnız EXP-011)
ÇIKTILAR
    <out_dir> (Drive experiments/EXP-01x_mfcc/): oof_predictions.csv, fold_metrics.csv, partial/ (kaldığı yerden devam)
    <report_dir> (git): EXP-01x_mfcc.{json,md}  ·  --registry verilirse results/registry.csv'ye satırlar eklenir
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier,
                              StackingClassifier, VotingClassifier)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit, StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

TASKS = {1: "aaa", 2: "araba", 3: "ana", 4: "ordu", 5: "gelecek", 6: "titiz", 7: "ünlem"}
SEED = 42
REFERENCE = {"context (hour+hour²+date)": ["start_hour", "hour2", "days"], "age": ["age"],
             "age+context": ["age", "start_hour", "hour2", "days"]}


# ----------------------------------------------------------------------------- EXP-010 modelleri (makale)
class WeightedSoftVoting(BaseEstimator, ClassifierMixin):
    """Makalenin "Weighted Voting Ensemble"ı: ağırlık ∝ doğrulama AUC'si. Makale hangi doğrulama setini
    kullandığını yazmıyor (A6) → burada SIZINTISIZ seçenek: train içinde 3-fold OOF AUC [DECISION, D-031]."""

    def __init__(self, estimators=None, cv=3, random_state=SEED):
        self.estimators = estimators
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        cv = StratifiedKFold(self.cv, shuffle=True, random_state=self.random_state)
        aucs = [roc_auc_score(y, cross_val_predict(clone(e), X, y, cv=cv, method="predict_proba")[:, 1])
                for _, e in self.estimators]
        self.weights_ = np.array(aucs) / np.sum(aucs)
        self.fitted_ = [clone(e).fit(X, y) for _, e in self.estimators]
        return self

    def predict_proba(self, X):
        return np.sum([w * e.predict_proba(X) for w, e in zip(self.weights_, self.fitted_)], axis=0)

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


def paper_models() -> dict:
    from catboost import CatBoostClassifier
    from xgboost import XGBClassifier

    def ens_base():
        return [("gb", GradientBoostingClassifier(random_state=SEED)),
                ("xgb", XGBClassifier(random_state=SEED, eval_metric="logloss", n_jobs=2)),
                ("cat", CatBoostClassifier(random_state=SEED, verbose=0, thread_count=2, allow_writing_files=False)),
                ("mlp", MLPClassifier(random_state=SEED))]
    return {
        "NaiveBayes": GaussianNB(), "KNN": KNeighborsClassifier(), "DecisionTree": DecisionTreeClassifier(random_state=SEED),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=SEED),
        "SVM": SVC(probability=True, random_state=SEED), "MLP": MLPClassifier(random_state=SEED),
        "RandomForest": RandomForestClassifier(random_state=SEED), "AdaBoost": AdaBoostClassifier(random_state=SEED),
        "GradientBoosting": GradientBoostingClassifier(random_state=SEED),
        "CatBoost": CatBoostClassifier(random_state=SEED, verbose=0, thread_count=2, allow_writing_files=False),
        "XGBoost": XGBClassifier(random_state=SEED, eval_metric="logloss", n_jobs=2),
        "VotingEnsemble": VotingClassifier(ens_base(), voting="soft"),
        "StackingEnsemble": StackingClassifier(ens_base(), final_estimator=LogisticRegression(max_iter=1000),
                                               cv=5, stack_method="predict_proba"),
        "WeightedVotingEnsemble": WeightedSoftVoting(ens_base()),
    }


# ----------------------------------------------------------------------------- EXP-011 modelleri (bizim)
def our_models() -> dict:
    """(pipeline, parametre ızgarası, örnek ağırlığı gerekir mi). Izgaralar önceden sabit [DECISION, D-031]."""
    return {
        "LR": (Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(class_weight="balanced", max_iter=5000))]),
               {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100]}, False),
        "SVM-RBF": (Pipeline([("sc", StandardScaler()), ("clf", SVC(class_weight="balanced", probability=True, random_state=SEED))]),
                    {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01, 0.1]}, False),
        "GradientBoosting": (Pipeline([("sc", StandardScaler()), ("clf", GradientBoostingClassifier(random_state=SEED, learning_rate=0.05))]),
                             {"clf__n_estimators": [100, 300], "clf__max_depth": [2, 3]}, True),
    }


# ----------------------------------------------------------------------------- yardımcılar
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).resolve().parent).stdout.strip()
    except Exception:
        return ""


def fold_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    yhat = (p >= 0.5).astype(int)
    tp, tn = int(((yhat == 1) & (y == 1)).sum()), int(((yhat == 0) & (y == 0)).sum())
    return {"auc": roc_auc_score(y, p), "balanced_accuracy": balanced_accuracy_score(y, yhat),
            "accuracy": float((yhat == y).mean()), "f1": f1_score(y, yhat, zero_division=0),
            "sensitivity": tp / max(1, int((y == 1).sum())), "specificity": tn / max(1, int((y == 0).sum()))}


def boot_ci(y: np.ndarray, p: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Katılımcı bootstrap'ı (D-010): katılımcılar yerine konarak yeniden örneklenir."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            vals.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def load(features: Path, context: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    F = pd.read_csv(features)
    C = pd.read_csv(context)
    feat = [c for c in F.columns if c not in {"participant_id", "slot", "kept_s"}]
    assert len(feat) == 44, f"44 özellik bekleniyordu, {len(feat)} var"
    assert not F.duplicated(["participant_id", "slot"]).any()
    assert F[feat].notna().all().all()
    assert C["participant_id"].is_unique
    missing = set(F.participant_id) - set(C.participant_id)
    assert not missing, f"bağlam tablosunda olmayan {len(missing)} katılımcı"
    C = C[C.participant_id.isin(F.participant_id)].copy()
    C["hour2"] = C["start_hour"] ** 2
    assert C["label"].isin([0, 1]).all()
    return F, C.set_index("participant_id"), feat


# ----------------------------------------------------------------------------- EXP-010
def run_exp010(F, C, feat, out_dir: Path, models_sel: list[str] | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = paper_models()
    if models_sel:
        models = {k: v for k, v in models.items() if k in models_sel}
    fold_rows, oof_rows = [], []
    for slot, task in TASKS.items():
        part = out_dir / "partial" / f"EXP-010_{slot}.csv"
        part_oof = out_dir / "partial" / f"EXP-010_{slot}_oof.csv"
        if part.exists() and part_oof.exists():
            got = pd.read_csv(part)
            if set(got.model) >= set(models):
                print(f"  görev {task}: önceki sonuç kullanıldı ({part.name})")
                fold_rows.append(got)
                oof_rows.append(pd.read_csv(part_oof))
                continue
        d = F[F.slot == slot].sort_values("participant_id")
        X, y = d[feat].to_numpy(), C.loc[d.participant_id, "label"].to_numpy()
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
        rows, task_oof = [], []
        for name, est in models.items():
            t0 = time.time()
            for k, (tr, te) in enumerate(skf.split(X, y)):
                assert not set(d.participant_id.iloc[tr]) & set(d.participant_id.iloc[te])
                pipe = ImbPipeline([("sc", StandardScaler()), ("smote", SMOTE(random_state=SEED)), ("clf", clone(est))])
                pipe.fit(X[tr], y[tr])
                p = pipe.predict_proba(X[te])[:, 1]
                rows.append({"exp": "EXP-010", "task": task, "slot": slot, "model": name, "repeat": 0, "fold": k,
                             "n_test": len(te), **fold_metrics(y[te], p)})
                task_oof.append(pd.DataFrame({"participant_id": d.participant_id.iloc[te].to_numpy(), "task": task,
                                              "model": name, "repeat": 0, "fold": k, "prob": p}))
            print(f"  görev {task:8s} {name:24s} AUC {np.mean([r['auc'] for r in rows if r['model'] == name]):.3f} "
                  f"({time.time() - t0:.0f} s)", flush=True)
        df, toof = pd.DataFrame(rows), pd.concat(task_oof, ignore_index=True)
        part.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(part, index=False)
        toof.to_csv(part_oof, index=False)
        fold_rows.append(df)
        oof_rows.append(toof)
    oof = pd.concat(oof_rows, ignore_index=True)
    return pd.concat(fold_rows, ignore_index=True), oof


# ----------------------------------------------------------------------------- EXP-011
def run_exp011(F, C, feat, splits_dir: Path, out_dir: Path, models_sel, n_repeats: int):
    models = our_models()
    if models_sel:
        models = {k: v for k, v in models.items() if k in models_sel}
    fold_rows, oof_rows = [], []
    for r in range(n_repeats):
        S = pd.read_csv(splits_dir / f"outer_r{r}.csv")
        assert set(S.participant_id) >= set(F.participant_id), "split dosyasında olmayan katılımcı"
        part = out_dir / "partial" / f"EXP-011_r{r}.csv"
        part_oof = out_dir / "partial" / f"EXP-011_r{r}_oof.csv"
        if part.exists() and part_oof.exists():
            got = pd.read_csv(part)
            if set(got.model) >= set(models) | set(REFERENCE):
                print(f"  tekrar {r}: önceki sonuç kullanıldı")
                fold_rows.append(got)
                oof_rows.append(pd.read_csv(part_oof))
                continue
        rows, oofs = [], []
        t0 = time.time()
        for k in range(5):
            B = S[S.outer_fold == k].set_index("participant_id")
            test_ids, train_ids = set(B.index[B.role == "test"]), set(B.index[B.role == "train"])
            assert not test_ids & train_ids, f"r{r} k{k}: SIZINTI"
            # Görev başına modeller
            for slot, task in TASKS.items():
                d = F[F.slot == slot]
                tr, te = d[d.participant_id.isin(train_ids)], d[d.participant_id.isin(test_ids)]
                ytr, yte = C.loc[tr.participant_id, "label"].to_numpy(), C.loc[te.participant_id, "label"].to_numpy()
                inner = PredefinedSplit(B.loc[tr.participant_id, "inner_fold"].to_numpy())
                for name, (pipe, grid, needs_sw) in models.items():
                    gs = GridSearchCV(clone(pipe), grid, cv=inner, scoring="neg_log_loss", refit=True)
                    fit_kw = {"clf__sample_weight": compute_sample_weight("balanced", ytr)} if needs_sw else {}
                    gs.fit(tr[feat].to_numpy(), ytr, **fit_kw)
                    p = gs.predict_proba(te[feat].to_numpy())[:, 1]
                    rows.append({"exp": "EXP-011", "task": task, "slot": slot, "model": name, "repeat": r, "fold": k,
                                 "n_test": len(te), "best_params": json.dumps(gs.best_params_), **fold_metrics(yte, p)})
                    oofs.append(pd.DataFrame({"participant_id": te.participant_id.to_numpy(), "task": task, "model": name,
                                              "repeat": r, "fold": k, "prob": p}))
            # Referans çizgileri aynı fold'larda (D-005, D-028): sesi kullanmayan lojistik regresyon
            for ref, cols in REFERENCE.items():
                ok = C.dropna(subset=cols)
                tr_ids, te_ids = [i for i in ok.index if i in train_ids], [i for i in ok.index if i in test_ids]
                m = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(class_weight="balanced", max_iter=5000))])
                m.fit(ok.loc[tr_ids, cols], ok.loc[tr_ids, "label"])
                p = m.predict_proba(ok.loc[te_ids, cols])[:, 1]
                rows.append({"exp": "EXP-011", "task": "reference", "slot": 0, "model": ref, "repeat": r, "fold": k,
                             "n_test": len(te_ids), "best_params": "{}", **fold_metrics(ok.loc[te_ids, "label"].to_numpy(), p)})
                oofs.append(pd.DataFrame({"participant_id": te_ids, "task": "reference", "model": ref, "repeat": r,
                                          "fold": k, "prob": p}))
        oof_r = pd.concat(oofs, ignore_index=True)
        # Füzyon: katılımcının 7 görev olasılığının ortalaması (hepsi o katılımcıyı görmemiş modellerden)
        fus = oof_r[oof_r.task != "reference"].groupby(["participant_id", "model", "repeat", "fold"], as_index=False)["prob"].mean()
        fus["task"] = "fusion (7 görev)"
        for (name, k), g in fus.groupby(["model", "fold"]):
            yy = C.loc[g.participant_id, "label"].to_numpy()
            rows.append({"exp": "EXP-011", "task": "fusion (7 görev)", "slot": -1, "model": name, "repeat": r, "fold": k,
                         "n_test": len(g), "best_params": "{}", **fold_metrics(yy, g.prob.to_numpy())})
        oof_r = pd.concat([oof_r, fus], ignore_index=True)
        df = pd.DataFrame(rows)
        part.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(part, index=False)
        oof_r.to_csv(part_oof, index=False)
        fold_rows.append(df)
        oof_rows.append(oof_r)
        fus_auc = df[df.task.str.startswith("fusion")].groupby("model").auc.mean().round(3).to_dict()
        print(f"  tekrar {r} bitti ({time.time() - t0:.0f} s); füzyon AUC (fold ort.) {fus_auc}", flush=True)
    return pd.concat(fold_rows, ignore_index=True), pd.concat(oof_rows, ignore_index=True)


# ----------------------------------------------------------------------------- özet
def summarize(fm: pd.DataFrame, oof: pd.DataFrame, C: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (task, model), g in fm.groupby(["task", "model"], sort=False):
        row = {"task": task, "model": model, "n_folds": len(g), "auc_mean": g.auc.mean(), "auc_sd": g.auc.std(ddof=1),
               "auc_fold_p2.5": g.auc.quantile(0.025), "auc_fold_p97.5": g.auc.quantile(0.975),
               "balacc_mean": g.balanced_accuracy.mean(), "balacc_sd": g.balanced_accuracy.std(ddof=1),
               "sens_mean": g.sensitivity.mean(), "spec_mean": g.specificity.mean(), "acc_mean": g.accuracy.mean(),
               "f1_mean": g.f1.mean()}
        o = oof[(oof.task == task) & (oof.model == model)] if len(oof) else pd.DataFrame()
        if len(o):  # tekrarlar boyunca ortalama OOF olasılığı → havuzlanmış AUC + katılımcı bootstrap CI
            m = o.groupby("participant_id").prob.mean()
            yy = C.loc[m.index, "label"].to_numpy()
            row["pooled_auc"] = roc_auc_score(yy, m.to_numpy())
            row["pooled_ci_low"], row["pooled_ci_high"] = boot_ci(yy, m.to_numpy())
            row["n_participants"], row["n_asthma"] = len(m), int(yy.sum())
        out.append(row)
    return pd.DataFrame(out)


def registry_rows(exp, summ, feats_sha, split_sha, ver) -> list[dict]:
    rows = []
    for _, r in summ.iterrows():
        is_ref = r["task"] == "reference"
        n, n_a = r.get("n_participants", np.nan), r.get("n_asthma", np.nan)
        rows.append({
            "experiment_id": exp, "date": time.strftime("%Y-%m-%d"), "status": "done", "research_question": "RQ1",
            "n_participants": n, "n_patients": n_a, "n_controls": n - n_a if pd.notna(n) else np.nan,
            "n_recordings": "", "n_segments": 0, "cohort": "audio cohort (342; görev 4: 341)", "recording_types": r["task"],
            "split_strategy": "StratifiedKFold(5, shuffle, rs=42), participant-level" if exp == "EXP-010"
            else "split files outer_r0..4 (D-029), participant-level", "split_file_sha256": split_sha,
            "input_representation": "none (context/age)" if is_ref else f"44-d MFCC+spectral summary (features sha256 {feats_sha[:12]})",
            "sampling_rate": "" if is_ref else 22050,
            "preprocessing": "" if is_ref else ("orig m4a → 22.05 kHz → librosa trim top_db=60" if exp == "EXP-010"
                                                else "harmonized cache 32k (D-030) → 22.05 kHz"),
            "model": r["model"], "pretrained_weights": "none",
            "imbalance_strategy": "SMOTE (train fold)" if exp == "EXP-010" else "class_weight / sample_weight balanced",
            "augmentation": "none",
            "hyperparameters": "library defaults (paper)" if exp == "EXP-010" else "inner 5-fold grid, neg_log_loss (D-031)",
            "seeds": 42, "n_folds_total": r["n_folds"], "metric_level": "participant", "primary_metric": "roc_auc",
            "mean": round(r["auc_mean"], 4), "sd": round(r["auc_sd"], 4),
            "ci_low": round(r.get("pooled_ci_low", np.nan), 4), "ci_high": round(r.get("pooled_ci_high", np.nan), 4),
            "fold_p2_5": round(r["auc_fold_p2.5"], 4), "fold_p97_5": round(r["auc_fold_p97.5"], 4),
            "balanced_accuracy_mean": round(r["balacc_mean"], 4), "balanced_accuracy_sd": round(r["balacc_sd"], 4),
            "hardware": "Colab CPU", "git_commit": ver["git_commit"], "script": "scripts/run_mfcc_baselines.py",
            "result_location": f"reports/mfcc/{exp}_mfcc.json",
            "interpretation": "reference line (no audio)" if is_ref else "upper bound — recording context alone AUC ~0.93 (D-028)",
            "known_limitations": "single split; best-of-14 selected on test folds (paper rule)" if exp == "EXP-010"
            else "context confounding not yet evaluated (D-028)"})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=["EXP-010", "EXP-011"])
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--context", required=True, type=Path)
    ap.add_argument("--splits-dir", type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--report-dir", required=True, type=Path)
    ap.add_argument("--registry", type=Path)
    ap.add_argument("--models", type=str, default="", help="virgülle ayrılmış alt küme (test için)")
    ap.add_argument("--n-repeats", type=int, default=5)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    sel = [m for m in args.models.split(",") if m] or None
    t0 = time.time()

    F, C, feat = load(args.features, args.context)
    print(f"{args.exp}: {F.participant_id.nunique()} katılımcı ({int(C.label.sum())} astım / {int((C.label == 0).sum())} sağlıklı), "
          f"{len(F)} kayıt")
    if args.exp == "EXP-010":
        fm, oof = run_exp010(F, C, feat, args.out_dir, sel)
        split_sha = ""
    else:
        assert args.splits_dir, "--splits-dir gerekli"
        fm, oof = run_exp011(F, C, feat, args.splits_dir, args.out_dir, sel, args.n_repeats)
        split_sha = ";".join(sha256_file(args.splits_dir / f"outer_r{r}.csv")[:12] for r in range(args.n_repeats))
    fm.to_csv(args.out_dir / "fold_metrics.csv", index=False)
    if len(oof):
        oof.to_csv(args.out_dir / "oof_predictions.csv", index=False)
    summ = summarize(fm, oof, C)

    ver = {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "git_commit": git_commit()}
    for mod in ["imblearn", "xgboost", "catboost", "librosa"]:
        try:
            ver[mod] = __import__(mod).__version__
        except Exception:
            pass
    feats_sha = sha256_file(args.features)
    rep = {"exp": args.exp, "features_sha256": feats_sha, "versions": ver, "runtime_s": round(time.time() - t0, 1),
           "n_participants": int(F.participant_id.nunique()), "summary": summ.round(4).to_dict(orient="records")}
    if args.exp == "EXP-010":
        best = summ[summ.task != "reference"].sort_values("auc_mean", ascending=False).groupby("task", sort=False).head(1)
        rep["paper_rule_best_per_task"] = best[["task", "model", "auc_mean", "auc_sd"]].round(4).to_dict(orient="records")
        med = summ.groupby("task").auc_mean.median()
        rep["median_over_models_per_task"] = med.round(4).to_dict()
    (args.report_dir / f"{args.exp}_mfcc.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=float))

    # Markdown rapor
    order = list(TASKS.values()) + ["fusion (7 görev)", "reference"]
    L = [f"# {args.exp} — MFCC baseline (otomatik üretildi; katılımcı ID'si içermez)", "",
         "**Üst sınır uyarısı (D-028):** Kayıt bağlamı (tarih + saat) sesi kullanmadan etiketi AUC ≈ 0.93 ile ele veriyor; "
         "aşağıdaki AUC'ler astım tespit başarısı değil, bağlamdan etkilenmiş olabilecek üst sınırlardır. "
         "Bağlam değerlendirmesi model geliştirme bitince yapılacak (D-023).", "",
         f"- Katılımcı: {rep['n_participants']}; özellik dosyası sha256 `{feats_sha[:16]}…`; süre {rep['runtime_s']} s",
         f"- Sürümler: {ver}", ""]
    if args.exp == "EXP-010":
        L += ["## Makalenin kuralıyla \"en iyi\" model (en yüksek ortalama fold AUC'si — test fold'larına bakarak seçim)", "",
              "| görev | model | AUC ort ± SD | 14 modelin medyan AUC'si | makale (Tablo 4 / Tablo 2) |", "|---|---|---|---|---|"]
        paper = {"aaa": "0.700 / 0.649", "ana": "0.690 / 0.650", "araba": "0.656 / 0.704", "ordu": "0.674 / 0.686",
                 "gelecek": "0.769 / 0.717", "titiz": "0.723 / 0.736", "ünlem": "0.749 / 0.726"}
        for b in rep["paper_rule_best_per_task"]:
            L.append(f"| {b['task']} | {b['model']} | {b['auc_mean']:.3f} ± {b['auc_sd']:.3f} | "
                     f"{rep['median_over_models_per_task'][b['task']]:.3f} | {paper.get(b['task'], '')} |")
        L += ["", "\"En iyi\" ile medyan arasındaki fark, 14 model arasından test sonuçlarına bakarak seçmenin iyimserliğinin kaba bir ölçüsüdür.", ""]
    L += ["## Bütün modeller", "",
          "| görev | model | fold AUC ort ± SD (fold %2.5–97.5) | havuzlanmış OOF AUC [%95 CI] | dengeli doğr. | duyarlılık / özgüllük |",
          "|---|---|---|---|---|---|"]
    for task in order:
        for r in summ[summ.task == task].itertuples():
            ci = f"{r.pooled_auc:.3f} [{r.pooled_ci_low:.3f}–{r.pooled_ci_high:.3f}]" if hasattr(r, "pooled_auc") and r.pooled_auc == r.pooled_auc else "—"
            L.append(f"| {task} | {r.model} | {r.auc_mean:.3f} ± {r.auc_sd:.3f} ({summ.loc[r.Index, 'auc_fold_p2.5']:.2f}–"
                     f"{summ.loc[r.Index, 'auc_fold_p97.5']:.2f}) | {ci} | {r.balacc_mean:.3f} | {r.sens_mean:.2f} / {r.spec_mean:.2f} |")
    (args.report_dir / f"{args.exp}_mfcc.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))

    if args.registry:
        reg = pd.read_csv(args.registry)
        old = int((reg.experiment_id == args.exp).sum())
        if old:  # aynı deneyin yeniden çalıştırılması: eski satırlar yenileriyle değiştirilir (parametre değiştiyse YENİ deney ID'si kullan)
            print(f"UYARI registry: {args.exp} için {old} eski satır yenileriyle değiştiriliyor")
            reg = reg[reg.experiment_id != args.exp]
        new = pd.DataFrame(registry_rows(args.exp, summ, feats_sha, split_sha, ver)).reindex(columns=reg.columns)
        pd.concat([reg, new], ignore_index=True).to_csv(args.registry, index=False)
        print(f"registry: {len(new)} satır yazıldı")


if __name__ == "__main__":
    main()
