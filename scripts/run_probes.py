"""
Faz 2 — Dondurulmuş gömmeler üzerinde lineer prob: EXP-016 (onaylayıcı), EXP-016S (kısa kayıt politikası
duyarlılığı), EXP-017 (katman, keşifsel), EXP-018 (tüm kayıt vs 4 s, keşifsel)
(D-032, D-033, D-034, D-035; docs/PHASE2_DESIGN.md)

BİLİMSEL AMAÇ
    Altı backbone'un temsili, astım / sağlıklı ayrımında MFCC'den daha fazla doğrusal olarak çözülebilir bilgi
    taşıyor mu (RQ1)? Backbone'lar birbirinden farklı mı (RQ3)? Sınıflandırıcı EXP-011'deki LR'nin AYNISI;
    tek değişken temsil. MFCC kolları (mfcc_lr, mfcc_mlp) aynı kodla, aynı fold'larda yeniden çalıştırılır.

KOLLAR (arm) — adlar benzersizdir; ara sonuçlar ad + imza ile saklanır ve deneyler arasında yeniden kullanılır
    <backbone>          win4 görünümü, birincil temsil (PANNs: fc1; SSL: katman 1..L, fold içi z-skoru + ortalama)
    <backbone>@full     tüm kayıt görünümü, birincil temsil (EXP-018)
    <backbone>@L<l>     tek katman l (EXP-017; hiçbir seçimde kullanılmaz)
    <backbone>@alt      win4, birincil temsil, ama 4 s'den kısa kayıtlarda DİĞER dolgu politikası (EXP-016S)
    mfcc_lr / mfcc_mlp  44 MFCC özelliği; LR (aynı ızgara) / StandardScaler → SMOTE → MLP (D-032)
    ref_context / ref_age  sesi kullanmayan referans çizgileri (D-005, D-028)

PROTOKOL (her dış fold × görev × kol)
    C: iç 5-fold (split dosyasındaki inner_fold) neg_log_loss, ızgara 1e-5…1e2 → tüm train'de yeniden fit → test
    iç OOF: seçilen C ile iç 5-fold cross_val_predict → eşik (dengeli doğruluk) ve iç doğrulama skorları
    füzyon: katılımcının görev olasılıklarının ortalaması (test ve iç OOF için ayrı ayrı)

SIZINTI KONTROLLERİ
    Katılımcı düzeyinde: bir katılımcının 7 kaydı da aynı role (train ya da dış test) düşer (assert, bütün görevlerde).
    Bütün kollar aynı split dosyalarını, aynı katılımcı kümesini ve aynı değerlendirme birimini (katılımcı) kullanır (assert).
    LayerAverage / StandardScaler / SMOTE yalnız train'de fit (Pipeline); C ve eşik yalnız iç döngüden.
    D-035: Faz 3 seçimi her dış fold'da YALNIZ o fold'un eğitim katılımcılarının iç OOF tahminleriyle yapılır
    (select_per_fold; dış test tahminlerini parametre olarak almaz; iç satırların eğitim rolünde olduğu assert edilir).

AYRI RAPORLAR
    Kayıt bağlamı / süre gibi meta veri karşılaştırmaları bu script'te YOK → scripts/report_metadata_monitor.py
    (META-016). Ana tabloda yalnız D-028'in iki referans satırı (bağlam, yaş) bulunur; onlarla test yapılmaz.

ÇIKTILAR
    <out_dir> (Drive experiments/frozen_probe/): partial/<kol>_r<r>_{test,inner}.csv + .json (imza)
    <out_dir>/d035_selection.csv  fold başına D-035 seçimi (katılımcı ID'si içermez; Faz 3'ün girdisi)
    <report_dir> (git): EXP-016/016S/017/018_frozen.{json,md} (yalnız agrega)  ·  --registry → results/registry.csv
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_imbalance_selection as ais  # noqa: E402  (eşik, kalibrasyon, bağlam hücreleri: EXP-013/014 ile AYNI)
import run_mfcc_baselines as rb  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

TASKS, SEED = rb.TASKS, rb.SEED
C_GRID = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
BACKBONES = ["cnn10", "cnn14", "cnn14_16k", "beats", "wavlm_base_plus", "wavlm_large"]
SSL = {"beats", "wavlm_base_plus", "wavlm_large"}
D035_CANDIDATES = ["beats", "wavlm_base_plus", "wavlm_large"]
PARAMS_APPROX = {"cnn10": 5.2e6, "cnn14": 80.8e6, "cnn14_16k": 80.8e6, "beats": 90e6, "wavlm_base_plus": 94.7e6,
                 "wavlm_large": 315.5e6}   # yalnız info.json'da n_params yoksa (D-035 eşitlik kuralı)
EXP011_LR_FUSION = 0.771                   # EXP-011 LR füzyon fold ortalaması (regresyon testi, D-034 madde 5)
FUSION = "fusion"
ALPHA_ONE_SIDED = 0.025                    # tek yönlü; iki yönlü 0.05'in pozitif yarısıyla aynı katılık (D-034 madde 6)


# ----------------------------------------------------------------------------- dönüşüm
class LayerAverage(BaseEstimator, TransformerMixin):
    """(n, n_layers*dim) → (n, dim): her katman eğitim verisinde z-skoruna çevrilir, sonra katmanlar ortalanır.
    Katmanlar arası ölçek farkını (ör. WavLM Large stable-layer-norm) giderir; yalnız train'de fit (sızıntı yok)."""

    def __init__(self, n_layers: int = 1, dim: int = 1):
        self.n_layers = n_layers
        self.dim = dim

    def fit(self, X, y=None):
        Z = np.asarray(X).reshape(len(X), self.n_layers, self.dim)
        self.mean_ = Z.mean(0)
        sd = Z.std(0)
        self.std_ = np.where(sd < 1e-8, 1.0, sd)
        return self

    def transform(self, X):
        Z = np.asarray(X).reshape(len(X), self.n_layers, self.dim)
        return ((Z - self.mean_) / self.std_).mean(1)


def lr_pipe(layer_avg: tuple[int, int] | None = None) -> Pipeline:
    steps = [("la", LayerAverage(*layer_avg))] if layer_avg else []
    steps += [("sc", StandardScaler()), ("clf", LogisticRegression(class_weight="balanced", max_iter=5000))]
    return Pipeline(steps)


def mlp_pipe() -> ImbPipeline:
    return ImbPipeline([("sc", StandardScaler()), ("smote", SMOTE(random_state=SEED)), ("clf", MLPClassifier(random_state=SEED))])


# ----------------------------------------------------------------------------- istatistik
def _midrank(x: np.ndarray) -> np.ndarray:
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        a = i
        while a < n and z[a] == z[i]:
            a += 1
        t[i:a] = 0.5 * (i + a - 1) + 1
        i = a
    out = np.empty(n)
    out[j] = t
    return out


def delong_paired(y: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> tuple[float, float, float]:
    """Eşleştirilmiş DeLong testi (Sun & Xu 2014 hızlı algoritması). Döner: (AUC1 − AUC2, z, iki yönlü p)."""
    y = np.asarray(y).astype(bool)
    P = np.vstack([p1, p2]).astype(float)
    X, Y = P[:, y], P[:, ~y]
    m, n = X.shape[1], Y.shape[1]
    tx = np.array([_midrank(r) for r in X])
    ty = np.array([_midrank(r) for r in Y])
    tz = np.array([_midrank(r) for r in np.hstack([X, Y])])
    auc = tz[:, :m].sum(1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    d = float(auc[0] - auc[1])
    if var <= 0:
        return d, 0.0, 1.0
    z = d / np.sqrt(var)
    return d, float(z), float(2 * stats.norm.sf(abs(z)))


def _pair_matrix(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    pos, neg = p[y == 1], p[y == 0]
    return (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])


def _boot_draws(y: np.ndarray, n: int, seed: int):
    """rb.boot_ci / ais.paired ile AYNI çekiliş dizisi (aynı tohum, aynı atlama kuralı) → tekrar sayıları."""
    rng = np.random.default_rng(seed)
    N = len(y)
    for _ in range(n):
        i = rng.integers(0, N, N)
        if 0 < y[i].sum() < len(i):
            yield np.bincount(i, minlength=N)


def boot_ci_fast(y, p, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Katılımcı bootstrap'lı AUC CI'ı. rb.boot_ci ile aynı sonuç (aynı çekilişler); AUC, çift matrisi üzerinden
    ağırlıklı Mann–Whitney olarak hesaplanır (roc_auc_score'un ağır doğrulama yükü olmadan). Testte eşitlik sınanır."""
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    M, ip, ineg = _pair_matrix(y, p), np.where(y == 1)[0], np.where(y == 0)[0]
    vals = [c[ip] @ M @ c[ineg] / (c[ip].sum() * c[ineg].sum()) for c in _boot_draws(y, n, seed)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_fast(y, p1, p2, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """ais.paired ile aynı: (ΔAUC, %2.5, %97.5), katılımcı bootstrap'ı."""
    y = np.asarray(y).astype(int)
    M1, M2 = _pair_matrix(y, np.asarray(p1, float)), _pair_matrix(y, np.asarray(p2, float))
    ip, ineg = np.where(y == 1)[0], np.where(y == 0)[0]
    d = []
    for c in _boot_draws(y, n, seed):
        cp, cn = c[ip], c[ineg]
        den = cp.sum() * cn.sum()
        d.append((cp @ M1 @ cn - cp @ M2 @ cn) / den)
    return (float(roc_auc_score(y, p1) - roc_auc_score(y, p2)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def nadeau_bengio(d: np.ndarray, ratio: float, one_sided: bool = False) -> tuple[float, float]:
    """Düzeltilmiş tekrarlı-CV t-testi (Nadeau & Bengio 2003; D-011). one_sided=True: H1 ortalama Δ > 0."""
    d = np.asarray(d, float)
    J = len(d)
    v = d.var(ddof=1)
    if v == 0:
        t = 0.0 if d.mean() == 0 else float(np.sign(d.mean()) * np.inf)
    else:
        t = float(d.mean() / np.sqrt((1 / J + ratio) * v))
    if one_sided:
        return t, float(stats.t.sf(t, J - 1))
    return t, float(2 * stats.t.sf(abs(t), J - 1))


def holm(p: dict) -> dict:
    keys = sorted(p, key=p.get)
    m, out, run = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * p[k]))
        out[k] = run
    return out


# ----------------------------------------------------------------------------- veri
@functools.lru_cache(maxsize=None)
def sha256_short(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()[:16]


def load_context(path: Path, cohort: set) -> pd.DataFrame:
    """Bağlam tablosu split kohortuna süzülür; erken / geç sınırı (son sağlıklı kaydın tarihi) EXP-014'teki gibi
    yalnız ses kohortunda hesaplanır."""
    C = pd.read_csv(path)
    assert C.participant_id.is_unique
    missing = cohort - set(C.participant_id)
    assert not missing, f"bağlam tablosunda olmayan {len(missing)} katılımcı"
    C = C[C.participant_id.isin(cohort)].copy()
    assert C.label.isin([0, 1]).all()
    C["hour2"] = C["start_hour"] ** 2
    return ais.add_cells(C.set_index("participant_id"))


class Arm:
    """Bir kol: görev başına (katılımcı → satır) eşlemesi ve X matrisi + pipeline + ızgara + imza."""

    def __init__(self, name, pids_by_slot, X_by_slot, pipe, grid, signature):
        self.name, self.pids, self.X, self.pipe, self.grid, self.signature = name, pids_by_slot, X_by_slot, pipe, grid, signature


def emb_arm(emb_root: Path, backbone: str, view: str, layer: int | None, short_alt: bool = False) -> Arm:
    """Gömme kolu. Şekiller (ör. WavLM Large): rec_win4.npy (N=2393, n_store=25, D=1024) → birincil katmanlar 1..24
    seçilir (N, 24, 1024) → (N, 24·1024) düzleştirilir → Pipeline içinde LayerAverage (fit yalnız train) → (n, 1024)."""
    d = emb_root / backbone
    E = np.load(d / f"rec_{view}.npy", mmap_mode="r")
    meta = pd.read_csv(d / "recordings.csv")
    info = json.loads((d / "info.json").read_text())
    assert E.shape[0] == len(meta) and E.ndim == 3, f"{backbone}: gömme şekli {E.shape}"
    if "n_store" in info:
        assert E.shape[1:] == (info["n_store"], info["dim"]), f"{backbone}: {E.shape} ≠ info ({info['n_store']}, {info['dim']})"
    assert not meta.duplicated(["participant_id", "slot"]).any()
    assert (meta.row.to_numpy() == np.arange(len(meta))).all(), f"{backbone}: recordings.csv 'row' dizi sırasıyla aynı değil"
    policy = info.get("short_policy", "nopad")
    if short_alt:
        assert view == "win4" and layer is None, "alternatif dolgu politikası yalnız win4 birincil temsilde"
        z = np.load(d / "short_alt_win4.npz")
        E = np.array(E)                                   # kopya: yalnız kısa kayıt satırları değişir
        assert np.array_equal(z["rows"], meta.row[meta.dur_s < 4.0].to_numpy()), "kısa kayıt satırları uyuşmuyor"
        E[z["rows"]] = z["emb"]
        policy = str(z["policy"])
    if layer is None:
        layers = info.get("primary_layers", [0] if E.shape[1] == 1 else list(range(1, E.shape[1])))
        if E.shape[1] > 1:
            assert layers == list(range(1, E.shape[1])), f"{backbone}: birincil katmanlar 1..L olmalı (D-034 madde 3)"
        la = (len(layers), E.shape[2]) if len(layers) > 1 else None
        name = (backbone if view == "win4" else f"{backbone}@{view}") + ("@alt" if short_alt else "")
    else:
        layers, la, name = [layer], None, f"{backbone}@L{layer}"
    X2 = np.asarray(E[:, layers, :], dtype=np.float32).reshape(len(meta), -1)
    assert X2.shape == (len(meta), len(layers) * E.shape[2]) and np.isfinite(X2).all()
    pids, Xs = {}, {}
    for slot in TASKS:
        m = (meta.slot == slot).to_numpy()
        pids[slot], Xs[slot] = meta.participant_id.to_numpy()[m], X2[m]
    sig = {"kind": "emb", "backbone": backbone, "view": view, "layers": layers, "layer_average": bool(la),
           "grid": C_GRID, "emb_sha256": sha256_short(d / f"rec_{view}.npy"), "random_init": info.get("random_init", False),
           "short_policy": policy if view == "win4" else "n/a (tüm kayıt, dolgusuz)"}
    if short_alt:
        sig["short_alt_sha256"] = sha256_short(d / "short_alt_win4.npz")
    arm = Arm(name, pids, Xs, lr_pipe(la), {"clf__C": C_GRID}, sig)
    arm.n_params = info.get("n_params")
    return arm


def mfcc_arms(features: Path) -> list[Arm]:
    F = pd.read_csv(features)
    feat = [c for c in F.columns if c not in {"participant_id", "slot", "kept_s"}]
    assert len(feat) == 44 and not F.duplicated(["participant_id", "slot"]).any() and F[feat].notna().all().all()
    pids = {s: F.loc[F.slot == s, "participant_id"].to_numpy() for s in TASKS}
    Xs = {s: F.loc[F.slot == s, feat].to_numpy(np.float64) for s in TASKS}
    sha = sha256_short(features)
    return [Arm("mfcc_lr", pids, Xs, lr_pipe(), {"clf__C": C_GRID}, {"kind": "mfcc_lr", "grid": C_GRID, "features_sha256": sha}),
            Arm("mfcc_mlp", pids, Xs, mlp_pipe(), None, {"kind": "mfcc_mlp", "features_sha256": sha})]


# ----------------------------------------------------------------------------- eğitim
def fit_one(arm: Arm, Xtr, ytr, Xte, inner, n_jobs):
    cv = PredefinedSplit(inner)
    if arm.grid:
        gs = GridSearchCV(clone(arm.pipe), arm.grid, cv=cv, scoring="neg_log_loss", refit=True, n_jobs=n_jobs).fit(Xtr, ytr)
        est = clone(arm.pipe).set_params(**gs.best_params_)
        p_te = gs.predict_proba(Xte)[:, 1]
        c = float(gs.best_params_["clf__C"])
    else:
        est = clone(arm.pipe)
        p_te = clone(arm.pipe).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        c = np.nan
    p_in = cross_val_predict(est, Xtr, ytr, cv=cv, method="predict_proba", n_jobs=n_jobs)[:, 1]
    return p_te, p_in, c


def run_arm_repeat(arm: Arm, C: pd.DataFrame, S: pd.DataFrame, r: int, n_jobs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, inner_rows = [], []
    for k in range(5):
        Bk = S[S.outer_fold == k].set_index("participant_id")
        test_ids, train_ids = set(Bk.index[Bk.role == "test"]), set(Bk.index[Bk.role == "train"])
        assert not test_ids & train_ids, f"r{r} k{k}: SIZINTI"
        assert test_ids | train_ids == set(S.participant_id), f"r{r} k{k}: rolü olmayan katılımcı"
        for slot, task in TASKS.items():
            pid, X = arm.pids[slot], arm.X[slot]
            tr, te = np.isin(pid, list(train_ids)), np.isin(pid, list(test_ids))
            # katılımcı düzeyi: bu görevin kayıtları, katılımcının split rolüne göre atanır (aynı kişi iki tarafta olamaz)
            assert not (tr & te).any() and (tr | te).all()
            assert set(pid[tr]) <= train_ids and set(pid[te]) <= test_ids
            ytr, yte = C.loc[pid[tr], "label"].to_numpy(), C.loc[pid[te], "label"].to_numpy()
            assert 0 < ytr.sum() < len(ytr) and 0 < yte.sum() < len(yte)
            inner = Bk.loc[pid[tr], "inner_fold"].to_numpy()
            p_te, p_in, c = fit_one(arm, X[tr], ytr, X[te], inner, n_jobs)
            rows.append(pd.DataFrame({"participant_id": pid[te], "task": task, "arm": arm.name, "repeat": r, "fold": k,
                                      "prob": p_te, "C": c}))
            inner_rows.append(pd.DataFrame({"participant_id": pid[tr], "task": task, "arm": arm.name, "repeat": r,
                                            "fold": k, "prob": p_in}))
    return pd.concat(rows, ignore_index=True), pd.concat(inner_rows, ignore_index=True)


def run_references(C: pd.DataFrame, S: pd.DataFrame, r: int) -> pd.DataFrame:
    rows = []
    for k in range(5):
        Bk = S[S.outer_fold == k].set_index("participant_id")
        test_ids, train_ids = set(Bk.index[Bk.role == "test"]), set(Bk.index[Bk.role == "train"])
        for ref, cols in {"ref_context": ["start_hour", "hour2", "days"], "ref_age": ["age"]}.items():
            ok = C.dropna(subset=cols)
            tr = [i for i in ok.index if i in train_ids]
            te = [i for i in ok.index if i in test_ids]
            m = lr_pipe().set_params(clf__C=1.0).fit(ok.loc[tr, cols], ok.loc[tr, "label"])
            rows.append(pd.DataFrame({"participant_id": te, "task": "reference", "arm": ref, "repeat": r, "fold": k,
                                      "prob": m.predict_proba(ok.loc[te, cols])[:, 1], "C": 1.0}))
    return pd.concat(rows, ignore_index=True)


def run_all(arms: list[Arm], C, splits: dict, out_dir: Path, n_jobs: int, data_sig: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    part = out_dir / "partial"
    part.mkdir(parents=True, exist_ok=True)
    tests, inners = [], []
    for r, S in splits.items():
        for arm in arms:
            ft, fi, fj = part / f"{arm.name}_r{r}_test.csv", part / f"{arm.name}_r{r}_inner.csv", part / f"{arm.name}_r{r}.json"
            sig = {**arm.signature, "split_sha256": data_sig["splits"][r], "context_sha256": data_sig["context"]}
            if ft.exists() and fi.exists() and fj.exists() and json.loads(fj.read_text()) == sig:
                print(f"  {arm.name} r{r}: önceki sonuç kullanıldı", flush=True)
                tests.append(pd.read_csv(ft))
                inners.append(pd.read_csv(fi))
                continue
            t0 = time.time()
            te, inn = run_arm_repeat(arm, C, S, r, n_jobs)
            te.to_csv(ft, index=False)
            inn.to_csv(fi, index=False)
            fj.write_text(json.dumps(sig))
            tests.append(te)
            inners.append(inn)
            fus = te.groupby(["participant_id", "fold"]).prob.mean()
            print(f"  {arm.name} r{r}: {time.time() - t0:.0f} s; füzyon AUC (havuz) "
                  f"{roc_auc_score(C.loc[fus.index.get_level_values(0), 'label'], fus):.3f}", flush=True)
        tests.append(run_references(C, S, r))
    return pd.concat(tests, ignore_index=True), pd.concat(inners, ignore_index=True)


# ----------------------------------------------------------------------------- özet
def add_fusion(df: pd.DataFrame) -> pd.DataFrame:
    a = df[df.task != "reference"]
    fus = a.groupby(["participant_id", "arm", "repeat", "fold"], as_index=False).prob.mean().assign(task=FUSION)
    ref = df[df.task == "reference"].assign(task=FUSION)
    return pd.concat([a, fus, ref], ignore_index=True)


def fold_table(test: pd.DataFrame, inner: pd.DataFrame, C) -> pd.DataFrame:
    thr = {}
    for (arm, task, r, k), g in inner.groupby(["arm", "task", "repeat", "fold"]):
        thr[(arm, task, r, k)] = ais.best_threshold(C.loc[g.participant_id, "label"].to_numpy(), g.prob.to_numpy())
    out = []
    for (arm, task, r, k), g in test.groupby(["arm", "task", "repeat", "fold"]):
        y, p = C.loc[g.participant_id, "label"].to_numpy(), g.prob.to_numpy()
        t = thr.get((arm, task, r, k), 0.5)
        yh = (p >= t).astype(int)
        out.append({"arm": arm, "task": task, "repeat": r, "fold": k, "n_test": len(g), "auc": roc_auc_score(y, p),
                    "pr_auc_asthma": average_precision_score(y, p), "pr_auc_healthy": average_precision_score(1 - y, -p),
                    "threshold": t, "balacc": ais.bal_acc(y, p, t),
                    "sens": ((yh == 1) & (y == 1)).sum() / max(1, (y == 1).sum()),
                    "spec": ((yh == 0) & (y == 0)).sum() / max(1, (y == 0).sum()), "mcc": matthews_corrcoef(y, yh),
                    "C": g.C.iloc[0] if "C" in g else np.nan})
    return pd.DataFrame(out)


def pooled(test: pd.DataFrame, arm: str, task: str, C) -> pd.Series:
    return test[(test.arm == arm) & (test.task == task)].groupby("participant_id").prob.mean()


def summarize(test, FT, C) -> pd.DataFrame:
    out = []
    for (arm, task), g in FT.groupby(["arm", "task"], sort=False):
        pm = pooled(test, arm, task, C)
        y = C.loc[pm.index, "label"].to_numpy()
        lo, hi = boot_ci_fast(y, pm.to_numpy())
        allrows = test[(test.arm == arm) & (test.task == task)]
        row = {"arm": arm, "task": task, "n_folds": len(g), "auc_mean": g.auc.mean(), "auc_sd": g.auc.std(ddof=1),
               "auc_fold_p2_5": g.auc.quantile(0.025), "auc_fold_p97_5": g.auc.quantile(0.975),
               "pooled_auc": roc_auc_score(y, pm), "pooled_ci_low": lo, "pooled_ci_high": hi,
               "n_participants": len(pm), "n_asthma": int(y.sum()),
               "balacc_mean": g.balacc.mean(), "balacc_sd": g.balacc.std(ddof=1), "sens_mean": g.sens.mean(),
               "spec_mean": g.spec.mean(), "mcc_mean": g.mcc.mean(), "pr_auc_asthma": g.pr_auc_asthma.mean(),
               "pr_auc_healthy": g.pr_auc_healthy.mean(),
               **ais.calib(C.loc[allrows.participant_id, "label"].to_numpy(), allrows.prob.to_numpy())}
        cs = g.C.dropna()
        if len(cs) and task != FUSION:
            row["C_at_grid_edge"] = float(((cs <= min(C_GRID)) | (cs >= max(C_GRID))).mean())
        out.append(row)
    return pd.DataFrame(out)


def compare(test, FT, C, a: str, b: str, task: str, ratio: float) -> dict:
    fa = FT[(FT.arm == a) & (FT.task == task)].set_index(["repeat", "fold"]).auc
    fb = FT[(FT.arm == b) & (FT.task == task)].set_index(["repeat", "fold"]).auc
    d = (fa - fb).dropna()
    t, p = nadeau_bengio(d.to_numpy(), ratio)
    p1 = nadeau_bengio(d.to_numpy(), ratio, one_sided=True)[1]
    pa, pb = pooled(test, a, task, C), pooled(test, b, task, C)
    ids = pa.index.intersection(pb.index)
    y = C.loc[ids, "label"].to_numpy()
    dd, lo, hi = paired_fast(y, pa[ids].to_numpy(), pb[ids].to_numpy())
    dl = []
    for r in sorted(test.repeat.unique()):
        ta = test[(test.arm == a) & (test.task == task) & (test.repeat == r)].set_index("participant_id").prob
        tb = test[(test.arm == b) & (test.task == task) & (test.repeat == r)].set_index("participant_id").prob
        ii = ta.index.intersection(tb.index)
        dl.append(delong_paired(C.loc[ii, "label"].to_numpy(), ta[ii].to_numpy(), tb[ii].to_numpy())[2])
    return {"a": a, "b": b, "task": task, "fold_delta_mean": float(d.mean()), "fold_delta_sd": float(d.std(ddof=1)),
            "n_folds": int(len(d)), "nb_t": t, "nb_p": p, "nb_p_one_sided": p1, "pooled_delta": dd, "boot_ci_low": lo, "boot_ci_high": hi,
            "delong_p_per_repeat": dl, "delong_p_median": float(np.median(dl))}


def check_same_units(test: pd.DataFrame, arms: list[str]) -> None:
    """EXP-016: her (tekrar, fold, görev) için bütün kollar AYNI dış test katılımcılarında değerlendirilir."""
    t = test[test.arm.isin(arms)]
    for (r, k, task), g in t.groupby(["repeat", "fold", "task"]):
        sets = {a: frozenset(gg.participant_id) for a, gg in g.groupby("arm")}
        assert len(sets) == len(arms) and len(set(sets.values())) == 1, f"r{r} k{k} {task}: kollar farklı test kümesinde"


def select_per_fold(inner_raw: pd.DataFrame, C: pd.DataFrame, splits: dict, candidates: list[str],
                    params: dict, tie: float = 0.01) -> pd.DataFrame:
    """D-035 (Alper'in koşuluyla KABUL): her dış fold (r, k) için seçim YALNIZ o fold'un eğitim katılımcılarının iç
    5-fold OOF tahminlerinden yapılır. Bu fonksiyon dış test tahminlerini parametre olarak ALMAZ.
    Güvenceler: (1) iç satırların her katılımcısı split dosyasında o fold için role == 'train' (assert);
    (2) iç tahminler run_arm_repeat → fit_one içinde yalnız X[tr], y[tr] ile üretildi (cross_val_predict, iç fold'lar);
    (3) tests/test_phase2.py P7: dış test tahminleri bozulduğunda seçimin değişmediği sınanır.
    Ölçüt: aday başına, 7 görevin iç OOF olasılıklarının katılımcı ortalamasıyla (füzyon) AUC. En yükseğe `tie`'dan
    yakın olanlar arasından en az parametreli seçilir."""
    rows = []
    inn = inner_raw[inner_raw.arm.isin(candidates)]
    for (r, k), g in inn.groupby(["repeat", "fold"]):
        S = splits[r]
        role = S[S.outer_fold == k].set_index("participant_id").role
        assert (role.loc[g.participant_id.unique()] == "train").all(), f"r{r} k{k}: iç tahminlerde dış test katılımcısı var"
        assert set(g.participant_id) == set(role.index[role == "train"]), f"r{r} k{k}: iç tahminler eğitim kümesinin tamamını kapsamıyor"
        f = g.groupby(["arm", "participant_id"]).prob.mean().reset_index()
        sc = {a: float(roc_auc_score(C.loc[ff.participant_id, "label"], ff.prob)) for a, ff in f.groupby("arm")}
        assert set(sc) == set(candidates), f"r{r} k{k}: eksik aday {set(candidates) - set(sc)}"
        top = max(sc.values())
        tied = [a for a in candidates if top - sc[a] < tie]
        chosen = min(tied, key=lambda a: params[a])
        rows.append({"repeat": int(r), "fold": int(k), **{f"inner_auc_{a}": round(v, 4) for a, v in sc.items()},
                     "tied": ",".join(tied), "selected": chosen})
    assert len(rows) == 5 * len(splits), f"D-035: {len(rows)} fold seçildi, {5 * len(splits)} bekleniyordu"
    return pd.DataFrame(rows)


def procedure_estimate(sel: pd.DataFrame, FT: pd.DataFrame, candidates: list[str]) -> dict:
    """Seçim prosedürünün dürüst (iç içe) tahmini: her fold'da seçilen kolun o fold'daki dış test füzyon AUC'si.
    Dış test AUC'si burada yalnız DEĞERLENDİRME için okunur; seçim `sel`'de önceden yapılmıştır."""
    ft = FT[(FT.task == FUSION) & FT.arm.isin(candidates)].set_index(["arm", "repeat", "fold"]).auc
    got = np.array([ft[(s.selected, s.repeat, s.fold)] for s in sel.itertuples()])
    fixed = ft.groupby(level="arm").mean()
    return {"candidates": candidates, "procedure_auc_mean": float(got.mean()), "procedure_auc_sd": float(got.std(ddof=1)),
            "best_fixed_arm_post_hoc": str(fixed.idxmax()), "best_fixed_auc_mean": float(fixed.max()),
            "winners_curse": float(fixed.max() - got.mean()), "chosen_counts": sel.selected.value_counts().to_dict(),
            "n_folds": int(len(sel))}


# ----------------------------------------------------------------------------- rapor
def fmt_ci(v, lo, hi):
    return f"{v:.3f} [{lo:.3f}–{hi:.3f}]"


def report_md(exp, rep, summ) -> str:
    L = [f"# {exp} — dondurulmuş gömme + lineer prob (otomatik üretildi; katılımcı ID'si içermez)", "",
         f"**Rol:** {rep['role']}  ", "**Üst sınır uyarısı (D-028):** bütün AUC'ler kayıt bağlamı değerlendirilmeden hesaplandı. "
         "Aynı fold'larda yalnız bağlam (saat + saat² + tarih) ve yalnız yaş referansları aşağıda.", "",
         f"Split: {rep['split_files']} · tekrar: {rep['n_repeats']} · test/train oranı (Nadeau–Bengio): {rep['test_train_ratio']:.3f}", "",
         f"**İstatistik notu:** Bütün Nadeau–Bengio (NB) testleri {rep['n_repeats'] * 5} fold skoruna ({rep['n_repeats']} tekrar × 5 dış fold) "
         "uygulanır. Bu skorlar **bağımsız değildir**: aynı tekrardaki fold'ların eğitim kümeleri büyük ölçüde örtüşür ve tekrarlar aynı "
         "katılımcıları yeniden böler. NB düzeltmesi (varyansa n_test/n_train terimi eklenir) bu bağımlılığı kabaca telafi eden sezgisel "
         f"bir düzeltmedir; df = {rep['n_repeats'] * 5 - 1} yaklaşıktır. Düz t-testi burada fazla iyimser olurdu.", "",
         "Değerlendirme birimi: **katılımcı**. Bütün kollar aynı split dosyalarında, aynı dış test katılımcılarında "
         "değerlendirildi (assert). Kayıt bağlamı / süre gibi meta veri karşılaştırmaları bu raporda yok → META-016.", "",
         "## Füzyon (katılımcı düzeyi; 7 görev olasılığının ortalaması) — betimsel", "",
         "| kol | AUC fold ort ± SD [p2.5–p97.5] | havuzlanmış AUC [%95 CI] | dengeli doğr. (iç-CV eşiği) | duyarlılık / özgüllük | Brier | ort. tahmin − oran | kal. eğimi |",
         "|---|---|---|---|---|---|---|---|"]
    for r in summ[summ.task == FUSION].itertuples():
        thr_note = " (eşik 0.5)" if r.arm.startswith("ref_") else ""
        name = f"{r.arm} — referans çizgisi (D-028; test edilmez)" if r.arm.startswith("ref_") else r.arm
        L.append(f"| {name} | {r.auc_mean:.3f} ± {r.auc_sd:.3f} [{r.auc_fold_p2_5:.3f}–{r.auc_fold_p97_5:.3f}] | "
                 f"{fmt_ci(r.pooled_auc, r.pooled_ci_low, r.pooled_ci_high)} | {r.balacc_mean:.3f} ± {r.balacc_sd:.3f}{thr_note} | "
                 f"{r.sens_mean:.2f} / {r.spec_mean:.2f} | {r.brier:.3f} | {r.mean_pred_minus_prev:+.3f} | {r.calib_slope:.2f} |")
    if rep.get("confirmatory"):
        L += ["", "## A. Önceden belirlenmiş onaylayıcı analiz (RQ1; D-034 madde 6) — backbone füzyonu vs iki MFCC tabanı", "",
              f"NB düzeltilmiş t ({rep['n_repeats'] * 5} bağımsız olmayan fold skoru; yukarıdaki not). Yönlü iddia (backbone > MFCC) için "
              f"**tek yönlü** p; kesişim-birleşim p = max(p_LR, p_MLP); Holm (6); Holm-düzeltilmiş p **{ALPHA_ONE_SIDED}** ile karşılaştırılır. "
              f"Bu eşik sonuçlar görülmeden belirlenmiş, muhafazakâr bir karar eşiğidir: altı backbone'dan en az birini yanlışlıkla "
              f"\"MFCC'den iyi\" ilan etme olasılığını (aile bazında) en çok {ALPHA_ONE_SIDED}'te tutar; tek bir test için iki yönlü 0.05'in "
              "pozitif kuyruğuna karşılık gelir ve \"daha iyi\" kararları için iki yönlü Holm 0.05'ten hiçbir zaman gevşek değildir "
              "(docs/PHASE2_DESIGN.md 5.3a). \"Destekleniyor\" ayrıca iki bootstrap CI'ının da 0'ı dışlamasını gerektirir. "
              "Parantez içinde iki yönlü p (tek yönlü test \"daha kötü\"yü kanıtlayamaz; negatif farklar betimsel olarak korunur). "
              "Destek: havuzlanmış ΔAUC katılımcı bootstrap CI ve tekrar başına DeLong (iki yönlü, medyan p).", "",
              "| backbone | ΔAUC vs mfcc_lr: fold ort. · havuz [%95 CI] · tek yönlü p (iki yönlü) | ΔAUC vs mfcc_mlp: aynı | kesişim-birleşim p | Holm p | DeLong medyan p (LR / MLP) | sonuç |",
              "|---|---|---|---|---|---|---|"]
        for c in rep["confirmatory"]:
            a, b = c["vs_mfcc_lr"], c["vs_mfcc_mlp"]
            L.append(f"| {c['backbone']} | {a['fold_delta_mean']:+.3f} · {a['pooled_delta']:+.3f} [{a['boot_ci_low']:+.3f}, {a['boot_ci_high']:+.3f}] · {a['nb_p_one_sided']:.3f} ({a['nb_p']:.3f}) | "
                     f"{b['fold_delta_mean']:+.3f} · {b['pooled_delta']:+.3f} [{b['boot_ci_low']:+.3f}, {b['boot_ci_high']:+.3f}] · {b['nb_p_one_sided']:.3f} ({b['nb_p']:.3f}) | "
                     f"{c['iut_p']:.3f} | {c['holm_p']:.3f} | {a['delong_p_median']:.3f} / {b['delong_p_median']:.3f} | {c['verdict']} |")
    if rep.get("exploratory_pairs"):
        head = {"EXP-016S": "## Önceden belirlenmiş duyarlılık: kısa kayıtlarda diğer dolgu politikası − birincil (füzyon, tekrar 0)",
                "EXP-018": "## Önceden listelenmiş keşifsel karşılaştırma: tüm kayıt − 4 s pencere (füzyon)"}.get(
            exp, "## B. Önceden listelenmiş keşifsel karşılaştırmalar (füzyon; Holm aile içinde; yorum keşifsel)")
        L += ["", head, ""]
        if exp == "EXP-016":
            L += ["Yorum: farklı **hazır ses temsillerinin** karşılaştırması. Modeller ön-eğitim hedefi, etiket kullanımı (BEATs iter3+ "
                  "AudioSet etiketlerini tokenizer öğretmeni üzerinden dolaylı kullanır), veri alanı, mimari, boyut ve bant genişliğinde aynı "
                  "anda farklıdır; ön-eğitim yöntemi hakkında çıkarım yapılmaz (docs/PHASE2_DESIGN.md 9.6).", ""]
        L += ["| a − b | fold ort. Δ ± SD | havuzlanmış Δ [%95 CI] | NB p | Holm p |", "|---|---|---|---|---|"]
        for c in rep["exploratory_pairs"]:
            L.append(f"| {c['a']} − {c['b']} | {c['fold_delta_mean']:+.3f} ± {c['fold_delta_sd']:.3f} | "
                     f"{c['pooled_delta']:+.3f} [{c['boot_ci_low']:+.3f}, {c['boot_ci_high']:+.3f}] | {c['nb_p']:.3f} | {c.get('holm_p', np.nan):.3f} |")
    if rep.get("per_task_vs_mfcc_lr"):
        L += ["", f"## Görev başına Δ (backbone − mfcc_lr; keşifsel, RQ4; Holm {len(rep['per_task_vs_mfcc_lr'])} karşılaştırma içinde)", "",
              "| backbone | görev | fold ort. Δ ± SD | havuzlanmış Δ [%95 CI] | NB p (iki yönlü) | Holm p |", "|---|---|---|---|---|---|"]
        for c in rep["per_task_vs_mfcc_lr"]:
            L.append(f"| {c['a']} | {c['task']} | {c['fold_delta_mean']:+.3f} ± {c['fold_delta_sd']:.3f} | "
                     f"{c['pooled_delta']:+.3f} [{c['boot_ci_low']:+.3f}, {c['boot_ci_high']:+.3f}] | {c['nb_p']:.3f} | {c['holm_p']:.3f} |")
    if rep.get("vowel_vs_words"):
        L += ["", "## Ünlü (aaa) − kelimelerin ortalaması (fold düzeyinde AUC farkı; keşifsel, RQ5; Holm kollar içinde)", "",
              "| kol | ort. Δ ± SD | NB p | Holm p |", "|---|---|---|---|"]
        for v in rep["vowel_vs_words"]:
            L.append(f"| {v['arm']} | {v['aaa_minus_words_mean']:+.3f} ± {v['sd']:.3f} | {v['nb_p']:.3f} | {v['holm_p']:.3f} |")
    tasks = [t for t in TASKS.values() if t in set(summ.task)]
    if tasks:
        L += ["", "## Görev başına AUC (fold ortalaması ± SD; keşifsel, RQ4/RQ5)", "", "| kol | " + " | ".join(tasks) + " |",
              "|---|" + "---|" * len(tasks)]
        for arm in [a for a in summ.arm.unique() if not a.startswith("ref_")]:
            vals = []
            for t in tasks:
                q = summ[(summ.arm == arm) & (summ.task == t)]
                vals.append(f"{q.auc_mean.iloc[0]:.3f} ± {q.auc_sd.iloc[0]:.3f}" if len(q) else "—")
            L.append(f"| {arm} | " + " | ".join(vals) + " |")
    if rep.get("d035_per_fold") is not None:
        sel = pd.DataFrame(rep["d035_per_fold"])
        L += ["", "## C. D-035 — Faz 3 için fold başına seçim (yalnız o fold'un eğitim verisindeki iç doğrulama)", "",
              "Seçim her dış fold'da, o fold'un eğitim katılımcılarının iç 5-fold OOF füzyon AUC'siyle yapıldı; dış test "
              "tahminleri seçime girmedi. Fold'lar farklı backbone seçebilir; Faz 3 her fold'da o fold'un seçtiğini fine-tune eder. "
              "Tablo `d035_selection.csv` (Drive) ile aynıdır.", "",
              "| tekrar | fold | " + " | ".join(f"iç AUC {a}" for a in D035_CANDIDATES if f"inner_auc_{a}" in sel) + " | eşit (<0.01) | seçilen |",
              "|---|---|" + "---|" * sum(f"inner_auc_{a}" in sel for a in D035_CANDIDATES) + "---|---|"]
        for r_ in sel.itertuples(index=False):
            d = r_._asdict()
            L.append(f"| {d['repeat']} | {d['fold']} | " + " | ".join(f"{d[f'inner_auc_{a}']:.3f}" for a in D035_CANDIDATES
                                                                      if f"inner_auc_{a}" in d) + f" | {d['tied']} | **{d['selected']}** |")
    if rep.get("affected_shift"):
        L += ["", "### Kısa kaydı olan katılımcılarda füzyon olasılığının değişimi (|alternatif − birincil|, tekrar 0)", "",
              "Kısa kaydı olmayanlarda da küçük değişim beklenir: eğitim kümesindeki kısa kayıtlar değiştiği için model biraz değişir.", "",
              "| backbone | etkilenen katılımcı | ort. \\|Δp\\| | en büyük \\|Δp\\| | etkilenmeyenlerde ort. \\|Δp\\| |", "|---|---|---|---|---|"]
        for b, v in rep["affected_shift"].items():
            L.append(f"| {b} | {v['n_affected']} | {v['affected_mean_abs_dprob']:.4f} | {v['affected_max_abs_dprob']:.4f} | {v['unaffected_mean_abs_dprob']:.4f} |")
    if rep.get("procedure_estimates"):
        L += ["", "## Seçim prosedürünün dürüst tahmini (iç içe; dış test AUC'si yalnız değerlendirmede okunur)", "",
              "| kapsam | prosedür AUC ort ± SD | sonradan en iyi sabit kol (ort.) | kazananın laneti | seçim sıklıkları |", "|---|---|---|---|---|"]
        for k, v in rep["procedure_estimates"].items():
            L.append(f"| {k} | {v['procedure_auc_mean']:.3f} ± {v['procedure_auc_sd']:.3f} | {v['best_fixed_arm_post_hoc']} ({v['best_fixed_auc_mean']:.3f}) | "
                     f"{v['winners_curse']:+.3f} | {v['chosen_counts']} |")
    if rep.get("regression_check"):
        g = rep["regression_check"]
        L += ["", f"## Regresyon testi (D-034 madde 5)", "",
              f"mfcc_lr füzyonu {g['mfcc_lr_fusion_auc_mean']:.3f}; EXP-011 LR füzyonu {EXP011_LR_FUSION:.3f}; "
              f"fark {g['delta']:+.3f} → **{'GEÇTİ' if g['pass'] else 'KALDI — önce kod incelenir'}**"]
    edge = summ.dropna(subset=["C_at_grid_edge"]) if "C_at_grid_edge" in summ else pd.DataFrame()
    if len(edge):
        e = edge.groupby("arm").C_at_grid_edge.mean()
        L += ["", "## C ızgara ucu (seçilen C'nin 1e-5 ya da 1e2 olduğu fold oranı; yüksekse ızgara dar)", "",
              ", ".join(f"{a}: {v:.2f}" for a, v in e.items())]
    return "\n".join(L) + "\n"


def registry_rows(exp, summ, rep) -> list[dict]:
    rows = []
    for r in summ.itertuples():
        if exp != "EXP-016" and r.task != FUSION:
            continue
        if exp == "EXP-016S" and not r.arm.endswith("@alt"):
            continue
        is_ref = r.arm.startswith("ref_")
        rows.append({
            "experiment_id": exp, "date": time.strftime("%Y-%m-%d"), "status": "done",
            "research_question": "RQ1/RQ3" if exp == "EXP-016" else "RQ3 (exploratory)",
            "n_participants": r.n_participants, "n_patients": r.n_asthma, "n_controls": r.n_participants - r.n_asthma,
            "n_recordings": "", "n_segments": 0, "cohort": "audio cohort (342; görev 4: 341)", "recording_types": r.task,
            "split_strategy": "split files outer_r0..4 (D-029), participant-level", "split_file_sha256": rep["split_sha256"],
            "input_representation": "none" if is_ref else ("44-d MFCC" if r.arm.startswith("mfcc") else f"frozen embedding {r.arm}"),
            "sampling_rate": "", "preprocessing": "harmonized cache (D-030); 4 s/2 s windows end-aligned; short-recording policy: "
            + str(rep["arms"].get(r.arm, {}).get("short_policy", "n/a")),
            "model": r.arm, "pretrained_weights": "" if is_ref or r.arm.startswith("mfcc") else r.arm.split("@")[0],
            "frozen_layers": "all (frozen)", "classification_head": "LR (inner-CV C)" if r.arm != "mfcc_mlp" else "MLP (SMOTE)",
            "imbalance_strategy": "class_weight balanced" if r.arm != "mfcc_mlp" else "SMOTE (D-032)", "augmentation": "none",
            "hyperparameters": "C grid 1e-5..1e2, inner 5-fold neg_log_loss; threshold inner OOF (D-033)",
            "seeds": SEED, "n_folds_total": r.n_folds, "metric_level": "participant", "primary_metric": "roc_auc",
            "mean": round(r.auc_mean, 4), "sd": round(r.auc_sd, 4), "ci_low": round(r.pooled_ci_low, 4),
            "ci_high": round(r.pooled_ci_high, 4), "fold_p2_5": round(r.auc_fold_p2_5, 4), "fold_p97_5": round(r.auc_fold_p97_5, 4),
            "balanced_accuracy_mean": round(r.balacc_mean, 4), "balanced_accuracy_sd": round(r.balacc_sd, 4),
            "hardware": "Colab CPU (probe)", "git_commit": rep["git_commit"], "script": "scripts/run_probes.py",
            "result_location": f"reports/frozen/{exp}_frozen.json",
            "interpretation": "reference line (no audio)" if is_ref else "upper bound — recording context not evaluated (D-028)",
            "known_limitations": "frozen linear probe; context confounding deferred (D-028)"})
    return rows


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=["EXP-016", "EXP-016S", "EXP-017", "EXP-018"])
    ap.add_argument("--emb-root", required=True, type=Path)
    ap.add_argument("--backbones", default=",".join(BACKBONES))
    ap.add_argument("--mfcc-features", type=Path, help="features_cache.csv (EXP-016)")
    ap.add_argument("--context", required=True, type=Path)
    ap.add_argument("--splits-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--report-dir", required=True, type=Path)
    ap.add_argument("--registry", type=Path)
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()
    t0 = time.time()
    n_rep = 1 if args.exp in {"EXP-017", "EXP-016S"} else args.n_repeats
    splits = {r: pd.read_csv(args.splits_dir / f"outer_r{r}.csv") for r in range(n_rep)}
    cohort = set(splits[0].participant_id)
    assert all(set(S.participant_id) == cohort for S in splits.values()), "split dosyalarının kohortları farklı"
    C = load_context(args.context, cohort)
    ratio = float(np.mean([(S[(S.outer_fold == k) & (S.role == "test")].shape[0] / S[(S.outer_fold == k) & (S.role == "train")].shape[0])
                           for S in splits.values() for k in range(5)]))
    split_sha = ",".join(sha256_short(args.splits_dir / f"outer_r{r}.csv") for r in range(n_rep))
    bbs = [b for b in args.backbones.split(",") if b]
    assert set(bbs) <= set(BACKBONES)

    arms: list[Arm] = []
    if args.exp == "EXP-016":
        assert args.mfcc_features, "EXP-016 MFCC kollarını gerektirir (--mfcc-features)"
        arms = [emb_arm(args.emb_root, b, "win4", None) for b in bbs] + mfcc_arms(args.mfcc_features)
        role = "onaylayıcı (füzyon, 6 backbone vs 2 MFCC tabanı) + keşifsel (görev başına, çiftler)"
    elif args.exp == "EXP-016S":
        for b in bbs:
            arms += [emb_arm(args.emb_root, b, "win4", None), emb_arm(args.emb_root, b, "win4", None, short_alt=True)]
        role = "önceden belirlenmiş duyarlılık analizi (kısa kayıt dolgu politikası; yalnız tekrar 0; seçimde kullanılmaz)"
    elif args.exp == "EXP-017":
        for b in [b for b in bbs if b in SSL]:
            n_store = np.load(args.emb_root / b / "rec_win4.npy", mmap_mode="r").shape[1]
            arms += [emb_arm(args.emb_root, b, "win4", None)] + [emb_arm(args.emb_root, b, "win4", l) for l in range(n_store)]
        role = "keşifsel (katman katman, yalnız tekrar 0; hiçbir seçimde kullanılmaz)"
    else:
        for b in bbs:
            arms += [emb_arm(args.emb_root, b, "win4", None), emb_arm(args.emb_root, b, "full", None)]
        role = "keşifsel (tüm kayıt vs 4 s pencere; seçimde kullanılmaz)"
    for a in arms:
        n = {s: len(a.pids[s]) for s in TASKS}
        print(f"  kol {a.name}: kayıt/görev {n}; boyut {a.X[1].shape[1]}", flush=True)
    print(f"{args.exp}: {len(arms)} kol × {n_rep} tekrar × 5 fold × 7 görev", flush=True)

    # (9) bütün kollar her görevde aynı katılımcı kümesini kapsamalı (eşleştirilmiş karşılaştırma)
    for slot in TASKS:
        ref = set(arms[0].pids[slot])
        assert ref <= cohort, f"görev {slot}: split kohortunda olmayan katılımcı"
        for a in arms[1:]:
            assert set(a.pids[slot]) == ref, f"görev {slot}: {a.name} ile {arms[0].name} farklı katılımcı kümesi"
    data_sig = {"splits": {r: sha256_short(args.splits_dir / f"outer_r{r}.csv") for r in splits},
                "context": sha256_short(args.context)}
    test, inner = run_all(arms, C, splits, args.out_dir, args.n_jobs, data_sig)
    inner_raw = inner
    test, inner = add_fusion(test), add_fusion(inner)
    FT = fold_table(test, inner, C)
    summ = summarize(test, FT, C)
    audio_arms = [a.name for a in arms]
    check_same_units(test[test.task != FUSION], audio_arms)
    check_same_units(test[test.task == FUSION], audio_arms)
    git = rb.git_commit()
    rep = {"exp": args.exp, "role": role, "date": time.strftime("%Y-%m-%d"), "n_repeats": n_rep, "test_train_ratio": ratio,
           "split_files": [f"outer_r{r}.csv" for r in range(n_rep)], "split_sha256": split_sha, "git_commit": git,
           "arms": {a.name: a.signature for a in arms}, "c_grid": C_GRID}
    arm_names = [a.name for a in arms]

    if args.exp == "EXP-016":
        bb_present = [b for b in BACKBONES if b in arm_names]
        conf, p_iut = [], {}
        for b in bb_present:
            c_lr = compare(test, FT, C, b, "mfcc_lr", FUSION, ratio)
            c_mlp = compare(test, FT, C, b, "mfcc_mlp", FUSION, ratio)
            p_iut[b] = max(c_lr["nb_p_one_sided"], c_mlp["nb_p_one_sided"])   # yönlü iddia: H1 Δ > 0 (D-034 madde 6)
            conf.append({"backbone": b, "vs_mfcc_lr": c_lr, "vs_mfcc_mlp": c_mlp, "iut_p": p_iut[b]})
        hp = holm(p_iut) if p_iut else {}
        for c in conf:
            c["holm_p"] = hp[c["backbone"]]
            ok = c["holm_p"] < ALPHA_ONE_SIDED and c["vs_mfcc_lr"]["boot_ci_low"] > 0 and c["vs_mfcc_mlp"]["boot_ci_low"] > 0
            neg = c["vs_mfcc_lr"]["pooled_delta"] < 0 and c["vs_mfcc_mlp"]["pooled_delta"] < 0
            c["verdict"] = ("destekleniyor (üst sınır, geçici)" if ok else
                            ("desteklenmiyor; iki tabandan da düşük (nokta tahmini)" if neg else "desteklenmiyor"))
        rep["confirmatory"] = conf
        order = [(b, a) if (a, b) == ("wavlm_base_plus", "wavlm_large") else (a, b) for a, b in combinations(bb_present, 2)]
        pairs = [compare(test, FT, C, a, b, FUSION, ratio) for a, b in order]   # Large − Base+ (önceden yazılmış yön)
        hp2 = holm({i: c["nb_p"] for i, c in enumerate(pairs)}) if pairs else {}
        for i, c in enumerate(pairs):
            c["holm_p"] = hp2[i]
        rep["exploratory_pairs"] = pairs
        pt = [compare(test, FT, C, b, "mfcc_lr", t, ratio) for b in bb_present for t in TASKS.values()]
        hp3 = holm({i: c["nb_p"] for i, c in enumerate(pt)}) if pt else {}
        for i, c in enumerate(pt):
            c["holm_p"] = hp3[i]
        rep["per_task_vs_mfcc_lr"] = pt
        vowel = []
        for a in bb_present + ["mfcc_lr", "mfcc_mlp"]:
            ft = FT[FT.arm == a].pivot_table(index=["repeat", "fold"], columns="task", values="auc")
            d = (ft["aaa"] - ft[[t for t in TASKS.values() if t != "aaa"]].mean(1)).to_numpy()
            t_, p_ = nadeau_bengio(d, ratio)
            vowel.append({"arm": a, "aaa_minus_words_mean": float(d.mean()), "sd": float(d.std(ddof=1)), "nb_p": p_})
        hp4 = holm({i: v["nb_p"] for i, v in enumerate(vowel)})
        for i, v in enumerate(vowel):
            v["holm_p"] = hp4[i]
        rep["vowel_vs_words"] = vowel
        params = {a.name: (getattr(a, "n_params", None) or PARAMS_APPROX.get(a.name)) for a in arms if a.name in BACKBONES}
        cands = [b for b in D035_CANDIDATES if b in bb_present]
        rep["procedure_estimates"] = {}
        if cands:
            sel = select_per_fold(inner_raw, C, splits, cands, params)          # yalnız iç OOF (dış test yok)
            sel.to_csv(args.out_dir / "d035_selection.csv", index=False)
            rep["d035_per_fold"] = sel.to_dict(orient="records")
            rep["procedure_estimates"]["D-035 (BEATs / WavLM adayları)"] = procedure_estimate(sel, FT, cands)
        sel6 = select_per_fold(inner_raw, C, splits, bb_present, params)
        rep["procedure_estimates"]["6 backbone (keşifsel)"] = procedure_estimate(sel6, FT, bb_present)
        m = summ[(summ.arm == "mfcc_lr") & (summ.task == FUSION)].auc_mean.iloc[0]
        rep["regression_check"] = {"mfcc_lr_fusion_auc_mean": float(m), "delta": float(m - EXP011_LR_FUSION),
                                   "pass": bool(abs(m - EXP011_LR_FUSION) <= 0.01)}
    elif args.exp == "EXP-017":
        curves = {}
        for b in [b for b in BACKBONES if b in SSL and b in arm_names]:
            rows = []
            for a in [x for x in arm_names if x == b or x.startswith(f"{b}@L")]:
                q = summ[(summ.arm == a) & (summ.task == FUSION)].iloc[0]
                per = summ[(summ.arm == a) & (summ.task.isin(TASKS.values()))].auc_mean.mean()
                rows.append({"arm": a, "fusion_auc_mean": q.auc_mean, "fusion_auc_sd": q.auc_sd, "pooled_auc": q.pooled_auc,
                             "ci": [q.pooled_ci_low, q.pooled_ci_high], "mean_task_auc": per})
            curves[b] = rows
        rep["layer_curves"] = curves
    elif args.exp == "EXP-016S":
        rep["exploratory_pairs"] = [compare(test, FT, C, f"{b}@alt", b, FUSION, ratio) for b in BACKBONES
                                    if b in arm_names and f"{b}@alt" in arm_names]
        hp2 = holm({i: c["nb_p"] for i, c in enumerate(rep["exploratory_pairs"])}) if rep["exploratory_pairs"] else {}
        for i, c in enumerate(rep["exploratory_pairs"]):
            c["holm_p"] = hp2[i]
        # Etkilenen katılımcılar düzeyinde: füzyon olasılığının politika değişince ne kadar değiştiği (AUC farkı seyrelir)
        shift = {}
        for b in [b for b in BACKBONES if b in arm_names and f"{b}@alt" in arm_names]:
            meta = pd.read_csv(args.emb_root / b / "recordings.csv")
            affected = set(meta.participant_id[meta.dur_s < 4.0])
            pa, pb = pooled(test, b, FUSION, C), pooled(test, f"{b}@alt", FUSION, C)
            dlt = (pb - pa.reindex(pb.index)).abs()
            aff = dlt[dlt.index.isin(affected)]
            shift[b] = {"n_affected": int(len(aff)), "affected_mean_abs_dprob": float(aff.mean()) if len(aff) else float("nan"),
                        "affected_max_abs_dprob": float(aff.max()) if len(aff) else float("nan"),
                        "unaffected_mean_abs_dprob": float(dlt[~dlt.index.isin(affected)].mean())}
        rep["affected_shift"] = shift
    else:
        rep["exploratory_pairs"] = [compare(test, FT, C, f"{b}@full", b, FUSION, ratio) for b in BACKBONES
                                    if b in arm_names and f"{b}@full" in arm_names]
        hp2 = holm({i: c["nb_p"] for i, c in enumerate(rep["exploratory_pairs"])})
        for i, c in enumerate(rep["exploratory_pairs"]):
            c["holm_p"] = hp2[i]
    rep["summary"] = summ.round(4).to_dict(orient="records")
    rep["runtime_s"] = round(time.time() - t0, 1)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / f"{args.exp}_frozen.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=float))
    md = report_md(args.exp, rep, summ)
    if args.exp == "EXP-017":
        md += "\n## Katman eğrileri (füzyon AUC, tekrar 0; keşifsel)\n\n| kol | füzyon AUC fold ort ± SD | havuzlanmış [%95 CI] | görev AUC ortalaması |\n|---|---|---|---|\n"
        for b, rows in rep["layer_curves"].items():
            for r in rows:
                md += f"| {r['arm']} | {r['fusion_auc_mean']:.3f} ± {r['fusion_auc_sd']:.3f} | {fmt_ci(r['pooled_auc'], *r['ci'])} | {r['mean_task_auc']:.3f} |\n"
    (args.report_dir / f"{args.exp}_frozen.md").write_text(md)
    print(md)
    ids = set(C.index.astype(str))
    txt = (args.report_dir / f"{args.exp}_frozen.md").read_text() + (args.report_dir / f"{args.exp}_frozen.json").read_text()
    assert not any(f'"{i}"' in txt or f" {i} " in txt for i in ids), "raporda katılımcı ID'si var"
    if args.registry:
        reg = pd.read_csv(args.registry)
        reg = reg[reg.experiment_id != args.exp]
        new = pd.DataFrame(registry_rows(args.exp, summ, rep))
        pd.concat([reg, new.reindex(columns=reg.columns)], ignore_index=True).to_csv(args.registry, index=False)
    print(f"bitti ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
