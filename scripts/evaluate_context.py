"""
EVAL-016 — Dondurulmuş aile için bağlam değerlendirmesi, Aşama 1 (CPU). Plan: docs/EVAL016_PLAN.md (D-025, D-037 KABUL).

BİLİMSEL AMAÇ
    Projenin birincil sorusu (D-023): ses, ölçülen kayıt bağlamı (saat, tarih) ve yaşın ötesinde astım bilgisi taşıyor mu?
    EXP-016'nın kaydedilmiş dış-test tahminleri üzerinde, önceden belirlenmiş testler (yeniden eğitim YOK):
      L2  T1 (birincil): y ~ cr(saat, 4) + cr(tarih, 4) + yaş + ŝ; ŝ için Firth cezalı olabilirlik oranı testi (S9); Holm (6)
          + SD başına OR (katılımcı bootstrap CI) + gün-kümeli SE + cr df 6 + tekrar başına T1 + ikinci düzey CV ΔAUC (betimsel)
      L4  E2h-KLR: zaman penceresi içinde 1 saatlik tabakalara göre Jeffreys-cezalı koşullu lojistik regresyon (S12)
      L1  E1 / E1h / E2 / E2h / E3 / E3h tabakalı AUC'leri + yalnız-bağlam referans çizgileri (betimsel)
      L3  C2 kısmi testi: etiket-içi tam permütasyon (S10); Spisak tam testi (mlconfound, 'linear'; C = bağlam eğilim skoru)
      L5  karşıt hücreler (etiket × AM/PM) ve hasta-içi PM katsayısı (tarih ayarlı)
      N2  hastalarda skor ↔ tarih
      EXP-020 G1 probu: gömmelerden hastalarda AM/PM (tarih ayarlı) ve erken/geç dönem
      N5  bağlam-vekili kontrolü (yalnız G1+ backbone'larda; S11: betimsel, C1'de veto değil)
      L7  codec zinciri: katılımcı içi skor farkı + kayıt düzeyi zincir probu
      L8  fold / tekrar oynaklığı
      Kalibrasyon tanılaması: gerçek bağlam + yapay (sinyalsiz) skorlarla T1-LRT'nin yanlış pozitif oranı
    C1 / C2 kararları önceden yazılmış kurallarla (docs/CONFOUND_CONTROL_DESIGN.md 6; EVAL016_PLAN 4), PROTOCOL ayarlarıyla.
    N3 Aşama 2'dedir. PROTOCOL["status"] "KABUL" değilse gerçek veri kipinde çalışmaz (--allow-unapproved yalnız testler için).

VERİ / BİRİM
    Birim katılımcı. ŝ = 5 tekrarın dış-test füzyon olasılığının (görev olasılıklarının ortalaması) logit'lerinin ortalaması.
    T1 kohortu = EXP-016 kohortu ∩ saat, tarih, yaş ve kayıt günü bilinenler (dışlananlar sayılır ve raporlanır).

SIZINTI / BÜTÜNLÜK KONTROLLERİ (assert)
    partial/<kol>_r<r>_test.csv imzaları (split ve bağlam sha256) şimdiki dosyalarla aynı; her tekrarda her katılımcı tek
    fold'da ve o fold'un split dosyasındaki test rolünde; bütün kollar aynı katılımcı kümesi; G1/L7'deki gömme dosyası
    EXP-016'nın kullandığıyla aynı (sha256). G1 probu yalnız hastalarda eğitilir; N5'te sağlıklılar eğitime hiç girmez.
    Git'e giden raporlar katılımcı ID'si içermez (D-014); katılımcı düzeyi tablo yalnız Drive'a yazılır.

ÇALIŞTIRMA (Colab CPU; notebooks/22_context_eval_frozen.ipynb)
    python scripts/evaluate_context.py --probe-dir .../experiments/frozen_probe --context .../participant_context.csv \\
        --splits-dir .../splits --emb-root .../embeddings_v1 --cache-index .../audio_cache_v1/index.csv \\
        --out-dir .../experiments/context_eval --report-dir reports/context
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
import patsy
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (GridSearchCV, GroupKFold, RepeatedStratifiedKFold, StratifiedGroupKFold,
                                     StratifiedKFold)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.discrete.conditional_models import ConditionalLogit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_recording_context as arc  # noqa: E402  (stratified_auc: CONFOUND 2.3 ile aynı)
import run_probes as RP  # noqa: E402  (aynı prob pipeline'ı, C ızgarası, imza fonksiyonları, Holm)

warnings.filterwarnings("ignore")

BACKBONES = RP.BACKBONES
MFCC = ["mfcc_lr", "mfcc_mlp"]
TASK_SLOT = {v: k for k, v in RP.TASKS.items()}
ALPHA = 0.05
# Uygulama kararları (docs/EVAL016_PLAN.md Bölüm 7 ve 9). Gerçek veride çalıştırma yalnız status == "KABUL" iken mümkün;
# testler ve simülasyon --allow-unapproved kullanır. Değerler Alper'in onayıyla güncellenir ve git'te izlenir.
PROTOCOL = {
    "version": "EVAL-016 uygulama v1.1 (15. tur)",
    "status": "ONAY BEKLİYOR",               # → "KABUL" (onaydan sonra)
    "t1_fit": "firth",                       # S9: Firth + cezalı LRT (D-025 metni: klasik ML LRT; ayrışmada çöküyor)
    "c2_partial": "within_label_permutation",  # S10: etiket-içi tam permütasyon (mlconfound kısmi yalnız bilgi)
    "c1_cond3": "cluster_p",                 # K2: "cluster_p" = gün-kümeli p < 0.05 ve OR > 1 (öneri) | "direction" (kabul edilen metin, boş koşul)
    "c1_n5_veto": False,                     # S11: N5 betimsel (öneri) | True (kabul edilen metin: N5 anlamlıysa C1 düşer)
}
CLINICAL_PERIOD_REF = 0.855      # klinik profilin erken/geç dönemi ayırma AUC'si (EXP-002; CONFOUND 6, EXP-020 b)
EPS = 1e-6


# ----------------------------------------------------------------------------- yardımcılar
def zstd(x) -> np.ndarray:
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x - x.mean()


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def context_matrix(D: pd.DataFrame, df: int = 4, hour: bool = True, date: bool = True, age: bool = True) -> np.ndarray:
    """SIM-001 ile aynı bağlam matrisi: cr spline (patsy, merkezlenmiş), z-standardize sütunlar + sabit."""
    terms = []
    if hour:
        terms.append(f"cr(start_hour, df={df}, constraints='center')")
    if date:
        terms.append(f"cr(days, df={df}, constraints='center')")
    if age:
        terms.append("age")
    M = np.asarray(patsy.dmatrix(" + ".join(terms) + " - 1", D, return_type="dataframe"), float)
    M = np.column_stack([zstd(c) for c in M.T])
    X = sm.add_constant(M, has_constant="add")
    assert np.linalg.matrix_rank(X) == X.shape[1], "bağlam matrisi tam ranklı değil"
    return X


def logit_fit(y, X, **kw):
    """Klasik ML lojistik (statsmodels) — yalnız ikincil rapor için; ayrışmada (separation) yakınsamayabilir."""
    try:
        r = sm.Logit(y, X).fit(disp=0, maxiter=500, **kw)
        ok = bool(r.mle_retvals.get("converged", True)) and bool(np.all(np.isfinite(r.params)))
        return r, ok
    except Exception:
        return None, False


def _pll(X, y, b):
    """Firth (Jeffreys) cezalı log-olabilirlik: l(b) + 0.5 log det I(b), I = X' W X (tam tasarım matrisiyle)."""
    eta = X @ b
    p = 1.0 / (1.0 + np.exp(-eta))
    W = p * (1 - p)
    I = X.T @ (X * W[:, None])
    sign, logdet = np.linalg.slogdet(I)
    ll = float(np.sum(y * eta - np.logaddexp(0, eta)))
    return ll + 0.5 * logdet, p, W, I


def firth_fit(X, y, fixed: int | None = None, maxiter: int = 200, tol: float = 1e-8, max_step: float = 5.0) -> dict:
    """Firth lojistik regresyon (Heinze & Schemper 2002; logistf ile aynı tanım). fixed=j → b_j = 0 kısıtlı profil fit
    (ceza yine TAM tasarımın bilgi matrisiyle) — cezalı olabilirlik oranı testinin (PLR) sıfır modeli.
    Ayrışmada (separation) bile sonlu tahmin verir. Doğrulama: tests/test_eval016.py E10 (logistf / brglm2 referansları)."""
    y = np.asarray(y, float)
    k = X.shape[1]
    free = np.ones(k, bool)
    if fixed is not None:
        free[fixed] = False
    b = np.zeros(k)
    cur, p, W, I = _pll(X, y, b)
    conv = False
    for it in range(maxiter):
        Iinv = np.linalg.pinv(I)
        h = np.einsum("ij,jk,ik->i", X, Iinv, X) * W
        U = X.T @ (y - p + h * (0.5 - p))
        step = np.zeros(k)
        step[free] = np.linalg.lstsq(I[np.ix_(free, free)], U[free], rcond=None)[0]
        mx = np.max(np.abs(step)) / max_step
        if mx > 1:
            step /= mx
        for _ in range(60):                                   # adım yarılama: cezalı olabilirlik azalmasın
            new, p2, W2, I2 = _pll(X, y, b + step)
            if new >= cur - 1e-12:
                break
            step *= 0.5
        b = b + step
        cur, p, W, I = new, p2, W2, I2
        if np.max(np.abs(step)) < tol:
            conv = True
            break
    Iinv = np.linalg.pinv(I)
    h = np.einsum("ij,jk,ik->i", X, Iinv, X) * W
    return {"coef": b, "pll": cur, "converged": conv, "cov": Iinv, "p": p, "W": W, "h": h}


def t1(y: np.ndarray, Xc: np.ndarray, s: np.ndarray, groups: np.ndarray | None = None, ml: bool = True) -> dict:
    """T1: bağlam modeline ŝ (z) eklemenin testi. BİRİNCİL: Firth cezalı olabilirlik oranı testi (PLR, 1 sd; EVAL016_PLAN
    S9). İkincil: Firth Wald, klasik ML LRT (yakınsarsa), gün-kümeli dayanıklı SE (Firth skor katkılarıyla sandviç)."""
    y = np.asarray(y, float)
    sz = zstd(s)
    X1 = np.column_stack([Xc, sz])
    j = X1.shape[1] - 1
    full, null = firth_fit(X1, y), firth_fit(X1, y, fixed=j)
    plr = max(0.0, 2 * (full["pll"] - null["pll"]))
    b, se = float(full["coef"][j]), float(np.sqrt(max(full["cov"][j, j], 0)))
    out = {"converged": bool(full["converged"] and null["converged"]), "lr": float(plr), "p_lrt": float(stats.chi2.sf(plr, 1)),
           "coef": b, "se": se, "or_sd": float(np.exp(b)), "p_wald": float(2 * stats.norm.sf(abs(b) / se)) if se > 0 else float("nan")}
    if ml:
        m0, ok0 = logit_fit(y, Xc)
        m1, ok1 = logit_fit(y, X1)
        if m0 is not None and m1 is not None:
            lr_ml = max(0.0, 2 * (m1.llf - m0.llf))
            out.update({"ml_converged": bool(ok0 and ok1), "ml_p_lrt": float(stats.chi2.sf(lr_ml, 1)), "ml_or_sd": float(np.exp(m1.params[-1]))})
        else:
            out.update({"ml_converged": False, "ml_p_lrt": float("nan"), "ml_or_sd": float("nan")})
    if groups is not None:
        u = X1 * (y - full["p"] + full["h"] * (0.5 - full["p"]))[:, None]      # gözlem başına Firth skor katkıları
        G = pd.DataFrame(u).groupby(np.asarray(groups)).sum().to_numpy()
        ng = len(G)
        V = full["cov"] @ (G.T @ G) @ full["cov"] * (ng / (ng - 1))
        sec = float(np.sqrt(max(V[j, j], 0)))
        out.update({"p_cluster": float(2 * stats.norm.sf(abs(b) / sec)) if sec > 0 else float("nan"), "se_cluster": sec,
                    "or_sd_cluster": float(np.exp(b)), "n_clusters": int(ng)})
    return out


def boot_or(y, Xc, s, B: int, seed: int) -> dict:
    """Katılımcı bootstrap: her örneklemde T1 tam modeli (Firth) yeniden fit edilir; ŝ tam örneklem SD'siyle ölçeklenir."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    sz, n, coefs, fails = zstd(s), len(y), [], 0
    X1 = np.column_stack([Xc, sz])
    for _ in range(B):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            fails += 1
            continue
        f = firth_fit(X1[i], y[i])
        if not f["converged"] or not np.isfinite(f["coef"][-1]):
            fails += 1
            continue
        coefs.append(f["coef"][-1])
    if len(coefs) < 0.9 * B:
        return {"lo": float("nan"), "hi": float("nan"), "n_ok": len(coefs), "n_fail": fails}
    lo, hi = np.exp(np.percentile(coefs, [2.5, 97.5]))
    return {"lo": float(lo), "hi": float(hi), "n_ok": len(coefs), "n_fail": fails}


def cv_delta_auc(y, Xc, s, seed: int, n_rep: int = 20) -> dict:
    """İkinci düzey CV ΔAUC (bağlam vs bağlam + ŝ) — yalnız betimsel (SIM-001: karar için güçsüz)."""
    sz = zstd(s)
    d = []
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep, random_state=seed).split(Xc, y):
        a = []
        for X in (Xc[:, 1:], np.column_stack([Xc[:, 1:], sz])):
            m = LogisticRegression(C=1e4, max_iter=5000).fit(X[tr], y[tr])
            a.append(roc_auc_score(y[te], m.decision_function(X[te])))
        d.append(a[1] - a[0])
    return {"mean": float(np.mean(d)), "sd": float(np.std(d, ddof=1))}


def boot_mean_ci(x, B: int, seed: int) -> tuple[float, float]:
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return tuple(float(v) for v in np.percentile(m, [2.5, 97.5]))


def ids_free(txt: str, ids) -> bool:
    import re
    found = {int(m) for m in re.findall(r"(?<![\d.])(\d{6})(?![\d.])", txt)} & set(int(i) for i in ids)
    return not found


# ----------------------------------------------------------------------------- girdi
def load_scores(probe_dir: Path, arms: list[str], splits: dict, splits_dir: Path, context: Path):
    """partial/<kol>_r<r>_test.csv → katılımcı ŝ (tekrar ortalaması), tekrar başına logit, kayıt düzeyi tablo, imzalar."""
    ctx_sha = RP.sha256_short(context)
    s_hat, per_rep, rec, sigs = {}, {}, [], {}
    cohort = set(splits[0].participant_id)
    for arm in arms:
        reps = []
        for r, S in splits.items():
            f, j = probe_dir / "partial" / f"{arm}_r{r}_test.csv", probe_dir / "partial" / f"{arm}_r{r}.json"
            assert f.exists() and j.exists(), f"EXP-016 ara sonucu yok: {f.name} (önce notebook 21 EXP-016)"
            sig = json.loads(j.read_text())
            assert sig.get("split_sha256") == RP.sha256_short(splits_dir / f"outer_r{r}.csv"), \
                f"{arm} r{r}: imza uyuşmuyor (split dosyası EXP-016'dakinden farklı)"
            assert sig.get("context_sha256") == ctx_sha, f"{arm} r{r}: imza uyuşmuyor (bağlam tablosu EXP-016'dakinden farklı)"
            sigs[(arm, r)] = sig
            df = pd.read_csv(f)
            assert set(df.arm) == {arm} and (df["repeat"] == r).all()
            assert not df.duplicated(["participant_id", "task"]).any(), f"{arm} r{r}: yinelenen (katılımcı, görev)"
            assert df.groupby("participant_id").fold.nunique().max() == 1, f"{arm} r{r}: katılımcı iki fold'da"
            assert set(df.participant_id) == cohort, f"{arm} r{r}: katılımcı kümesi split kohortundan farklı"
            test_pairs = set(map(tuple, S.loc[S.role == "test", ["participant_id", "outer_fold"]].to_numpy()))
            pairs = set(map(tuple, df[["participant_id", "fold"]].drop_duplicates().to_numpy()))
            assert pairs <= test_pairs, f"{arm} r{r}: tahmin, split dosyasında test rolünde olmayan bir fold'dan"
            assert df.prob.between(0, 1).all()
            fused = df.groupby("participant_id").prob.mean()
            reps.append(pd.Series(logit(fused.to_numpy()), index=fused.index, name=r))
            rec.append(df.assign(logit=logit(df.prob.to_numpy())))
        R = pd.concat(reps, axis=1)
        assert R.notna().all().all()
        per_rep[arm] = R
        s_hat[arm] = R.mean(1)
    S_hat = pd.DataFrame(s_hat)
    return S_hat, per_rep, pd.concat(rec, ignore_index=True), sigs


def load_context_table(path: Path, cohort: set) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    C = RP.load_context(path, cohort)            # add_cells: period (son sağlıklı kayıt tarihi), ampm (saat < 12)
    need = {"label", "start_hour", "days", "age", "recording_date", "in_time_overlap", "period", "ampm"}
    miss = need - set(C.columns)
    assert not miss, f"participant_context.csv eksik sütun: {sorted(miss)}"
    full = C.dropna(subset=["start_hour", "days", "age", "recording_date"]).copy()
    full["in_time_overlap"] = full["in_time_overlap"].astype(str).str.lower().isin(["true", "1", "1.0"])
    full["hb"] = np.floor(full.start_hour)
    full["_all"] = 0
    excl = C.drop(full.index)
    info = {"n_cohort": int(len(C)), "n_t1": int(len(full)), "n_t1_asthma": int(full.label.sum()),
            "n_t1_healthy": int((full.label == 0).sum()), "n_excluded": int(len(excl)),
            "excluded_by_label": {"astım": int(excl.label.sum()), "sağlıklı": int((excl.label == 0).sum())},
            "n_days": int(full.recording_date.nunique())}
    assert full.label.nunique() == 2
    return C, full, info


def participant_embeddings(emb_root: Path, bb: str, expect_sha: str | None):
    d = emb_root / bb
    E = np.load(d / "rec_win4.npy", mmap_mode="r")
    meta = pd.read_csv(d / "recordings.csv")
    info = json.loads((d / "info.json").read_text())
    assert E.ndim == 3 and E.shape[0] == len(meta) and (meta.row.to_numpy() == np.arange(len(meta))).all()
    if expect_sha is not None:
        assert RP.sha256_short(d / "rec_win4.npy") == expect_sha, f"{bb}: gömme dosyası EXP-016'daki ile aynı değil"
    layers = info.get("primary_layers", [0] if E.shape[1] == 1 else list(range(1, E.shape[1])))
    la = (len(layers), E.shape[2]) if len(layers) > 1 else None
    X = np.asarray(E[:, layers, :], dtype=np.float32).reshape(len(meta), -1)
    assert np.isfinite(X).all()
    return meta, X, la


def mean_by(pids: np.ndarray, X: np.ndarray, order: list) -> np.ndarray:
    codes = pd.Categorical(pids, categories=order).codes
    keep = codes >= 0                      # T1 kohortu dışındaki katılımcıların kayıtları atlanır
    out = np.zeros((len(order), X.shape[1]), np.float64)
    np.add.at(out, codes[keep], X[keep])
    cnt = np.bincount(codes[keep], minlength=len(order))
    assert (cnt > 0).all(), "gömmesi olmayan katılımcı"
    return (out / cnt[:, None]).astype(np.float32)


# ----------------------------------------------------------------------------- problar (EXP-020)
def oof_probe(X, y, la, n_rep: int, seed: int, n_jobs: int, extra_X=None):
    """Tekrarlı tabakalı 5-fold, C iç 5-fold neg_log_loss (EXP-016 ile aynı pipeline). Dönüş: OOF logit (tekrar ort.),
    her dış modelin extra_X tahmininin logit ortalaması (N5: sağlıklılar eğitime hiç girmez)."""
    oof = np.full((n_rep, len(y)), np.nan)
    extra = []
    for i, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep, random_state=seed).split(X, y)):
        gs = GridSearchCV(RP.lr_pipe(la), {"clf__C": RP.C_GRID}, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                          scoring="neg_log_loss", n_jobs=n_jobs).fit(X[tr], y[tr])
        oof[i // 5, te] = logit(gs.predict_proba(X[te])[:, 1])
        if extra_X is not None:
            extra.append(logit(gs.predict_proba(extra_X)[:, 1]))
    assert not np.isnan(oof).any(), "her tekrarda her katılımcı tam bir kez test edilmeli"
    return oof.mean(0), (np.mean(extra, 0) if extra else None)


def g1_probe(Ct: pd.DataFrame, Xp: np.ndarray, la, args) -> dict:
    """(a) hastalarda PM (tarih ayarlı, LR testi); (b) hastalarda geç dönem (betimsel AUC). Xp: Ct sırasıyla hizalı."""
    pat = (Ct.label == 1).to_numpy()
    Cp = Ct[pat]
    y_pm = (Cp.ampm == "PM").astype(int).to_numpy()
    assert 0 < y_pm.sum() < len(y_pm), "hastalarda AM/PM ikisi de olmalı"
    e, e_healthy = oof_probe(Xp[pat], y_pm, la, args.g1_repeats, args.seed, args.n_jobs, extra_X=Xp[~pat])
    Xd = context_matrix(Cp, args.spline_df, hour=False, date=True, age=False)
    res = t1(y_pm, Xd, e)
    res["auc_oof_pm"] = float(roc_auc_score(y_pm, e))
    res["cv_delta_auc"] = cv_delta_auc(y_pm, Xd, e, args.seed, n_rep=10)
    res.update({"n_patients": int(len(y_pm)), "n_pm": int(y_pm.sum())})
    out = {"a_pm": res, "_e_patients": e, "_e_healthy": e_healthy}
    per = Cp.period.isin(["early", "late"]).to_numpy()
    y_late = (Cp.period[per] == "late").astype(int).to_numpy()
    if 0 < y_late.sum() < len(y_late):
        e2, _ = oof_probe(Xp[pat][per], y_late, la, args.g1_repeats, args.seed, args.n_jobs)
        auc = float(roc_auc_score(y_late, e2))
        rng = np.random.default_rng(args.seed)
        bs = []
        for _ in range(args.n_boot):
            i = rng.integers(0, len(y_late), len(y_late))
            if y_late[i].min() != y_late[i].max():
                bs.append(roc_auc_score(y_late[i], e2[i]))
        out["b_period"] = {"auc": auc, "ci": [float(v) for v in np.percentile(bs, [2.5, 97.5])],
                           "n": int(len(y_late)), "n_late": int(y_late.sum()), "clinical_reference": CLINICAL_PERIOD_REF}
    return out


def chain_probe(meta: pd.DataFrame, X: np.ndarray, la, chain: pd.Series, args) -> dict | None:
    """L7b: kayıt düzeyinde zincir (ffmpeg vs apple) probu, katılımcı-gruplu CV (betimsel)."""
    key = list(zip(meta.participant_id, meta.slot))
    ch = chain.reindex(key)
    ok = ch.notna().to_numpy()
    y = (ch[ok] == "ffmpeg").astype(int).to_numpy()
    g = meta.participant_id.to_numpy()[ok]
    if y.sum() < 10 or (1 - y).sum() < 10:
        return None
    Xo = X[ok]
    oof = np.full(len(y), np.nan)
    for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed).split(Xo, y, g):
        gs = GridSearchCV(RP.lr_pipe(la), {"clf__C": RP.C_GRID}, cv=GroupKFold(5), scoring="neg_log_loss",
                          n_jobs=args.n_jobs).fit(Xo[tr], y[tr], groups=g[tr])
        oof[te] = gs.predict_proba(Xo[te])[:, 1]
    assert not np.isnan(oof).any()
    auc = float(roc_auc_score(y, oof))
    rng = np.random.default_rng(args.seed)
    ug = np.unique(g)
    idx_by = {k: np.where(g == k)[0] for k in ug}
    bs = []
    for _ in range(min(args.n_boot, 1000)):
        i = np.concatenate([idx_by[k] for k in rng.choice(ug, len(ug))])
        if y[i].min() != y[i].max():
            bs.append(roc_auc_score(y[i], oof[i]))
    return {"auc": auc, "ci": [float(v) for v in np.percentile(bs, [2.5, 97.5])], "n_recordings": int(len(y)),
            "n_ffmpeg": int(y.sum())}


# ----------------------------------------------------------------------------- lensler
def l1_designs(Ct: pd.DataFrame, s: pd.Series) -> dict:
    D = Ct.assign(score=s.reindex(Ct.index).to_numpy())
    e2 = D[D.in_time_overlap]
    designs = {"E1": (D, ["_all"]), "E1h": (D, ["hb"]), "E2": (e2, ["_all"]), "E2h": (e2, ["hb"]),
               "E3": (D, ["recording_date"]), "E3h": (D, ["recording_date", "hb"])}
    out = {}
    for name, (df, key) in designs.items():
        auc, pairs = arc.stratified_auc(df, key, "score")
        out[name] = {"auc": float(auc), "pairs": pairs}
    return out


def l1_references(Ct: pd.DataFrame, seed: int, n_rep: int = 20) -> dict:
    """CONFOUND 2.3 / analyze_recording_context F ile aynı yöntem: yalnız-bağlam OOF skorlarının tabakalı AUC'si."""
    feats = {"yaş": ["age"], "saat + tarih": ["start_hour", "hour2", "days"], "yaş + saat + tarih": ["age", "start_hour", "hour2", "days"]}
    D = Ct.copy()
    D["hour2"] = D.start_hour ** 2
    y = D.label.to_numpy()
    out = {}
    for fn, cols in feats.items():
        per = {}
        oof = np.full((n_rep, len(D)), np.nan)
        for i, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep, random_state=seed).split(D, y)):
            m = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced")).fit(D.iloc[tr][cols], y[tr])
            oof[i // 5, te] = m.predict_proba(D.iloc[te][cols])[:, 1]
        for k in range(n_rep):
            for name, v in l1_designs(Ct, pd.Series(oof[k], index=D.index)).items():
                per.setdefault(name, []).append(v["auc"])
        out[fn] = {name: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))} for name, v in per.items()}
    return out


def clogit_firth(y, X, groups) -> dict:
    """Jeffreys-cezalı (Firth tipi) koşullu lojistik regresyon (Heinze & Puhr 2010): l_c(b) + 0.5 log det I_c(b).
    l_c ve skor statsmodels ConditionalLogit'ten (tam koşullu olabilirlik); I_c skorun sayısal türevi. Tabaka içinde
    ayrışmada da sonlu tahmin verir (EVAL016_PLAN S12). Doğrulama: tests/test_eval016.py E13 (R survival::clogit ile)."""
    from scipy.optimize import minimize
    from statsmodels.tools.numdiff import approx_fprime
    m = ConditionalLogit(np.asarray(y), np.asarray(X, float), groups=np.asarray(groups))

    def info(b):
        Hs = approx_fprime(b, m.score, centered=True)
        return -(Hs + Hs.T) / 2

    def negpl(b):
        sign, logdet = np.linalg.slogdet(info(b))
        if sign <= 0 or not np.isfinite(logdet):
            return 1e12
        return -(m.loglike(b) + 0.5 * logdet)

    k = X.shape[1]
    r = minimize(negpl, np.zeros(k), method="BFGS", options={"gtol": 1e-7, "maxiter": 500})
    best, ok = r, bool(r.success)
    if not ok:                                              # yedek: BFGS sonucundan Nelder–Mead ile cilala
        r2 = minimize(negpl, r.x, method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10, "maxfev": 6000})
        best, ok = (r2 if r2.fun <= r.fun else r), bool(r2.success)
    b = best.x
    cov = np.linalg.pinv(info(b))
    return {"coef": b, "se": np.sqrt(np.clip(np.diag(cov), 0, None)), "converged": ok, "pl": float(-best.fun)}


def e2h_klr(Ct: pd.DataFrame, s: pd.Series) -> dict:
    E = Ct[Ct.in_time_overlap].copy()
    E["s"] = s.reindex(E.index).to_numpy()
    inf = E.groupby("hb").label.transform("nunique") == 2
    E = E[inf]
    n1, n0 = int(E.label.sum()), int((E.label == 0).sum())
    if n1 < 5 or n0 < 5:
        return {"n_asthma": n1, "n_healthy": n0, "or_sd": float("nan"), "note": "yetersiz tabaka"}
    X = np.column_stack([zstd(E.s), zstd(E.age), zstd(E.start_hour), zstd(E.days)])
    out = {"n_asthma": n1, "n_healthy": n0, "n_strata": int(E.hb.nunique()), "method": "Jeffreys-cezalı koşullu logit (S12)"}
    try:
        f = clogit_firth(E.label.to_numpy(), X, E.hb.to_numpy())
        b, se = float(f["coef"][0]), float(f["se"][0])
        out.update({"or_sd": float(np.exp(b)), "ci": [float(np.exp(b - 1.96 * se)), float(np.exp(b + 1.96 * se))],
                    "p": float(2 * stats.norm.sf(abs(b) / se)) if se > 0 else float("nan"), "converged": f["converged"]})
    except Exception as ex:
        out.update({"or_sd": float("nan"), "note": f"cezalı fit hatası: {type(ex).__name__}"})
    try:                                                   # klasik ML koşullu logit — yalnız karşılaştırma
        r = ConditionalLogit(E.label.to_numpy(), X, groups=E.hb.to_numpy()).fit(disp=0)
        out.update({"ml_or_sd": float(np.exp(r.params[0])), "ml_p": float(r.pvalues[0])})
    except Exception as ex:
        out.update({"ml_or_sd": float("nan"), "ml_note": f"ML fit hatası: {type(ex).__name__}"})
    return out


def propensity(Ct: pd.DataFrame, Xc: np.ndarray, seed: int) -> np.ndarray:
    """Çapraz-fit bağlam eğilim skoru logit p̂(Y | yaş, cr(saat), cr(tarih)) — Spisak testlerinin tek boyutlu C'si.
    Firth fit: tarih bağlamı etiketi neredeyse ayırdığı için ML logit'leri aşırı uç olurdu."""
    y = Ct.label.to_numpy()
    c = np.full(len(y), np.nan)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xc, y):
        f = firth_fit(Xc[tr], y[tr])
        c[te] = Xc[te] @ f["coef"]
    assert not np.isnan(c).any()
    return c


def spisak(y, s, c, n_perms: int, seed: int) -> dict:
    from mlconfound.stats import full_confound_test, partial_confound_test
    kw = dict(num_perms=n_perms, cat_y=True, cat_yhat=False, cat_c=False, cond_dist_method="linear",
              random_state=seed, progress=False)
    f = full_confound_test(y, s, c, **kw)
    p = partial_confound_test(y, s, c, **kw)
    return {"full_p": float(f.p), "full_p_ci": [float(v) for v in f.p_ci],
            "partial_p": float(p.p), "partial_p_ci": [float(v) for v in p.p_ci],
            "r2_y_c": float(p.r2_y_c), "r2_yhat_c": float(p.r2_yhat_c), "r2_y_yhat": float(p.r2_y_yhat)}


def partial_perm_test(y, s, Ct: pd.DataFrame, n_perm: int, seed: int, with_age: bool = False) -> dict:
    """L3-kısmi (C2'nin ölçütü; EVAL016_PLAN S10): H0: ŝ ⊥ C | Y. Y ikili olduğu için TAM koşullu permütasyon testi:
    C satırları (cr(saat, 4) + cr(tarih, 4) [+ yaş]) her etiket grubunun İÇİNDE karıştırılır → (Y, C) ve (Y, ŝ) ortak
    dağılımları korunur, yalnız etiket-içi ŝ–C bağı kırılır. İstatistik: ŝ ~ 1 + y + C regresyonunda C'nin kısmi R²'si.
    Model varsayımı yok (mlconfound'un 'linear' c|y modeli sentetik testte bağlamsız skoru yanlış işaretledi)."""
    y = np.asarray(y, float)
    sz = zstd(s)
    Cb = context_matrix(Ct, 4, hour=True, date=True, age=with_age)[:, 1:]
    X0 = np.column_stack([np.ones(len(y)), y])
    r0 = sz - X0 @ np.linalg.lstsq(X0, sz, rcond=None)[0]
    ss0 = float(r0 @ r0)

    def stat(Cm):
        X = np.column_stack([X0, Cm])
        r1 = sz - X @ np.linalg.lstsq(X, sz, rcond=None)[0]
        return 1 - float(r1 @ r1) / ss0

    T = stat(Cb)
    rng = np.random.default_rng(seed)
    groups = [np.where(y == g)[0] for g in (0, 1)]
    cnt = 0
    for _ in range(n_perm):
        perm = np.arange(len(y))
        for g in groups:
            perm[g] = rng.permutation(g)
        cnt += stat(Cb[perm]) >= T
    return {"partial_r2": float(T), "p": float((1 + cnt) / (1 + n_perm)), "with_age": with_age, "n_perm": n_perm}


def l5_cells(Ct: pd.DataFrame, s: pd.Series, args) -> dict:
    D = Ct.assign(z=zstd(s.reindex(Ct.index)))
    cells = {}
    for (lab, ap), g in D.groupby(["label", "ampm"]):
        if ap == "unknown":
            continue
        lo, hi = boot_mean_ci(g.z, args.n_boot, args.seed)
        cells[f"{'astım' if lab == 1 else 'sağlıklı'}-{ap}"] = {"n": int(len(g)), "mean": float(g.z.mean()), "ci": [lo, hi]}
    P = D[(D.label == 1) & (D.ampm != "unknown")].copy()
    P["pm"] = (P.ampm == "PM").astype(float)
    Xd = context_matrix(P, args.spline_df, hour=False, date=True, age=False)
    r = sm.OLS(P.z.to_numpy(), np.column_stack([Xd, P.pm.to_numpy()])).fit(cov_type="HC3")
    b, (lo, hi) = float(r.params[-1]), r.conf_int()[-1]
    H = D[(D.label == 0) & (D.ampm != "unknown")]
    am, pm = H.z[H.ampm == "AM"].to_numpy(), H.z[H.ampm == "PM"].to_numpy()
    rng = np.random.default_rng(args.seed)
    diffs = [am[rng.integers(0, len(am), len(am))].mean() - pm[rng.integers(0, len(pm), len(pm))].mean()
             for _ in range(args.n_boot)] if len(am) > 1 and len(pm) > 1 else [np.nan]
    return {"cells": cells,
            "patients_pm_coef": {"coef": b, "ci": [float(lo), float(hi)], "p": float(r.pvalues[-1]), "n": int(len(P)),
                                 "n_pm": int(P.pm.sum())},
            "healthy_am_minus_pm": {"diff": float(am.mean() - pm.mean()) if len(am) and len(pm) else float("nan"),
                                    "ci": [float(v) for v in np.nanpercentile(diffs, [2.5, 97.5])],
                                    "n_am": int(len(am)), "n_pm": int(len(pm))}}


def n2_date(Ct: pd.DataFrame, s: pd.Series, args) -> dict:
    P = Ct[Ct.label == 1]
    z = zstd(s.reindex(P.index))
    Xh = context_matrix(P, args.spline_df, hour=True, date=False, age=False)
    Xf = np.column_stack([Xh, context_matrix(P, args.spline_df, hour=False, date=True, age=False)[:, 1:]])
    rf = sm.OLS(z, Xf).fit(cov_type="HC3")
    k = Xf.shape[1] - Xh.shape[1]          # tarih terimleri son k sütun: HC3-dayanıklı ortak Wald testi
    R = np.zeros((k, Xf.shape[1]))
    R[:, -k:] = np.eye(k)
    w = rf.wald_test(R, scalar=True)
    return {"n": int(len(P)), "p_date_terms": float(w.pvalue), "stat": float(w.statistic)}


def l7_paired(rec: pd.DataFrame, arm: str, chain: pd.Series, Ct: pd.DataFrame, args) -> dict | None:
    R = rec[rec.arm == arm].groupby(["participant_id", "task"]).logit.mean().reset_index()
    R["slot"] = R.task.map(TASK_SLOT)
    R["chain"] = chain.reindex(list(zip(R.participant_id, R.slot))).to_numpy()
    R = R[R.participant_id.isin(Ct.index) & R.chain.notna()]
    mix = R.groupby("participant_id").chain.nunique()
    M = R[R.participant_id.isin(mix.index[mix == 2])]
    if M.participant_id.nunique() < 5:
        return {"n_participants_mixed": int(M.participant_id.nunique()), "note": "karışık zincirli katılımcı < 5"}
    d = M.groupby(["participant_id", "chain"]).logit.mean().unstack()
    diff = (d["ffmpeg"] - d["apple"]).to_numpy()
    lo, hi = boot_mean_ci(diff, args.n_boot, args.seed)
    return {"n_participants_mixed": int(len(diff)), "mean_ffmpeg_minus_apple_logit": float(diff.mean()), "ci": [lo, hi],
            "note": "görevler zincirle karışık; betimsel"}


# ----------------------------------------------------------------------------- kalibrasyon tanılaması
def calibration_diagnostic(Ct: pd.DataFrame, Xc: np.ndarray, n_sim: int, seed: int) -> dict:
    """Gerçek etiket + gerçek bağlam, sinyalsiz yapay skor (SIM-001 senaryoları, gamma = 0): T1-LRT yanlış pozitif oranı."""
    y = Ct.label.to_numpy()
    q = np.column_stack([zstd(Ct.age), zstd(Ct.start_hour), zstd(Ct.start_hour ** 2), zstd(Ct.days)])
    z_smooth = zstd(sm.add_constant(q) @ firth_fit(sm.add_constant(q), y)["coef"])
    z_step = zstd((Ct.start_hour < 12).astype(float).to_numpy() + (Ct.period == "late").astype(float).to_numpy())
    day = Ct.recording_date.astype("category").cat.codes.to_numpy()
    rng = np.random.default_rng(seed)
    out = {}
    for scen in ["S1_smooth", "S2_step", "S3_day"]:
        p, pc = [], []
        for _ in range(n_sim):
            z = z_smooth if scen == "S1_smooth" else z_step if scen == "S2_step" else zstd(rng.normal(0, 1, day.max() + 1)[day])
            s = z + rng.normal(0, 1, len(y))
            r = t1(y, Xc, s, groups=day if scen == "S3_day" else None, ml=False)
            p.append(r["p_lrt"])
            if scen == "S3_day":
                pc.append(r.get("p_cluster", np.nan))
        p = np.array(p)
        out[scen] = {"reject_05": float(np.mean(p < 0.05)), "reject_01": float(np.mean(p < 0.01)), "n_sim": n_sim}
        if pc:
            pc = np.array(pc)
            out[scen].update({"reject_05_cluster": float(np.nanmean(pc < 0.05)), "reject_01_cluster": float(np.nanmean(pc < 0.01))})
    return out


# ----------------------------------------------------------------------------- karar kuralları
def verdicts(T, holm_p, boot, df6, e2h, g1, n5) -> dict:
    """C1 Aşama-1 koşulları (CONFOUND 6 / EVAL016_PLAN 4; D-025 madde 5), PROTOCOL ayarlarıyla. N3 Aşama 2'de."""
    out = {}
    for b in holm_p:
        t = T[b]
        g1pos = g1.get(b, {}).get("g1_positive", False)
        if PROTOCOL["c1_cond3"] == "cluster_p":
            c3 = bool(t.get("p_cluster", 1.0) < ALPHA and t["or_sd"] > 1)
        else:
            c3 = bool(t.get("or_sd_cluster", np.nan) > 1)
        c = {"1_t1_holm": bool(holm_p[b] < ALPHA and t["or_sd"] > 1),
             "2_boot_ci_excludes_1": bool(boot[b]["lo"] > 1),
             "3_cluster": c3,
             "4_e2h_direction": bool(e2h[b].get("or_sd", np.nan) > 1),
             "5_df4_df6_same_direction": bool(t["or_sd"] > 1 and df6[b]["or_sd"] > 1),
             "7_n5_not_significant": (bool(n5[b]["p_lrt"] >= ALPHA) if g1pos and b in n5 else None)}
        stage1 = [v for k, v in c.items() if v is not None and (k != "7_n5_not_significant" or PROTOCOL["c1_n5_veto"])]
        all_ok = all(stage1)
        out[b] = {"conditions": c, "g1_positive": g1pos, "stage1_all_met": bool(all_ok),
                  "c1_status": "Aşama 2 bekliyor (N3)" if all_ok else "sağlanmadı",
                  "stage2_triggered": c["1_t1_holm"]}     # kabul edilen kural: T1 Holm sonrası pozitif
    return out


# ----------------------------------------------------------------------------- rapor
def f3(x):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.3f}"


def fp(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def report_md(R: dict) -> str:
    ci, arms = R["cohort"], R["arms"]
    L = ["# EVAL-016 — dondurulmuş aile için bağlam değerlendirmesi, Aşama 1 (otomatik üretildi; katılımcı ID'si içermez)", "",
         "**Plan:** `docs/EVAL016_PLAN.md` (D-025, D-037 KABUL). **Bu rapor Aşama 1'dir:** N3 (yalnız arka plan) Aşama 2'dedir; "
         "C1 hiçbir backbone için bu raporla tamamlanmış sayılmaz.", "",
         f"**Kohort:** EXP-016 kohortu {ci['n_cohort']} → T1 kohortu {ci['n_t1']} ({ci['n_t1_asthma']} astım / {ci['n_t1_healthy']} sağlıklı); "
         f"saat / tarih / yaş eksik olduğu için dışlanan {ci['n_excluded']} {ci['excluded_by_label']}; kayıt günü {ci['n_days']}.", "",
         f"**ŝ:** {R['n_repeats']} tekrarın dış-test füzyon logit'lerinin ortalaması; z-standardize, OR = ŝ'nin 1 SD'si başına. "
         f"Bootstrap B = {R['n_boot']}; Spisak permütasyon = {R['n_perms']}; spline df = {R['spline_df']} (duyarlılık 6).", "",
         f"**Uygulama protokolü:** {R['protocol']['version']} — durum **{R['protocol']['status']}** · T1 fit: {R['protocol']['t1_fit']} · "
         f"C2 kısmi test: {R['protocol']['c2_partial']} · C1 koşul 3: {R['protocol']['c1_cond3']} · N5 C1'de: {R['protocol']['c1_n5_veto']}"
         + (" · **UYARI: onaylanmamış protokol (yalnız test / simülasyon)**" if R['protocol']['status'] != 'KABUL' else ""), ""]
    cal = R["calibration"]
    L += ["## 0. Kalibrasyon tanılaması (betimsel; gerçek bağlam, sinyalsiz yapay skor, γ = 0)", "",
          "T1-PLR'nin (Firth) bu kohortta yanlış pozitif oranı. İdeal: 0.05 / 0.01. S3 (ölçülmemiş gün etkisi) bilinen kalıntı risktir (SIM-001).", "",
          "| senaryo | p < 0.05 | p < 0.01 | gün-kümeli p < 0.05 | n_sim |", "|---|---|---|---|---|"]
    for k, v in cal.items():
        L.append(f"| {k} | {f3(v['reject_05'])} | {f3(v['reject_01'])} | {f3(v.get('reject_05_cluster'))} | {v['n_sim']} |")
    L += ["", "## 1. T1 (L2, birincil) — onaylayıcı aile: 6 backbone, Holm; MFCC yalnız referans", "",
          "Karar istatistiği: **Firth cezalı olabilirlik oranı testi** (PLR; EVAL016_PLAN S9 — tarih bağlamı etiketi neredeyse "
          "ayırdığı için klasik ML fit'i yakınsamayabilir). Klasik ML LRT yalnız karşılaştırma için.", "",
          "| kol | PLR χ² | p (PLR) | Holm p | OR / SD [bootstrap %95 CI] | Firth Wald p | gün-kümeli p | OR df 6 (p) | ML LRT p (yakınsadı) | tekrar başına PLR aralığı | CV ΔAUC (betimsel) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in arms:
        t, b, d6 = R["t1"][a], R["boot"][a], R["t1_df6"][a]
        hp = R["holm"].get(a)
        rr = R["t1_per_repeat"][a]
        L.append(f"| {a} | {f3(t['lr'])} | {fp(t['p_lrt'])} | {fp(hp) if hp is not None else 'referans'} | "
                 f"{f3(t['or_sd'])} [{f3(b['lo'])}, {f3(b['hi'])}] (başarısız {b['n_fail']}) | {fp(t['p_wald'])} | {fp(t.get('p_cluster'))} | "
                 f"{f3(d6['or_sd'])} ({fp(d6['p_lrt'])}) | {fp(t.get('ml_p_lrt'))} ({'evet' if t.get('ml_converged') else 'hayır'}) | "
                 f"{f3(min(rr))}–{f3(max(rr))} | {R['cv_delta_auc'][a]['mean']:+.4f} ± {R['cv_delta_auc'][a]['sd']:.4f} |")
    L += ["", "## 2. E2h-KLR (L4) ve tabakalı AUC'ler (L1, betimsel)", "",
          "| kol | E2h n (astım/sağlıklı) | OR / SD [%95 CI] | p |", "|---|---|---|---|"]
    for a in arms:
        e = R["e2h"][a]
        L.append(f"| {a} | {e['n_asthma']}/{e['n_healthy']} | {f3(e.get('or_sd'))} [{f3((e.get('ci') or [np.nan])[0])}, "
                 f"{f3((e.get('ci') or [np.nan, np.nan])[1])}] | {fp(e.get('p'))} |")
    des = list(next(iter(R["l1"].values())).keys())
    L += ["", "Tabakalı AUC (yalnız aynı tabakadaki astım–sağlıklı çiftleri). **0.5'e karşı test edilmez**; her tasarımın gerçek "
          "şans çizgisi aşağıdaki yalnız-bağlam referanslarıdır (CONFOUND 2.3).", "",
          "| kol | " + " | ".join(des) + " |", "|---|" + "---|" * len(des)]
    for a in arms:
        L.append(f"| {a} | " + " | ".join(f3(R['l1'][a][d]['auc']) for d in des) + " |")
    for fn, v in R["l1_references"].items():
        L.append(f"| *referans: yalnız {fn}* | " + " | ".join(f"{v[d]['mean']:.3f}" for d in des) + " |")
    L += ["", "## 3. Bağlam kullanımı: Spisak (L3), karşıt hücreler (L5), N2", ""]
    if R.get("spisak_note"):
        L.append(f"Spisak: {R['spisak_note']}")
    L += ["", "| kol | **L3 kısmi (C2): etiket-içi permütasyon p**, C = saat + tarih [+ yaş] | Spisak tam p (H0: ŝ ⊥ Y \\| C) | "
          "mlconfound kısmi p (yalnız bilgi) | hasta-içi PM katsayısı (z) [%95 CI] | sağlıklı AM − PM (z) [%95 CI] | N2 tarih p (hastalar) |",
          "|---|---|---|---|---|---|---|"]
    for a in arms:
        sp, l5, n2, l3 = R["spisak"].get(a, {}), R["l5"][a], R["n2"][a], R["l3_partial"][a]
        pc, hd = l5["patients_pm_coef"], l5["healthy_am_minus_pm"]
        L.append(f"| {a} | {fp(l3['context']['p'])} [{fp(l3['context_age']['p'])}] | {fp(sp.get('full_p'))} | {fp(sp.get('partial_p'))} | "
                 f"{pc['coef']:+.3f} [{pc['ci'][0]:+.3f}, {pc['ci'][1]:+.3f}] | "
                 f"{hd['diff']:+.3f} [{hd['ci'][0]:+.3f}, {hd['ci'][1]:+.3f}] ({hd['n_am']}/{hd['n_pm']}) | {fp(n2['p_date_terms'])} |")
    L += ["", "## 4. EXP-020 G1 bağlam probu (gömmeler) ve N5", "",
          "| backbone | (a) hastalarda PM: OOF AUC | LR p (tarih ayarlı) | Holm p | OR / SD | G1 | (b) geç dönem AUC [%95 CI] (klinik ref. 0.855) | N5 p (vekil skor T1'de) |",
          "|---|---|---|---|---|---|---|---|"]
    for b in R["g1"]:
        g = R["g1"][b]
        a_ = g["a_pm"]
        bp = g.get("b_period", {})
        n5 = R["n5"].get(b)
        L.append(f"| {b} | {f3(a_['auc_oof_pm'])} | {fp(a_['p_lrt'])} | {fp(g['holm_p'])} | {f3(a_['or_sd'])} | "
                 f"{'**+**' if g['g1_positive'] else '−'} | {f3(bp.get('auc'))} [{f3((bp.get('ci') or [np.nan])[0])}, "
                 f"{f3((bp.get('ci') or [np.nan, np.nan])[1])}] | {fp(n5['p_lrt']) if n5 else 'bilgisiz (G1−)'} |")
    L += ["", "## 5. Codec zinciri (L7, betimsel)", "", "| kol | karışık zincirli katılımcı | ffmpeg − apple logit [%95 CI] | zincir probu AUC [%95 CI] |",
          "|---|---|---|---|"]
    for a in arms:
        p7 = R["l7"].get(a) or {}
        cp = R["l7_probe"].get(a) or {}
        if "ci" in p7:
            pair = f"{p7['mean_ffmpeg_minus_apple_logit']:+.3f} [{p7['ci'][0]:+.3f}, {p7['ci'][1]:+.3f}]"
        else:
            pair = p7.get("note", "—")
        L.append(f"| {a} | {p7.get('n_participants_mixed', '—')} | {pair} | "
                 f"{f3(cp.get('auc'))} [{f3((cp.get('ci') or [np.nan])[0])}, {f3((cp.get('ci') or [np.nan, np.nan])[1])}] |")
    L += ["", "## 6. Oynaklık (L8)", "", "| kol | fold AUC ort ± SD (25) | tekrar AUC aralığı |", "|---|---|---|"]
    for a in arms:
        v = R["l8"][a]
        L.append(f"| {a} | {v['fold_auc_mean']:.3f} ± {v['fold_auc_sd']:.3f} | {v['repeat_auc_min']:.3f}–{v['repeat_auc_max']:.3f} |")
    L += ["", "## 7. Önceden yazılmış kurallara göre Aşama 1 kararları", "",
          "C1 koşulları: 1 Holm-T1 (PLR) p < 0.05 ve OR > 1 · 2 bootstrap CI 1'i dışlıyor · 3 gün-kümeli: "
          + ("p < 0.05 ve OR > 1 (K2)" if PROTOCOL["c1_cond3"] == "cluster_p" else "yön (boş koşul; K2)")
          + " · 4 E2h-KLR OR > 1 · 5 df 4 ve df 6 aynı yön · 6 N3 (Aşama 2) · 7 G1+ ise N5 anlamlı değil "
          + ("(C1'e dahil)" if PROTOCOL["c1_n5_veto"] else "(S11: yalnız betimsel, C1'e dahil değil)") + ".", "",
          "| backbone | 1 | 2 | 3 | 4 | 5 | 7 | Aşama-1 koşulları | C1 | Aşama 2 (kabul edilen kural: Holm-T1 pozitif) | C2 (bağlam kullanımı) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    yn = {True: "✓", False: "✗", None: "n/a"}
    for b in R["backbones"]:
        v = R["verdicts"][b]
        c = v["conditions"]
        L.append(f"| {b} | {yn[c['1_t1_holm']]} | {yn[c['2_boot_ci_excludes_1']]} | {yn[c['3_cluster']]} | "
                 f"{yn[c['4_e2h_direction']]} | {yn[c['5_df4_df6_same_direction']]} | {yn[c['7_n5_not_significant']]} | "
                 f"{'hepsi ✓' if v['stage1_all_met'] else 'eksik'} | {v['c1_status']} | {'evet' if v['stage2_triggered'] else 'hayır'} | "
                 f"{R['c2'][b]} |")
    any_c1 = any(v["stage1_all_met"] for v in R["verdicts"].values())
    L += ["", "## 8. Bu rapor neyi gösterebilir, neyi gösteremez (önceden yazılmış metin)", ""]
    if not any_c1:
        L += ["- **Hiçbir backbone C1'in Aşama-1 koşullarını sağlamadı.** Önceden yazılmış ifade: \"Bu veriyle bağlamdan bağımsız bir ses "
              "sinyali gösterilemedi.\" Bu, \"seste astım bilgisi yok\" demek değildir: SIM-001'e göre %80 güç ancak aynı-bağlam AUC'si "
              "≈ 0.69'da (α 0.05) / ≈ 0.74'te (Holm düzeyi) vardı; bu değerler iyimserdi."]
    else:
        L += ["- Aşama-1 koşullarını sağlayan backbone(lar) için C1 **henüz kurulmadı**: N3 (Aşama 2) bekleniyor."]
    L += ["- T1 yalnız **ölçülmüş** bağlamı (saat, tarih) ve yaşı ayarlar. Ölçülmemiş, güne özgü bağlamda yanlış pozitif ~%12 "
          "(p < 0.05) olabilir (SIM-001 S3; yukarıdaki kalibrasyon tablosu). Gün-kümeli SE bunu düzeltmez.",
          "- Pozitiflik sınırı: sabah kaydedilen sağlıklı sayısı çok küçük, geç dönemde sağlıklı yok; bilgi yalnız örtüşme bölgesinden gelir.",
          "- Spirometri / bronkodilatör sonrası kayıt olasılığı (S13) saat kontrolüyle ayrılamaz; veri ekibinden cevap bekleniyor.",
          "- Pozitif bir C1 bile klinik tanı aracı anlamına gelmez (vaka-kontrol, tek merkez, dış doğrulama yok).",
          "- Bu sayılar EXP-016'nın dondurulmuş probları içindir; fine-tune modellerine genellenemez (C3 ayrı değerlendirilir).", ""]
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------- ana akış
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", required=True, type=Path)
    ap.add_argument("--context", required=True, type=Path)
    ap.add_argument("--splits-dir", required=True, type=Path)
    ap.add_argument("--emb-root", required=True, type=Path)
    ap.add_argument("--cache-index", required=True, type=Path, help="audio_cache_v1/index.csv (codec zinciri; L7)")
    ap.add_argument("--out-dir", required=True, type=Path, help="Drive: katılımcı düzeyi tablolar")
    ap.add_argument("--report-dir", required=True, type=Path, help="git: yalnız agrega")
    ap.add_argument("--backbones", default=",".join(BACKBONES))
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perms", type=int, default=1000)
    ap.add_argument("--n-sim", type=int, default=500)
    ap.add_argument("--g1-repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--skip-spisak", action="store_true", help="yalnız mlconfound kurulamazsa; raporda açıkça yazılır")
    ap.add_argument("--allow-unapproved", action="store_true", help="yalnız sentetik testler / simülasyon")
    args = ap.parse_args()
    if PROTOCOL["status"] != "KABUL" and not args.allow_unapproved:
        raise SystemExit(f"EVAL-016 uygulama protokolü onay bekliyor ({PROTOCOL['version']}; docs/EVAL016_PLAN.md Bölüm 9). "
                         "Gerçek veride çalıştırılmadı.")
    t0 = time.time()
    args.spline_df = 4                     # birincil spline df (D-025); df 6 duyarlılık olarak ayrıca hesaplanır

    splits = {r: pd.read_csv(args.splits_dir / f"outer_r{r}.csv") for r in range(args.n_repeats)}
    cohort = set(splits[0].participant_id)
    assert all(set(S.participant_id) == cohort for S in splits.values())
    bbs = [b for b in args.backbones.split(",") if b]
    assert set(bbs) <= set(BACKBONES)
    arms = bbs + MFCC
    S_hat, per_rep, rec, sigs = load_scores(args.probe_dir, arms, splits, args.splits_dir, args.context)
    C, Ct, cinfo = load_context_table(args.context, cohort)
    print(f"kohort: {cinfo}", flush=True)
    y = Ct.label.to_numpy()
    day = Ct.recording_date.astype("category").cat.codes.to_numpy()
    Xc4 = context_matrix(Ct, 4)
    Xc6 = context_matrix(Ct, 6)
    S = S_hat.reindex(Ct.index)
    assert S.notna().all().all() and len(S) == len(Ct)

    R = {"exp": "EVAL-016", "stage": 1, "date": time.strftime("%Y-%m-%d"), "cohort": cinfo, "arms": arms, "backbones": bbs,
         "n_repeats": args.n_repeats, "n_boot": args.n_boot, "n_perms": args.n_perms, "spline_df": args.spline_df,
         "plan": "docs/EVAL016_PLAN.md", "protocol": dict(PROTOCOL)}
    print("kalibrasyon tanılaması …", flush=True)
    R["calibration"] = calibration_diagnostic(Ct, Xc4, args.n_sim, args.seed)

    print("T1 …", flush=True)
    T, boot, df6, cvd, prep = {}, {}, {}, {}, {}
    for a in arms:
        s = S[a].to_numpy()
        T[a] = t1(y, Xc4, s, groups=day)
        df6[a] = t1(y, Xc6, s, ml=False)
        boot[a] = boot_or(y, Xc4, s, args.n_boot, args.seed)
        cvd[a] = cv_delta_auc(y, Xc4, s, args.seed)
        prep[a] = [t1(y, Xc4, per_rep[a][r].reindex(Ct.index).to_numpy(), ml=False)["lr"] for r in per_rep[a].columns]
        print(f"  {a}: LR {T[a]['lr']:.2f} p {T[a]['p_lrt']:.4f} OR {T[a]['or_sd']:.2f} "
              f"[{boot[a]['lo']:.2f}, {boot[a]['hi']:.2f}]", flush=True)
    holm_p = RP.holm({b: T[b]["p_lrt"] for b in bbs})
    R.update({"t1": T, "t1_df6": df6, "boot": boot, "cv_delta_auc": cvd, "t1_per_repeat": prep, "holm": holm_p})

    print("E2h-KLR, L1 …", flush=True)
    R["e2h"] = {a: e2h_klr(Ct, S[a]) for a in arms}
    R["l1"] = {a: l1_designs(Ct, S[a]) for a in arms}
    R["l1_references"] = l1_references(Ct, args.seed)

    print("L3 (kısmi permütasyon), Spisak, L5, N2 …", flush=True)
    R["l3_partial"] = {a: {"context": partial_perm_test(y, S[a].to_numpy(), Ct, args.n_perms, args.seed),
                           "context_age": partial_perm_test(y, S[a].to_numpy(), Ct, args.n_perms, args.seed, with_age=True)}
                       for a in arms}
    c = propensity(Ct, Xc4, args.seed)
    R["spisak"] = {}
    if args.skip_spisak:
        R["spisak_note"] = "ÇALIŞTIRILMADI (--skip-spisak; mlconfound kurulamadı). C2 yalnız L5'e dayanıyor."
    else:
        for a in arms:
            R["spisak"][a] = spisak(y, S[a].to_numpy(), c, args.n_perms, args.seed)
            print(f"  {a}: tam p {R['spisak'][a]['full_p']:.3f} kısmi p {R['spisak'][a]['partial_p']:.3f}", flush=True)
        R["spisak_note"] = ("mlconfound 0.21.3, cond_dist_method='linear' (gam güncel scipy ile çalışmıyor; EVAL016_PLAN S3). "
                            "Tam test T1'in destekleyicisidir. mlconfound KISMİ testi yalnız bilgi amaçlı: sentetik testte bağlamsız "
                            "güçlü bir skoru yanlışlıkla işaretledi; C2 kararı etiket-içi permütasyon testine dayanır (S10).")
    R["l5"] = {a: l5_cells(Ct, S[a], args) for a in arms}
    R["n2"] = {a: n2_date(Ct, S[a], args) for a in arms}

    print("EXP-020 G1, N5, L7 …", flush=True)
    chain_idx = pd.read_csv(args.cache_index)
    assert {"participant_id", "slot", "chain"} <= set(chain_idx.columns), "önbellek index.csv'de 'chain' sütunu yok"
    chain = chain_idx.set_index(["participant_id", "slot"]).chain
    R["g1"], R["n5"], R["l7"], R["l7_probe"], g1p, proxies = {}, {}, {}, {}, {}, {}
    for b in bbs:
        meta, X, la = participant_embeddings(args.emb_root, b, sigs[(b, 0)].get("emb_sha256"))
        Xp = mean_by(meta.participant_id.to_numpy(), X, list(Ct.index))
        g = g1_probe(Ct, Xp, la, args)
        R["g1"][b] = {k: v for k, v in g.items() if not k.startswith("_")}
        g1p[b] = g["a_pm"]["p_lrt"]
        proxies[b] = (g["_e_patients"], g["_e_healthy"])
        R["l7_probe"][b] = chain_probe(meta, X, la, chain, args)
        print(f"  {b}: G1 PM AUC {g['a_pm']['auc_oof_pm']:.3f} p {g['a_pm']['p_lrt']:.4f}", flush=True)
        del X, Xp
    holm_g1 = RP.holm(g1p)
    for b in bbs:
        g = R["g1"][b]
        g["holm_p"] = float(holm_g1[b])
        g["g1_positive"] = bool(holm_g1[b] < ALPHA and g["a_pm"]["or_sd"] > 1)
        e_pat, e_hlt = proxies[b]
        if g["g1_positive"]:
            proxy = pd.Series(np.nan, index=Ct.index)
            proxy[Ct.label == 1] = e_pat
            proxy[Ct.label == 0] = e_hlt
            assert proxy.notna().all()
            R["n5"][b] = t1(y, Xc4, proxy.to_numpy(), ml=False)
    for a in arms:
        R["l7"][a] = l7_paired(rec, a, chain, Ct, args)

    # L8: EXP-016 ile aynı fold / tekrar AUC'leri (bütün kohort)
    R["l8"] = {}
    for a in arms:
        rr = rec[rec.arm == a]
        fa, ra = [], []
        for (r, k), g in rr.groupby(["repeat", "fold"]):
            fu = g.groupby("participant_id").prob.mean()
            fa.append(roc_auc_score(C.loc[fu.index, "label"], fu))
        for r, g in rr.groupby("repeat"):
            fu = g.groupby("participant_id").prob.mean()
            ra.append(roc_auc_score(C.loc[fu.index, "label"], fu))
        R["l8"][a] = {"fold_auc_mean": float(np.mean(fa)), "fold_auc_sd": float(np.std(fa, ddof=1)),
                      "repeat_auc_min": float(min(ra)), "repeat_auc_max": float(max(ra))}

    R["verdicts"] = verdicts(T, holm_p, boot, df6, R["e2h"], R["g1"], R["n5"])
    R["c2"] = {}
    for b in bbs:
        pc = R["l5"][b]["patients_pm_coef"]["ci"]
        if PROTOCOL["c2_partial"] == "within_label_permutation":
            pp = R["l3_partial"][b]["context"]["p"]
        else:
            pp = R["spisak"].get(b, {}).get("partial_p", 1.0)
        pos = (pp < ALPHA) or (pc[0] > 0 or pc[1] < 0)
        R["c2"][b] = "pozitif" if pos else "gösterilemedi"
    R["elapsed_s"] = round(time.time() - t0, 1)

    # çıktılar
    args.out_dir.mkdir(parents=True, exist_ok=True)
    S.assign(label=Ct.label).to_csv(args.out_dir / "eval016_participant_scores.csv")      # Drive (katılımcı düzeyi)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    md = report_md(R)
    js = json.dumps(R, indent=2, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    assert ids_free(md + js, C.index), "raporda katılımcı ID'si var (D-014) → yazılmadı"
    (args.report_dir / "EVAL-016_context.json").write_text(js)
    (args.report_dir / "EVAL-016_context.md").write_text(md)
    print(md)
    print(f"bitti ({R['elapsed_s']} s)")


if __name__ == "__main__":
    main()
