"""
EVAL-016 — POST HOC duyarlılıklar (PH1–PH3). Aşama 1 sonuçları GÖRÜLDÜKTEN SONRA önerildi ve Alper'in onayıyla
(2026-10-11) çalıştırılır. Bunlar C1 / C2 kararlarını DEĞİŞTİRMEZ; yalnız yorumun sınırlarını gösterir.
Tanımlar ve yorum kuralı çalıştırmadan önce yazıldı: docs/EVAL016_STAGE2_PLAN.md Bölüm 5.

    PH0  bütünlük: kaydedilmiş ŝ'lerle birincil T1 yeniden hesaplanır; Aşama 1 raporuyla aynı olmalı (assert)
    PH1  yaş esnek: bağlam = cr(saat, 4) + cr(tarih, 4) + cr(yaş, 4)   (birincil modelde yaş doğrusal)
    PH2  demografi: birincil bağlam + cinsiyet + sigara (şu an / bırakmış / hiç)   (RQ9, D-006)
    PH3  K19 sayısal yeniden doğrulama: E2h Jeffreys-cezalı koşullu logit, aynı amaç fonksiyonu; birden fazla başlangıç
         (0 ve klasik ML), BFGS + Nelder–Mead; yakınsama = yerel minimum denetimi + başlangıçların uyuşması
Her PH'de: Firth PLR p, SD başına OR, CV3 p; Holm (6 backbone) yalnız karşılaştırma için.

YORUM KURALI (önceden yazıldı): Aşama-1 koşullarını sağlayan bir backbone, PH1 ya da PH2'de Holm p ≥ 0.05 ya da
CV3 p ≥ 0.05 çıkarsa "yaş / demografi ayarına duyarlı" diye işaretlenir. PH3'te yön değişirse K19 "karar etkilendi"
olarak yeniden açılır. Hiçbiri C1 durumunu kendiliğinden değiştirmez.

GİRDİ   Drive experiments/context_eval/eval016_participant_scores.csv (katılımcı düzeyi; Aşama 1 çıktısı),
        participant_context.csv, reports/context/EVAL-016_context.json (git; bütünlük karşılaştırması)
ÇIKTI   reports/context/EVAL-016_posthoc.{md,json} (yalnız agrega, ID taraması)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import minimize
from statsmodels.discrete.conditional_models import ConditionalLogit
from statsmodels.tools.numdiff import approx_fprime

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_context as E  # noqa: E402
import run_probes as RP  # noqa: E402



def design(D: pd.DataFrame, kind: str) -> np.ndarray:
    """Bağlam matrisi: 'primary' (E.context_matrix ile aynı), 'age_spline' (PH1), 'demog' (PH2)."""
    if kind == "primary":
        return E.context_matrix(D, 4)
    terms = ["cr(start_hour, df=4, constraints='center')", "cr(days, df=4, constraints='center')"]
    terms.append("cr(age, df=4, constraints='center')" if kind == "age_spline" else "age")
    if kind == "demog":
        terms += ["C(sex, Treatment('F'))", "C(smoking, Treatment('never'))"]
    Md = patsy.dmatrix(" + ".join(terms), D, return_type="dataframe")       # sabitli: kategorikler bir düzey eksik kodlanır
    assert Md.columns[0] == "Intercept"
    M = np.asarray(Md.iloc[:, 1:], float)
    M = np.column_stack([E.zstd(c) for c in M.T])
    X = sm.add_constant(M, has_constant="add")
    assert np.linalg.matrix_rank(X) == X.shape[1], f"{kind}: bağlam matrisi tam ranklı değil"
    return X


def clogit_firth_v2(y, X, groups, starts=None) -> dict:
    """Jeffreys-cezalı koşullu logit (E.clogit_firth ile AYNI amaç fonksiyonu); yalnız sayısal yöntem farklı (K19).
    Bilgi matrisi skorun sayısal türevi olduğundan amaç fonksiyonunda küçük bir gürültü var; Aşama 1'deki 1e-7 gradyan
    toleransı bu tabanın altında kaldığı için "yakınsamadı" bayrağı çıktı. Burada:
      her başlangıçtan BFGS (merkezi fark gradyanı, h = 1e-4) → Nelder–Mead cilası;
      yakınsama = (i) yerel minimum denetimi: her koordinatta ±1e-3 adım amaç fonksiyonunu 1e-9'dan fazla düşürmüyor,
                  (ii) en az iki başlangıç aynı noktaya varıyor (katsayı farkı < 1e-3)."""
    m = ConditionalLogit(np.asarray(y), np.asarray(X, float), groups=np.asarray(groups))

    def info(b):
        Hs = approx_fprime(b, m.score, centered=True)
        return -(Hs + Hs.T) / 2

    def negpl(b):
        sign, logdet = np.linalg.slogdet(info(b))
        if sign <= 0 or not np.isfinite(logdet):
            return 1e12
        return -(m.loglike(b) + 0.5 * logdet)

    def grad(b, h=1e-4):
        g = np.empty_like(b)
        for j in range(len(b)):
            e = np.zeros_like(b)
            e[j] = h
            g[j] = (negpl(b + e) - negpl(b - e)) / (2 * h)
        return g

    k = X.shape[1]
    starts = ([np.zeros(k), np.random.default_rng(0).normal(0, 1, k)]          # 0, sabit tohumlu rastgele, (sınırlıysa) ML
              + [np.asarray(s, float) for s in (starts or []) if np.all(np.abs(s) < 20)])
    fits = []
    for s0 in starts:
        r = minimize(negpl, s0, jac=grad, method="BFGS", options={"gtol": 1e-6, "maxiter": 2000})
        r2 = minimize(negpl, r.x, method="Nelder-Mead", options={"xatol": 1e-7, "fatol": 1e-11, "maxfev": 600})
        x, f = (r2.x, float(r2.fun)) if r2.fun <= r.fun else (r.x, float(r.fun))
        fits.append({"x": x, "f": f})
    best = min(fits, key=lambda d: d["f"])
    xb, fb = best["x"], best["f"]
    local_min = all(negpl(xb + sgn * 1e-3 * np.eye(k)[j]) >= fb - 1e-9 for j in range(k) for sgn in (1, -1))
    agree = sum(np.max(np.abs(f["x"] - xb)) < 1e-3 for f in fits)
    spread = float(max(np.max(np.abs(f["x"] - xb)) for f in fits))
    cov = np.linalg.pinv(info(xb))
    return {"coef": xb, "se": np.sqrt(np.clip(np.diag(cov), 0, None)), "converged": bool(local_min and agree >= 2),
            "local_min": bool(local_min), "n_agree": int(agree), "n_starts": len(fits), "start_spread": spread,
            "grad_max": float(np.max(np.abs(grad(xb)))), "pl": -fb}


def e2h_v2(Ct: pd.DataFrame, s: pd.Series) -> dict:
    """E.e2h_klr ile aynı alt küme ve tasarım; fit clogit_firth_v2; başlangıçlar: 0, klasik ML koşullu logit."""
    D = Ct[Ct.in_time_overlap].copy()
    D["s"] = s.reindex(D.index).to_numpy()
    D = D[D.groupby("hb").label.transform("nunique") == 2]
    X = np.column_stack([E.zstd(D.s), E.zstd(D.age), E.zstd(D.start_hour), E.zstd(D.days)])
    y, g = D.label.to_numpy(), D.hb.to_numpy()
    starts = []
    try:
        ml = ConditionalLogit(y, X, groups=g).fit(disp=0)
        starts.append(ml.params)
        ml_or = float(np.exp(ml.params[0]))
    except Exception:
        ml_or = float("nan")
    f = clogit_firth_v2(y, X, g, starts)
    b, se = float(f["coef"][0]), float(f["se"][0])
    return {"n_asthma": int(y.sum()), "n_healthy": int((1 - y).sum()), "or_sd": float(np.exp(b)),
            "ci": [float(np.exp(b - 1.96 * se)), float(np.exp(b + 1.96 * se))], "p": float(2 * stats.norm.sf(abs(b) / se)),
            "converged": f["converged"], "local_min": f["local_min"], "n_agree": f["n_agree"], "n_starts": f["n_starts"],
            "grad_max": f["grad_max"], "start_spread": f["start_spread"], "ml_or_sd": ml_or}


def t1_row(y, X, s, day) -> dict:
    r = E.t1(y, X, s, groups=day, ml=False, jackknife=True)
    return {k: r[k] for k in ("lr", "p_lrt", "or_sd", "coef", "se", "p_cv3", "se_cv3", "cv3_fails", "converged")}


def report_md(R: dict) -> str:
    f3 = E.f3
    fp = E.fp
    L = ["# EVAL-016 — POST HOC duyarlılıklar (PH1–PH3; otomatik üretildi; katılımcı ID'si içermez)", "",
         "**Aşama 1 sonuçları görüldükten sonra önerildi; C1 / C2 kararlarını değiştirmez.** Tanımlar ve yorum kuralı "
         "çalıştırmadan önce yazıldı (`docs/EVAL016_STAGE2_PLAN.md` Bölüm 5).", "",
         f"**PH0 bütünlük:** kaydedilmiş ŝ ile birincil T1 Aşama 1 raporunu yeniden üretti (en büyük |Δ LR| = "
         f"{R['ph0_max_abs_diff_lr']:.2e}). Kohort {R['n']} ({R['n_asthma']} / {R['n'] - R['n_asthma']}).", "",
         "## PH1 / PH2 — T1 farklı bağlam modelleriyle", "",
         "| kol | birincil: Holm p · OR · CV3 p | PH1 yaş spline: Holm p · OR · CV3 p | PH2 + cinsiyet + sigara: Holm p · OR · CV3 p | işaret |",
         "|---|---|---|---|---|"]
    for a in R["arms"]:
        cells = []
        for k in ("primary", "age_spline", "demog"):
            t = R["t1"][k][a]
            hp = R["holm"][k].get(a)
            cells.append(f"{fp(hp) if hp is not None else 'ref. ' + fp(t['p_lrt'])} · {f3(t['or_sd'])} · {fp(t['p_cv3'])}")
        L.append(f"| {a} | " + " | ".join(cells) + f" | {R['flags'].get(a, '—')} |")
    L += ["", "İşaret (önceden yazılmış kural): Aşama-1 koşullarını sağlayan bir backbone PH1 ya da PH2'de Holm p ≥ 0.05 ya da "
          "CV3 p ≥ 0.05 çıkarsa **yaş / demografi ayarına duyarlı**. MFCC kolları referans (Holm ailesinde değil).", "",
          "## PH3 — K19: E2h cezalı koşullu logitin sayısal yeniden doğrulaması", "",
          "| kol | Aşama 1: OR (yakınsadı) | PH3: OR [%95] | p | yakınsadı (yerel min., uyuşan başlangıç) | başlangıçlar arası en büyük fark | ML OR |",
          "|---|---|---|---|---|---|---|"]
    for a in R["arms"]:
        o, n = R["e2h_orig"][a], R["e2h_v2"][a]
        L.append(f"| {a} | {f3(o['or_sd'])} ({'evet' if o.get('converged') else 'hayır'}) | {f3(n['or_sd'])} "
                 f"[{f3(n['ci'][0])}, {f3(n['ci'][1])}] | {fp(n['p'])} | {'evet' if n['converged'] else 'hayır'} "
                 f"({'evet' if n['local_min'] else 'hayır'}, {n['n_agree']}/{n['n_starts']}) | {n['start_spread']:.1e} | {f3(n['ml_or_sd'])} |")
    L += ["", f"K19 sonucu: {R['k19']}", "", f"Süre {R['elapsed_s']} s.", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, type=Path)
    ap.add_argument("--context", required=True, type=Path)
    ap.add_argument("--orig-report", required=True, type=Path)
    ap.add_argument("--report-dir", required=True, type=Path)
    ap.add_argument("--n-jobs", type=int, default=2)
    a = ap.parse_args()
    t0 = time.time()
    orig = json.loads(a.orig_report.read_text())
    S = pd.read_csv(a.scores).set_index("participant_id")
    arms = list(orig["arms"])
    assert set(arms) <= set(S.columns), "skor dosyasında kol eksik"
    C, Ct, _ = E.load_context_table(a.context, set(S.index))
    if not {"sex", "smoking"} <= set(Ct.columns):
        raw = pd.read_csv(a.context).set_index("participant_id")
        Ct = Ct.join(raw[["sex", "smoking"]])
    assert list(Ct.index) == list(S.index) or set(Ct.index) == set(S.index)
    S = S.reindex(Ct.index)
    assert (S.label.to_numpy() == Ct.label.to_numpy()).all() and S[arms].notna().all().all()
    miss = Ct[["sex", "smoking"]].isna().sum().to_dict()
    assert not any(miss.values()), f"PH2: cinsiyet / sigara eksik {miss}"
    y = Ct.label.to_numpy()
    day = Ct.recording_date.astype("category").cat.codes.to_numpy()
    R = {"exp": "EVAL-016 post hoc", "date": time.strftime("%Y-%m-%d"), "n": int(len(y)), "n_asthma": int(y.sum()),
         "arms": arms, "t1": {}, "holm": {}, "note": "POST HOC; C1/C2 kararlarını değiştirmez"}
    for kind in ("primary", "age_spline", "demog"):
        X = design(Ct, kind)
        R["t1"][kind] = {arm: t1_row(y, X, S[arm].to_numpy(), day) for arm in arms}
        R["holm"][kind] = RP.holm({b: R["t1"][kind][b]["p_lrt"] for b in orig["backbones"]})
        print(kind, {b: round(R["t1"][kind][b]["p_lrt"], 5) for b in arms}, flush=True)
    d = [abs(R["t1"]["primary"][arm]["lr"] - orig["t1"][arm]["lr"]) for arm in arms]
    R["ph0_max_abs_diff_lr"] = float(max(d))
    assert R["ph0_max_abs_diff_lr"] < 1e-4, f"PH0: skor dosyası Aşama 1 raporunu yeniden üretmiyor ({max(d)})"
    R["flags"] = {}
    for b in orig["backbones"]:
        if not orig["verdicts"][b]["stage1_all_met"]:
            R["flags"][b] = "Aşama 1'de sağlanmadı"
            continue
        sens = any(R["holm"][k][b] >= 0.05 or R["t1"][k][b]["p_cv3"] >= 0.05 for k in ("age_spline", "demog"))
        R["flags"][b] = "**yaş / demografi ayarına duyarlı**" if sens else "PH1–PH2'ye dayanıklı"
    R["e2h_orig"] = {arm: orig["e2h"][arm] for arm in arms}
    from joblib import Parallel, delayed
    res = Parallel(n_jobs=a.n_jobs)(delayed(e2h_v2)(Ct, S[arm]) for arm in arms)
    R["e2h_v2"] = dict(zip(arms, res))
    for arm in arms:
        print("PH3", arm, round(R["e2h_v2"][arm]["or_sd"], 3), R["e2h_v2"][arm]["converged"], flush=True)
    dirs_same = all((R["e2h_v2"][x]["or_sd"] > 1) == (orig["e2h"][x]["or_sd"] > 1) for x in arms)
    all_conv = all(R["e2h_v2"][x]["converged"] for x in arms)
    R["k19"] = ("yön her kolda aynı; " if dirs_same else "**YÖN DEĞİŞTİ (karar etkilendi)**; ") + \
               ("bütün yeni fit'ler yakınsadı" if all_conv else "bazı yeni fit'ler yine yakınsamadı")
    R["elapsed_s"] = round(time.time() - t0, 1)
    md = report_md(R)
    js = json.dumps(R, indent=2, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    assert E.ids_free(md + js, Ct.index), "raporda katılımcı ID'si var (D-014) → yazılmadı"
    a.report_dir.mkdir(parents=True, exist_ok=True)
    (a.report_dir / "EVAL-016_posthoc.json").write_text(js)
    (a.report_dir / "EVAL-016_posthoc.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
