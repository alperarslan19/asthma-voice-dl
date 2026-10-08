"""
Faz 0 / SIM-001 — Birincil test T1'in (D-023) gücü ve yanlış pozitif riski: SES OLMADAN simülasyon

BİLİMSEL AMAÇ
    T1 şunu sorar: "ses skoru, yaş + saat + tarih modeline astım bilgisi ekliyor mu?"
    Ses modelini çalıştırmadan ÖNCE iki şeyi bilmek istiyoruz:
      1. GÜÇ: Bağlamdan bağımsız gerçek bir ses sinyali varsa, bu katılımcı sayısı ve bu bağlam yapısıyla
         T1 onu ne sıklıkla yakalar? (Tavan etkisi: bağlam tek başına AUC ~0.93.)
      2. YANLIŞ POZİTİF: Ses HİÇ astım bilgisi taşımayıp yalnız bağlamı kodlarsa, T1 yine de "bilgi ekliyor"
         der mi? Üç kestirme yolu senaryosu:
           S1 düzgün  : ses, bağlam modelinin zaten içerdiği biçimde bağlamı kodlar (doğru belirtilmiş model)
           S2 basamak : ses "sabah mı?" ve "geç dönem mi?" basamaklarını kodlar (kuadratik/lineer bağlam
                        modeli bunu tam yakalayamaz → yanlış belirtilmiş model)
           S3 gün     : ses, her kayıt gününe özgü ölçülmemiş bir akustik etkiyi kodlar (gün içi katılımcılar
                        bağımsız değil → katılımcı düzeyinde çıkarım fazla iyimser olabilir)
    VERİ: gerçek etiketler ve gerçek bağlam (saat, tarih, yaş, gün); YALNIZ ses skoru yapay:
          skor = gamma * astım + lambda * kestirme(bağlam) + gürültü,   gürültü ~ N(0, 1)
          gamma = bağlamdan bağımsız sinyal (aynı bağlamda astım–sağlıklı farkı, gürültü SD'si biriminde);
          aynı bağlamda ayrım gücü AUC = Phi(gamma / sqrt(2)).
    TESTLER
          T1-Wald  : lojistik regresyon y ~ bağlam + skor; skor katsayısı için Wald p (katılımcı düzeyi)
          T1-küme  : aynı model, kayıt gününe göre küme-dayanıklı SE
          E2h-KLR  : zaman penceresinde, 1 saatlik dilimlere göre koşullu lojistik regresyon,
                     y ~ skor + yaş + saat + tarih
          ΔAUC     : örneklem içi AUC(bağlam + skor) − AUC(bağlam) (yalnız büyüklük fikri vermek için)
    VARSAYIMLAR [HYPOTHESIS]: skor tek boyutlu ve gürültüsü normal; kestirme gücü lambda = 1 (güçlü);
          gerçek modelin OOF skoru daha gürültülü olacağından güç burada İYİMSER tahmin edilir.

GİRDİ  : participant_context.csv (Drive, katılımcı düzeyi)
ÇIKTI  : reports/recording_context/SIM-001_t1_power.{json,md} (yalnız agrega)
ÇALIŞTIRMA
    python scripts/simulate_t1_power.py --context .../participant_context.csv --out-report reports/recording_context
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from statsmodels.discrete.conditional_models import ConditionalLogit

warnings.filterwarnings("ignore")


def zstd(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std()


def fit_p(y, X, groups=None):
    """Son sütunun (skor) Wald p-değeri; küme verilirse küme-dayanıklı SE. Yakınsamazsa NaN."""
    try:
        m = sm.Logit(y, X)
        r = m.fit(disp=0, maxiter=200, cov_type="cluster", cov_kwds={"groups": groups}) if groups is not None \
            else m.fit(disp=0, maxiter=200)
        return float(r.pvalues[-1]), r.predict(X)
    except Exception:
        return float("nan"), None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", required=True, type=Path)
    ap.add_argument("--out-report", required=True, type=Path)
    ap.add_argument("--n-sim", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()

    d = pd.read_csv(args.context)
    a = d[d["has_audio"] & d["in_paper_cohort"]].dropna(subset=["start_hour", "days", "age", "recording_date"]).reset_index(drop=True)
    y = a["label"].to_numpy(int)
    assert set(np.unique(y)) == {0, 1}
    n1, n0 = int(y.sum()), int((1 - y).sum())
    day = a["recording_date"].astype("category").cat.codes.to_numpy()

    # Bağlam modelleri (D-023 T1'in "bağlam" kısmı)
    quad = np.column_stack([zstd(a["age"].to_numpy(float)), zstd(a["start_hour"].to_numpy(float)),
                            zstd(a["start_hour"].to_numpy(float) ** 2), zstd(a["days"].to_numpy(float))])
    spl = np.asarray(patsy.dmatrix("cr(start_hour, df=4, constraints='center') + cr(days, df=4, constraints='center') + age - 1", a, return_type="dataframe"))
    spl = np.column_stack([zstd(c) if c.std() > 0 else c for c in spl.T])
    ctx = {"quad": sm.add_constant(quad), "spline": sm.add_constant(spl)}
    for k, X in ctx.items():
        assert np.linalg.matrix_rank(X) == X.shape[1], f"{k} bağlam matrisi tam ranklı değil"
    auc_ctx = {k: float(roc_auc_score(y, sm.Logit(y, X).fit(disp=0).predict(X))) for k, X in ctx.items()}

    # Kestirme (shortcut) değişkenleri
    lin_q = sm.Logit(y, ctx["quad"]).fit(disp=0)
    z_smooth = zstd(ctx["quad"] @ lin_q.params)
    z_step = zstd((a["start_hour"] < 12).astype(float).to_numpy() + (a["period"] == "late").astype(float).to_numpy())

    # E2h alt kümesi (zaman penceresi + 1 saatlik dilim)
    e2 = a["in_time_overlap"].to_numpy(bool)
    hb = np.floor(a["start_hour"].to_numpy())
    Xe_cov = np.column_stack([zstd(a.loc[e2, "age"].to_numpy(float)), zstd(a.loc[e2, "start_hour"].to_numpy(float)),
                              zstd(a.loc[e2, "days"].to_numpy(float))])
    ye, ge = y[e2], hb[e2]
    informative = pd.DataFrame({"y": ye, "g": ge}).groupby("g")["y"].transform("nunique").to_numpy() == 2
    n_e2h = (int(ye[informative].sum()), int((1 - ye[informative]).sum()))

    rng = np.random.default_rng(args.seed)
    gammas = [0.0, 0.25, 0.5, 0.75, 1.0]
    out = {"n_asthma": n1, "n_healthy": n0, "n_days": int(len(np.unique(day))), "e2h_informative_asthma_healthy": n_e2h,
           "context_auc_in_sample": auc_ctx, "n_sim": args.n_sim, "lambda": 1.0, "results": []}
    for scen in ["S1_smooth", "S2_step", "S3_day"]:
        for gm in gammas:
            rec = {"scenario": scen, "gamma": gm, "within_context_auc": round(float(norm.cdf(gm / np.sqrt(2))), 3)}
            p = {k: [] for k in ["quad", "spline", "spline_cluster", "e2h_clr"]}
            dauc = []
            for _ in range(args.n_sim):
                if scen == "S1_smooth":
                    z = z_smooth
                elif scen == "S2_step":
                    z = z_step
                else:
                    u = rng.normal(0, 1, day.max() + 1)
                    z = zstd(u[day])
                s = zstd(gm * y + 1.0 * z + rng.normal(0, 1, len(y)))
                pq, _ = fit_p(y, np.column_stack([ctx["quad"], s]))
                ps, pred = fit_p(y, np.column_stack([ctx["spline"], s]))
                pc, _ = fit_p(y, np.column_stack([ctx["spline"], s]), groups=day)
                p["quad"].append(pq); p["spline"].append(ps); p["spline_cluster"].append(pc)
                if pred is not None:
                    dauc.append(roc_auc_score(y, pred) - auc_ctx["spline"])
                try:
                    r = ConditionalLogit(ye[informative], np.column_stack([s[e2][informative], Xe_cov[informative]]),
                                         groups=ge[informative]).fit(disp=0)
                    p["e2h_clr"].append(float(r.pvalues[0]))
                except Exception:
                    p["e2h_clr"].append(float("nan"))
            for k, v in p.items():
                v = np.array(v)
                rec[f"reject_rate_{k}"] = round(float(np.nanmean(v < 0.05)), 3)
                rec[f"reject_rate_{k}_a01"] = round(float(np.nanmean(v < 0.01)), 3)  # ~Holm eşiği (6 backbone: 0.0083)
                rec[f"n_failed_{k}"] = int(np.isnan(v).sum())
            rec["mean_delta_auc_in_sample_spline"] = round(float(np.mean(dauc)), 4)
            out["results"].append(rec)
            print(rec, flush=True)
    out["runtime_s"] = round(time.time() - t0, 1)

    args.out_report.mkdir(parents=True, exist_ok=True)
    (args.out_report / "SIM-001_t1_power.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    L = ["# SIM-001 — T1'in gücü ve yanlış pozitif riski (ses kullanılmadan, gerçek bağlamla simülasyon)", "",
         f"- n = {n1} astım / {n0} sağlıklı, {out['n_days']} kayıt günü; E2h'de bilgi taşıyan tabakalar: "
         f"{n_e2h[0]} astım / {n_e2h[1]} sağlıklı",
         f"- Bağlam modelinin örneklem içi AUC'si: kuadratik {auc_ctx['quad']:.3f}, spline {auc_ctx['spline']:.3f}",
         f"- Skor = gamma·astım + 1·kestirme(bağlam) + N(0,1); her hücrede {args.n_sim} simülasyon; ret = p < 0.05", "",
         "gamma = 0 satırları **yanlış pozitif oranıdır** (ideal: 0.05). gamma > 0 satırları **güçtür**.", "",
         "Hücreler: ret oranı p < 0.05 (parantez içinde p < 0.01).", "",
         "| senaryo | gamma | aynı bağlamda AUC | T1 kuadratik | T1 spline | T1 spline + gün kümesi | E2h KLR | ort. ΔAUC (spline) |",
         "|---|---|---|---|---|---|---|---|"]
    for r in out["results"]:
        c = lambda k: f"{r['reject_rate_' + k]} ({r['reject_rate_' + k + '_a01']})"
        L.append(f"| {r['scenario']} | {r['gamma']} | {r['within_context_auc']} | {c('quad')} | {c('spline')} | "
                 f"{c('spline_cluster')} | {c('e2h_clr')} | {r['mean_delta_auc_in_sample_spline']} |")
    (args.out_report / "SIM-001_t1_power.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
