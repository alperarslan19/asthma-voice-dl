"""
SIM-002 — K2 geçerlilik çalışması: gün-kümeli test bu veri yapısında geçerli mi? (SES OLMADAN; D-038)

BİLİMSEL AMAÇ
    EVAL-016'da C1 koşul 3 "gün-kümeli test p < 0.05 ve OR > 1" (K2). Alper K2'yi şu koşulla kabul etti: gün-kümeli testin
    mevcut veri yapısında GEÇERLİ biçimde uygulanabildiği teyit edilmeli. Bu script, gerçek etiket ve gerçek bağlam
    yapısında (saat, tarih, yaş, kayıt günü; 65 gün, 53'ü tek etiketli), YAPAY ve sinyalsiz ses skorlarıyla iki gün-kümeli
    varyans tahmincisinin yanlış pozitif oranını ve koşul 3'ün güç maliyetini ölçer:
        CR1  Firth skor katkılarıyla sandviç, G/(G−1) düzeltmesi, normal referans (planın ilk metni)
        CV3  gün-kümeli jackknife: her gün dışarıda bırakılarak Firth T1 yeniden fit, V = (G−1)/G Σ (b_(g) − b̂)²,
             t(G−1) referansı (MacKinnon, Nielsen & Webb 2023)
    Birincil T1-PLR (Firth) aynı simülasyonlarda raporlanır.
    Senaryolar SIM-001 ile aynı (S1 düzgün bağlam = doğru belirtilmiş model; S2 basamak; S3 ölçülmemiş gün etkisi):
        skor = γ · astım + kestirme(bağlam) + gürültü
VERİ   participant_context.csv (katılımcı düzeyi; Drive / özel kopya) + splits/outer_r0.csv (EXP-016 kohortu).
       Hiçbir ses / model skoru kullanılmaz: EVAL-016 sonuçlarına bakılmaz.
ÇIKTI  reports/context/SIM-002_k2_validity.{json,md} (yalnız agrega)
ÇALIŞTIRMA
    python scripts/simulate_k2_validity.py --context .../participant_context.csv --splits-dir .../splits \\
        --out-report reports/context [--n0 1000 --n1 400 --n-jobs 2]
VARSAYIMLAR [HYPOTHESIS]
    Skor tek boyutlu, gürültü normal; gerçek ŝ daha ağır kuyruklu olabilir. Gün etkisi S3'te normal ve etiketten bağımsız.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_context as E  # noqa: E402

CELLS = [("S1_smooth", 0.0), ("S2_step", 0.0), ("S3_day", 0.0), ("S1_smooth", 0.5), ("S1_smooth", 0.75), ("S3_day", 0.75),
         ("S1_smooth", 1.0), ("S1_smooth", 1.25)]       # tohum hücre sırasına bağlı: yeni hücreler yalnız sona eklenir


def one(y, Xc, day, z_smooth, z_step, scen, gamma, seed):
    rng = np.random.default_rng(seed)
    G = int(day.max() + 1)
    z = z_smooth if scen == "S1_smooth" else z_step if scen == "S2_step" else E.zstd(rng.normal(0, 1, G)[day])
    s = gamma * y + z + rng.normal(0, 1, len(y))
    r = E.t1(y, Xc, s, groups=day, ml=False, jackknife=True)
    return r["p_lrt"], r["p_cluster"], r["p_cv3"], r["or_sd"], r["cv3_fails"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", required=True, type=Path)
    ap.add_argument("--splits-dir", required=True, type=Path)
    ap.add_argument("--out-report", required=True, type=Path)
    ap.add_argument("--n0", type=int, default=1000, help="γ = 0 (yanlış pozitif) hücresi başına simülasyon")
    ap.add_argument("--n1", type=int, default=400, help="γ > 0 (güç) hücresi başına simülasyon")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n-jobs", type=int, default=2)
    a = ap.parse_args()
    t0 = time.time()
    coh = set(pd.read_csv(a.splits_dir / "outer_r0.csv").participant_id)
    _, Ct, info = E.load_context_table(a.context, coh)
    y = Ct.label.to_numpy(float)
    day = Ct.recording_date.astype("category").cat.codes.to_numpy()
    G = int(day.max() + 1)
    lab = Ct.groupby("recording_date").label.nunique()
    info.update({"single_label_days": int((lab == 1).sum()), "mixed_days": int((lab == 2).sum()),
                 "healthy_am": int(((Ct.label == 0) & (Ct.start_hour < 12)).sum()),
                 "patients_pm": int(((Ct.label == 1) & (Ct.start_hour >= 12)).sum()),
                 "late_patients": int(((Ct.period == "late") & (Ct.label == 1)).sum())})
    Xc = E.context_matrix(Ct, 4)
    q = np.column_stack([E.zstd(Ct.age), E.zstd(Ct.start_hour), E.zstd(Ct.start_hour ** 2), E.zstd(Ct.days)])
    z_smooth = E.zstd(sm.add_constant(q) @ E.firth_fit(sm.add_constant(q), y)["coef"])      # calibration_diagnostic ile aynı
    z_step = E.zstd((Ct.start_hour < 12).astype(float).to_numpy() + (Ct.period == "late").astype(float).to_numpy())
    print(info, "G =", G, flush=True)

    out = {"exp": "SIM-002", "date": time.strftime("%Y-%m-%d"), "cohort": info, "n_clusters": G, "n0": a.n0, "n1": a.n1,
           "seed": a.seed, "note": "yapay skor; gerçek ses / model skoru kullanılmadı", "results": {}}
    for k, (scen, gamma) in enumerate(CELLS):
        n = a.n0 if gamma == 0 else a.n1
        seeds = np.random.SeedSequence([a.seed, k]).generate_state(n)
        res = np.array(Parallel(n_jobs=a.n_jobs)(delayed(one)(y, Xc, day, z_smooth, z_step, scen, gamma, int(s)) for s in seeds))
        p, p1, p3, orr, fails = res.T
        pos = orr > 1
        R = {"n": n, "cv3_refit_failures": int(fails.sum()),
             "two_sided_05": {"PLR": float(np.mean(p < .05)), "CR1": float(np.mean(p1 < .05)), "CV3": float(np.mean(p3 < .05))},
             "two_sided_01": {"PLR": float(np.mean(p < .01)), "CR1": float(np.mean(p1 < .01)), "CV3": float(np.mean(p3 < .01))},
             "wilson_05": {t: E.wilson_ci(int((v < .05).sum()), n) for t, v in [("PLR", p), ("CR1", p1), ("CV3", p3)]}}
        for al in (0.05, 0.0083):                     # C1 biçimi: koşul 1 (PLR, α) ve koşul 3 (p < 0.05), ikisi de OR > 1
            R[f"c1_form_alpha_{al}"] = {"cond1": float(np.mean((p < al) & pos)),
                                        "cond1_and_CR1": float(np.mean((p < al) & (p1 < .05) & pos)),
                                        "cond1_and_CV3": float(np.mean((p < al) & (p3 < .05) & pos))}
        key = f"{scen} γ={gamma}"
        out["results"][key] = R
        print(key, json.dumps(R, ensure_ascii=False), f"{time.time() - t0:.0f} s", flush=True)
    out["elapsed_s"] = round(time.time() - t0, 1)

    a.out_report.mkdir(parents=True, exist_ok=True)
    f = lambda x: f"{x:.3f}"                                                                     # noqa: E731
    L = ["# SIM-002 — K2 geçerlilik çalışması: gün-kümeli test (otomatik üretildi; yalnız agrega)", "",
         f"Gerçek etiket + gerçek bağlam yapısı, **yapay** skor (ses / model skoru kullanılmadı). T1 kohortu {info['n_t1']} "
         f"({info['n_t1_asthma']} astım / {info['n_t1_healthy']} sağlıklı), {G} kayıt günü ({info['single_label_days']} tek etiketli), "
         f"sabah sağlıklı {info['healthy_am']}, öğleden sonra hasta {info['patients_pm']}, geç dönem hasta {info['late_patients']}.", "",
         "## Yanlış pozitif (γ = 0) ve güç (γ > 0): iki yönlü p < 0.05 oranı [Wilson %95]", "",
         "| senaryo | n | T1-PLR | CR1 sandviç | CV3 jackknife | PLR p < 0.01 | CV3 p < 0.01 |", "|---|---|---|---|---|---|---|"]
    for key, R in out["results"].items():
        w = R["wilson_05"]
        L.append(f"| {key} | {R['n']} | {f(R['two_sided_05']['PLR'])} [{f(w['PLR'][0])}, {f(w['PLR'][1])}] | "
                 f"{f(R['two_sided_05']['CR1'])} [{f(w['CR1'][0])}, {f(w['CR1'][1])}] | "
                 f"{f(R['two_sided_05']['CV3'])} [{f(w['CV3'][0])}, {f(w['CV3'][1])}] | {f(R['two_sided_01']['PLR'])} | "
                 f"{f(R['two_sided_01']['CV3'])} |")
    L += ["", "## C1 biçimi: koşul 1 (PLR p < α, OR > 1) tek başına ve koşul 3 ile birlikte (tek yönlü)", "",
          "α = 0.0083, 6 backbone için Holm'un en sıkı eşiği. Tek yönlü nominal oran α/2.", "",
          "| senaryo | α | koşul 1 | koşul 1 ∧ CR1 | koşul 1 ∧ CV3 |", "|---|---|---|---|---|"]
    for key, R in out["results"].items():
        for al in (0.05, 0.0083):
            c = R[f"c1_form_alpha_{al}"]
            L.append(f"| {key} | {al} | {f(c['cond1'])} | {f(c['cond1_and_CR1'])} | {f(c['cond1_and_CV3'])} |")
    L += ["", f"CV3 yeniden fit başarısızlığı (toplam): {sum(R['cv3_refit_failures'] for R in out['results'].values())}. "
          f"Süre {out['elapsed_s']} s.", ""]
    md, js = "\n".join(L), json.dumps(out, indent=1, ensure_ascii=False)
    assert E.ids_free(md + js, Ct.index), "raporda katılımcı ID'si var (D-014) → yazılmadı"
    (a.out_report / "SIM-002_k2_validity.json").write_text(js)
    (a.out_report / "SIM-002_k2_validity.md").write_text(md)
    print("\n".join(L))


if __name__ == "__main__":
    main()
