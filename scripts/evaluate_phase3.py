"""
Faz 3 değerlendirmesi (docs/PHASE3_DESIGN.md Bölüm 5): `frozen-dg` (G1) ve `finetune` (EXP-030; E1, T1, C3, C2, RQ2 +
betimsel O1 aşırı-uyum / kalibrasyon). Tanımlar tasarım belgesinde sonuçlardan önce sabit; O1 kullanıcı sorusu üzerine
eklenen BETİMSEL bir okumadır (karar kuralı yok).

frozen-dg — G1: "gün parmak izi" kontrolü (D-041)
    EXP-016'nın AYNI probu gün-gruplu split'lerde (EXP-016DG; 3 tekrar) → katılımcı ŝ_DG (tekrarların dış-test füzyon
    logit ortalaması) → EVAL-016 ile AYNI T1 (Firth PLR; cr(saat, 4) + cr(tarih, 4) + yaş), gün-kümeli CV3, katılımcı
    bootstrap OR (B 2000). Okuma kuralı (önceden yazıldı, PHASE3_DESIGN 5.1): iddiadaki backbone'lar (cnn10, beats,
    wavlm_large; ayrıca sınırda cnn14) için ŝ_DG ile T1-PLR p < 0.05 ve OR > 1 ise "gün parmak izi yoluyla açıklanmıyor";
    değilse o backbone için iddia "gün / seans artefaktı dışlanamıyor"a düşer. Holm yalnız betimsel.
    E1 (fold AUC ort ± SD) betimsel; katılımcı-gruplu Faz 2 değerleriyle yan yana (farklı split → doğrudan test yok).

ÇIKTI  reports/phase3/G1_frozen_daygrouped.{md,json} (agrega) · Drive: g1_participant_scores.csv

finetune — EXP-030 (PHASE3_DESIGN 5.2)
    Skor: kayıt logit'i (pencere ortalaması; finetune.py) → katılımcı logit'i (kayıt ortalaması) → ŝ_FT (3 tekrar ortalaması).
    Dondurulmuş eş (C3): aynı fold, aynı backbone'un EXP-016DG tahmini (katılımcı füzyon logit'i); ssl kolunda her (tekrar,
    fold) için o fold'un D-035 seçimi. cnn10_scratch'in dondurulmuş eşi yok (C3 uygulanmaz; RQ2 onun içindir).
    E1   fold AUC ort ± SD (15) + tekrar başına havuzlanmış AUC + ŝ havuzlanmış AUC [katılımcı bootstrap %95]; dengeli doğruluk
         (eşik 0 logit) betimsel. FT − dondurulmuş fold-eşli ΔAUC (Nadeau–Bengio) BETİMSEL.
    T1   ŝ_FT ile EVAL-016'nın aynı T1'i (Firth PLR, gün-kümeli CV3, bootstrap OR). Holm (4 kol) betimsel.
    C3   LR(ŝ_FT) − LR(ŝ_dondurulmuş), aynı katılımcılar; katılımcı bootstrap (her örneklemde iki T1 de yeniden fit edilir).
         Pozitif: %95 CI 0'ı dışlıyor ve fark > 0. Negatif: CI 0'ı dışlıyor ve fark < 0. Değilse belirsiz. Keşifsel.
    C2   etiket-içi permütasyon kısmi testi (S10) ve L5 (hasta-içi PM katsayısı, sağlıklı AM − PM); FT ve dondurulmuş eş.
    RQ2  cnn10 − cnn10_scratch fold-eşli ΔAUC (Nadeau–Bengio, iki yönlü) + iki kolun T1'i.
    O1   (betimsel; post hoc kullanıcı sorusu) aşırı uyum / kalibrasyon: test kalibrasyon eğimi (y ~ ŝ lojistik; < 1 aşırı
         güven), ortalama |ŝ|, son eğitim kaybı (fold.json) ve sabit tahminci kaybı.
ÇIKTI  reports/phase3/EXP-030_finetune.{md,json} (agrega) · Drive: exp030_participant_scores.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
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


# ============================================================================= finetune (EXP-030)
FT_ARMS = ["cnn10", "cnn10_scratch", "ssl", "cnn14"]
FROZEN_PAIR = {"cnn10": "cnn10", "cnn14": "cnn14", "ssl": "d035", "cnn10_scratch": None}
SSL_CANDIDATES = ["beats", "wavlm_base_plus", "wavlm_large"]


def load_ft(ft_root: Path, arm: str, splits: dict, splits_dir: Path, context: Path) -> dict:
    """Bir FT kolunun bütün fold tahminleri → katılımcı logit'i (tekrar başına), fold kimliği, fold.json özetleri.
    Bütünlük: her (tekrar, fold) var; imzalar şimdiki split / bağlam dosyalarıyla aynı; katılımcılar tam o fold'un testi."""
    ctx_sha = RP.sha256_short(context)
    reps, folds, fj_all = {}, {}, []
    for r, S in splits.items():
        split_sha = RP.sha256_short(splits_dir / f"outer_r{r}.csv")
        parts = []
        for k in sorted(S.outer_fold.unique()):
            d = ft_root / arm / f"r{r}_k{k}"
            fj_p, pr_p = d / "fold.json", d / "preds.csv"
            assert fj_p.exists() and pr_p.exists(), f"{arm} r{r} k{k}: fold bitmemiş (fold.json / preds.csv yok)"
            fj = json.loads(fj_p.read_text())
            sig = fj["sig"]
            assert sig["splits"][f"outer_r{r}.csv"] == split_sha, f"{arm} r{r} k{k}: split imzası farklı"
            assert sig["context"] == ctx_sha, f"{arm} r{r} k{k}: bağlam imzası farklı"
            P = pd.read_csv(pr_p)
            assert np.isfinite(P.logit).all() and (P["repeat"] == r).all() and (P.fold == k).all()
            test = set(S[(S.outer_fold == k) & (S.role == "test")].participant_id)
            assert set(P.participant_id) == test, f"{arm} r{r} k{k}: tahmin edilen katılımcılar fold testiyle aynı değil"
            parts.append(P.groupby("participant_id").logit.mean().rename(r).to_frame().assign(fold=k))
            fj_all.append({k2: fj.get(k2) for k2 in ["repeat", "fold", "backbone_arm", "epochs", "final_train_loss",
                                                     "skipped_steps", "total_s", "eval_amp"]})
        D = pd.concat(parts)
        assert D.index.is_unique, f"{arm} r{r}: katılımcı iki fold'da"
        reps[r], folds[r] = D[r], D["fold"]
    R = pd.DataFrame(reps)
    assert R.notna().all().all()
    return {"per_rep": R, "fold": pd.DataFrame(folds), "fold_json": pd.DataFrame(fj_all)}


def frozen_pairs(per_rep_fr: dict, fold_of: pd.DataFrame, selection: pd.DataFrame | None) -> dict:
    """FT kolu → dondurulmuş eşin tekrar başına katılımcı logit'i (aynı fold, aynı backbone)."""
    out = {}
    for arm, src in FROZEN_PAIR.items():
        if src is None:
            continue
        if src != "d035":
            if src in per_rep_fr:
                out[arm] = per_rep_fr[src].copy()
            continue
        if selection is None or not set(SSL_CANDIDATES) & set(per_rep_fr):
            continue
        cols = {}
        for r in fold_of.columns:
            v = pd.Series(index=fold_of.index, dtype=float)
            for k, idx in fold_of[r].groupby(fold_of[r]).groups.items():
                hit = selection[(selection["repeat"] == r) & (selection.fold == k)]
                assert len(hit) == 1, f"D-035 seçimi yok: r{r} k{k}"
                bb = hit.selected.iloc[0]
                v.loc[idx] = per_rep_fr[bb].loc[idx, r]
            cols[r] = v
        out[arm] = pd.DataFrame(cols)
    return out


def fold_auc_table(per_rep: pd.DataFrame, fold_of: pd.DataFrame, y: pd.Series) -> pd.Series:
    vals = {}
    for r in per_rep.columns:
        for k, idx in per_rep[r].groupby(fold_of[r]).groups.items():
            vals[(r, k)] = roc_auc_score(y[idx], per_rep.loc[idx, r])
    return pd.Series(vals).sort_index()


def boot_auc(y, s, B: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    y, s, n, a = np.asarray(y), np.asarray(s), len(y), []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if y[i].min() != y[i].max():
            a.append(roc_auc_score(y[i], s[i]))
    return [float(v) for v in np.percentile(a, [2.5, 97.5])]


def lr_pair(y, Xc, s1, s2) -> tuple[float, float]:
    """Aynı örneklemde iki skorun T1 PLR istatistiği (bağlam modeline ŝ eklemenin cezalı olabilirlik oranı)."""
    out = []
    for s in (s1, s2):
        X1 = np.column_stack([Xc, E.zstd(s)])
        j = X1.shape[1] - 1
        full, null = E.firth_fit(X1, y), E.firth_fit(X1, y, fixed=j)
        if not (full["converged"] and null["converged"]):
            return float("nan"), float("nan")
        out.append(max(0.0, 2 * (full["pll"] - null["pll"])))
    return out[0], out[1]


def c3_boot(y, Xc, s_ft, s_fr, B: int, seed: int, n_jobs: int) -> dict:
    from joblib import Parallel, delayed
    y = np.asarray(y, float)
    lf, lz = lr_pair(y, Xc, s_ft, s_fr)
    rng = np.random.default_rng(seed)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(B)]

    def one(i):
        if y[i].min() == y[i].max():
            return float("nan")
        a, b = lr_pair(y[i], Xc[i], s_ft[i], s_fr[i])
        return a - b
    d = np.array(Parallel(n_jobs=n_jobs)(delayed(one)(i) for i in idx), float)
    ok = d[np.isfinite(d)]
    lo, hi = (np.percentile(ok, [2.5, 97.5]) if len(ok) >= 0.9 * B else (np.nan, np.nan))
    verdict = ("pozitif" if lo > 0 else "negatif" if hi < 0 else "belirsiz") if np.isfinite(lo) else "hesaplanamadı"
    return {"lr_ft": lf, "lr_frozen": lz, "diff": lf - lz, "ci": [float(lo), float(hi)], "n_ok": int(len(ok)), "B": B,
            "verdict": verdict}


def calib_slope(y, s) -> dict:
    """y ~ a + b·ŝ (ham logit ölçeği). b < 1: tahminler fazla uç (aşırı güven / aşırı uyum işareti); b > 1: fazla çekingen."""
    import statsmodels.api as sm
    s = np.asarray(s, float)
    try:
        r = sm.Logit(np.asarray(y, float), sm.add_constant(s)).fit(disp=0)
        b, (lo, hi) = float(r.params[1]), r.conf_int()[1]
        return {"slope": b, "ci": [float(lo), float(hi)], "mean_abs_logit": float(np.abs(s).mean()), "sd_logit": float(s.std(ddof=1))}
    except Exception as e:                                                     # ayrışma vb.: rapora yazılır
        return {"slope": float("nan"), "ci": [float("nan")] * 2, "mean_abs_logit": float(np.abs(s).mean()),
                "sd_logit": float(s.std(ddof=1)), "error": str(e)[:120]}


def cmd_finetune(a) -> None:
    from types import SimpleNamespace
    t0 = time.time()
    splits = {r: pd.read_csv(a.splits_dir / f"outer_r{r}.csv") for r in range(a.n_repeats)}
    cohort = set(splits[0].participant_id)
    arms = [x for x in a.arms.split(",") if x]
    assert set(arms) <= set(FT_ARMS)
    FT = {arm: load_ft(a.ft_root, arm, splits, a.splits_dir, a.context) for arm in arms}
    fold_of = FT[arms[0]]["fold"]
    for arm in arms:
        assert FT[arm]["fold"].equals(fold_of.loc[FT[arm]["fold"].index]), f"{arm}: fold ataması kollar arasında farklı"
    fr_arms = [b for b in a.frozen_backbones.split(",") if b]
    S_fr, per_rep_fr, _, _ = E.load_scores(a.probe_dir, fr_arms, splits, a.splits_dir, a.context)
    sel = pd.read_csv(a.selection) if a.selection else None
    if sel is not None:
        assert set(sel.selected) <= set(fr_arms), "D-035 seçimindeki backbone'ların dondurulmuş tahmini yüklenmedi"
    FZ = frozen_pairs(per_rep_fr, fold_of, sel)
    C, Ct, cinfo = E.load_context_table(a.context, cohort)
    y = Ct.label.to_numpy().astype(float)
    labels = C.label
    day = Ct.recording_date.astype("category").cat.codes.to_numpy()
    X = E.context_matrix(Ct, 4)
    ratio = float(np.mean([len(S[(S.outer_fold == 0) & (S.role == "test")]) / len(S[(S.outer_fold == 0) & (S.role == "train")])
                           for S in splits.values()]))
    largs = SimpleNamespace(n_boot=a.n_boot, seed=a.seed, spline_df=4)
    R = {"exp": "EXP-030 (Faz 3 fine-tune değerlendirmesi)", "date": time.strftime("%Y-%m-%d"), "cohort": cinfo,
         "n_repeats": a.n_repeats, "n_boot": a.n_boot, "n_perms": a.n_perms, "nb_ratio": ratio, "arms": {}}
    scores = {}
    for arm in arms:
        pr = FT[arm]["per_rep"]
        s_ft = pr.mean(1).reindex(Ct.index)
        assert s_ft.notna().all(), f"{arm}: T1 kohortunda tahmini olmayan katılımcı"
        scores[f"ft_{arm}"] = s_ft
        fa = fold_auc_table(pr, fold_of, labels)
        pooled_rep = [float(roc_auc_score(labels[pr.index], pr[r])) for r in pr.columns]
        ba = []
        for r in pr.columns:
            for k, idx in pr[r].groupby(fold_of[r]).groups.items():
                yy, pp = labels[idx].to_numpy(), (pr.loc[idx, r].to_numpy() > 0).astype(int)
                ba.append(0.5 * ((pp[yy == 1] == 1).mean() + (pp[yy == 0] == 0).mean()))
        A = {"e1": {"fold_auc_mean": float(fa.mean()), "fold_auc_sd": float(fa.std(ddof=1)), "fold_auc_min": float(fa.min()),
                    "fold_auc_max": float(fa.max()), "pooled_auc_per_repeat": pooled_rep,
                    "pooled_auc_shat": float(roc_auc_score(y, s_ft)), "pooled_auc_shat_ci": boot_auc(y, s_ft, a.n_boot, a.seed),
                    "balanced_acc_thr0_mean": float(np.mean(ba))},
             "t1": E.t1(y, X, s_ft.to_numpy(), groups=day, ml=False, jackknife=True),
             "boot_or": E.boot_or(y, X, s_ft.to_numpy(), a.n_boot, a.seed),
             "c2_partial": E.partial_perm_test(y, s_ft.to_numpy(), Ct, a.n_perms, a.seed),
             "l5": E.l5_cells(Ct, s_ft, largs),
             "o1": calib_slope(y, s_ft.to_numpy()),
             "train": FT[arm]["fold_json"].agg({"epochs": "mean", "final_train_loss": ["mean", "std"],
                                                "skipped_steps": "sum", "total_s": "sum"}).to_dict(),
             "backbones_used": FT[arm]["fold_json"].backbone_arm.value_counts().to_dict()}
        if arm in FZ:
            zr = FZ[arm]
            s_fr = zr.mean(1).reindex(Ct.index)
            scores[f"frozen_{arm}"] = s_fr
            fz = fold_auc_table(zr, fold_of, labels)
            d = (fa - fz.reindex(fa.index)).to_numpy()
            t, p = RP.nadeau_bengio(d, ratio)
            A["frozen"] = {"e1": {"fold_auc_mean": float(fz.mean()), "fold_auc_sd": float(fz.std(ddof=1)),
                                  "pooled_auc_shat": float(roc_auc_score(y, s_fr))},
                           "t1": E.t1(y, X, s_fr.to_numpy(), groups=day, ml=False, jackknife=True),
                           "c2_partial": E.partial_perm_test(y, s_fr.to_numpy(), Ct, a.n_perms, a.seed),
                           "l5": E.l5_cells(Ct, s_fr, largs), "o1": calib_slope(y, s_fr.to_numpy()),
                           "corr_with_ft": float(np.corrcoef(s_ft, s_fr)[0, 1])}
            A["delta_e1_vs_frozen"] = {"mean": float(d.mean()), "sd": float(d.std(ddof=1)), "nb_t": t, "nb_p": p,
                                       "n_ft_better": int((d > 0).sum()), "n_folds": int(len(d))}
            A["c3"] = c3_boot(y, X, s_ft.to_numpy(), s_fr.to_numpy(), a.n_boot_c3, a.seed, a.n_jobs)
        R["arms"][arm] = A
        print(f"  {arm}: E1 {A['e1']['fold_auc_mean']:.3f} ± {A['e1']['fold_auc_sd']:.3f} | T1 p {A['t1']['p_lrt']:.3g} "
              f"OR {A['t1']['or_sd']:.2f}" + (f" | C3 {A['c3']['diff']:+.2f} {A['c3']['ci']} {A['c3']['verdict']}" if "c3" in A else ""),
              flush=True)
    R["holm_descriptive"] = RP.holm({arm: R["arms"][arm]["t1"]["p_lrt"] for arm in arms})
    if {"cnn10", "cnn10_scratch"} <= set(arms):
        fa = fold_auc_table(FT["cnn10"]["per_rep"], fold_of, labels)
        fb = fold_auc_table(FT["cnn10_scratch"]["per_rep"], fold_of, labels)
        d = (fa - fb.reindex(fa.index)).to_numpy()
        t, p = RP.nadeau_bengio(d, ratio)
        R["rq2"] = {"mean": float(d.mean()), "sd": float(d.std(ddof=1)), "nb_t": t, "nb_p": p, "n_pretrained_better": int((d > 0).sum())}
    R["trivial_loss"] = "sınıf-dengeli BCE'de sabit tahminci ≈ 0.23 (eğitim kümesi; CAL-001: iç doğrulamada 0.237)"
    R["elapsed_s"] = round(time.time() - t0, 1)
    md = finetune_md(R)
    js = json.dumps(R, indent=1, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    assert E.ids_free(md + js, C.index), "raporda katılımcı ID'si var (D-014)"
    a.report_dir.mkdir(parents=True, exist_ok=True)
    (a.report_dir / "EXP-030_finetune.md").write_text(md)
    (a.report_dir / "EXP-030_finetune.json").write_text(js)
    if a.out_dir:
        a.out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(scores).assign(label=Ct.label).to_csv(a.out_dir / "exp030_participant_scores.csv")
    print(md)


def finetune_md(R: dict) -> str:
    f3, fp = E.f3, E.fp
    L = ["# EXP-030 — Faz 3 uçtan uca fine-tune: değerlendirme (otomatik üretildi; katılımcı ID'si içermez)", "",
         f"Kohort {R['cohort']['n_t1']} ({R['cohort']['n_t1_asthma']} / {R['cohort']['n_t1_healthy']}); {R['n_repeats']} tekrar × 5 fold, "
         f"gün-gruplu (splits_day). Tanımlar PHASE3_DESIGN 5.2 (sonuçlardan önce). Bootstrap B {R['n_boot']}; permütasyon {R['n_perms']}. "
         "C3 keşifsel; Holm, ΔE1 ve O1 betimsel.", "",
         "## E1 ve T1", "",
         "| kol | skor | E1 fold AUC ort ± SD [min–maks] | ŝ havuzlanmış AUC [%95] | T1 PLR p | OR / SD [bootstrap %95] | CV3 p | dengeli doğr. (eşik 0) |",
         "|---|---|---|---|---|---|---|---|"]
    for arm, A in R["arms"].items():
        e, t, bo = A["e1"], A["t1"], A["boot_or"]
        L.append(f"| {arm} | fine-tune | {e['fold_auc_mean']:.3f} ± {e['fold_auc_sd']:.3f} [{e['fold_auc_min']:.2f}–{e['fold_auc_max']:.2f}] | "
                 f"{e['pooled_auc_shat']:.3f} [{e['pooled_auc_shat_ci'][0]:.3f}, {e['pooled_auc_shat_ci'][1]:.3f}] | {fp(t['p_lrt'])} | "
                 f"{f3(t['or_sd'])} [{f3(bo['lo'])}, {f3(bo['hi'])}] | {fp(t['p_cv3'])} | {e['balanced_acc_thr0_mean']:.3f} |")
        if "frozen" in A:
            z = A["frozen"]
            L.append(f"| | dondurulmuş eş | {z['e1']['fold_auc_mean']:.3f} ± {z['e1']['fold_auc_sd']:.3f} | {z['e1']['pooled_auc_shat']:.3f} | "
                     f"{fp(z['t1']['p_lrt'])} | {f3(z['t1']['or_sd'])} | {fp(z['t1']['p_cv3'])} | |")
    L += ["", "## C3 — fine-tune, dondurulmuş eşe göre bağlamın ötesinde daha fazla bilgi taşıyor mu? (keşifsel)", "",
          "| kol | LR fine-tune | LR dondurulmuş | fark [%95 bootstrap] | okuma | ΔE1 FT − dondurulmuş (fold-eşli; NB p) | FT daha iyi fold | ŝ korelasyonu |",
          "|---|---|---|---|---|---|---|---|"]
    for arm, A in R["arms"].items():
        if "c3" not in A:
            L.append(f"| {arm} | — | — | — | dondurulmuş eş yok | — | — | — |")
            continue
        c, d = A["c3"], A["delta_e1_vs_frozen"]
        L.append(f"| {arm} | {c['lr_ft']:.2f} | {c['lr_frozen']:.2f} | {c['diff']:+.2f} [{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}] | "
                 f"**{c['verdict']}** | {d['mean']:+.3f} ± {d['sd']:.3f} (p {fp(d['nb_p'])}) | {d['n_ft_better']} / {d['n_folds']} | "
                 f"{A['frozen']['corr_with_ft']:.2f} |")
    L += ["", "## C2 — bağlam kullanımı (etiket-içi permütasyon kısmi testi; L5)", "",
          "| kol | skor | kısmi R² (p) | hastalarda PM katsayısı [%95] | sağlıklılarda AM − PM [%95] |", "|---|---|---|---|---|"]
    for arm, A in R["arms"].items():
        for lab, Z in [("fine-tune", A)] + ([("dondurulmuş eş", A["frozen"])] if "frozen" in A else []):
            c2, l5 = Z["c2_partial"], Z["l5"]
            pm, hp = l5["patients_pm_coef"], l5["healthy_am_minus_pm"]
            L.append(f"| {arm} | {lab} | {c2['partial_r2']:.3f} ({fp(c2['p'])}) | {pm['coef']:+.2f} [{pm['ci'][0]:+.2f}, {pm['ci'][1]:+.2f}] | "
                     f"{hp['diff']:+.2f} [{hp['ci'][0]:+.2f}, {hp['ci'][1]:+.2f}] (n {hp['n_am']} / {hp['n_pm']}) |")
    if "rq2" in R:
        q = R["rq2"]
        L += ["", "## RQ2 — ön-eğitim (cnn10 − cnn10_scratch)", "",
              f"Fold-eşli ΔAUC {q['mean']:+.3f} ± {q['sd']:.3f}; Nadeau–Bengio t {q['nb_t']:.2f}, p {fp(q['nb_p'])}; "
              f"önceden eğitilmiş daha iyi: {q['n_pretrained_better']} / 15 fold. T1 iki kol için yukarıdaki tabloda."]
    L += ["", "## O1 — aşırı uyum ve kalibrasyon (betimsel; post hoc kullanıcı sorusu)", "",
          "Test kalibrasyon eğimi: y ~ ŝ lojistik regresyonunda ŝ'nin katsayısı. 1'den küçükse skorlar fazla uç (aşırı güven).",
          "Sabit tahminci eğitim kaybı ≈ 0.23.", "",
          "| kol | epoch | son eğitim kaybı (ort ± SD) | atlanan adım | FT eğim [%95] | FT ort. abs(ŝ) | dondurulmuş eğim | dondurulmuş ort. abs(ŝ) |",
          "|---|---|---|---|---|---|---|---|"]
    for arm, A in R["arms"].items():
        tr, o = A["train"], A["o1"]
        zo = A["frozen"]["o1"] if "frozen" in A else None
        L.append(f"| {arm} | {tr['epochs']['mean']:.1f} | {tr['final_train_loss']['mean']:.3f} ± {tr['final_train_loss']['std']:.3f} | "
                 f"{int(tr['skipped_steps']['sum'])} | {o['slope']:.2f} [{o['ci'][0]:.2f}, {o['ci'][1]:.2f}] | {o['mean_abs_logit']:.2f} | "
                 + (f"{zo['slope']:.2f} | {zo['mean_abs_logit']:.2f} |" if zo else "— | — |"))
    L += ["", f"Holm (betimsel, 4 kol): " + ", ".join(f"{k} {fp(v)}" for k, v in R["holm_descriptive"].items()),
          "", "- E1 artışı tek başına başarı sayılmaz (C3). Bütün kollar ve olumsuz sonuçlar raporlanır.",
          "- ssl kolu fold başına D-035 seçimidir: " + str(R["arms"].get("ssl", {}).get("backbones_used", {})) + "."]
    return "\n".join(L) + "\n"


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
    q = sub.add_parser("finetune")
    q.add_argument("--ft-root", required=True, type=Path, help="Drive: experiments/phase3/finetune")
    q.add_argument("--probe-dir", required=True, type=Path, help="EXP-016DG: experiments/phase3/frozen_probe_day")
    q.add_argument("--selection", type=Path, help="d035_selection.csv (ssl kolunun dondurulmuş eşi)")
    q.add_argument("--splits-dir", required=True, type=Path)
    q.add_argument("--context", required=True, type=Path)
    q.add_argument("--report-dir", required=True, type=Path)
    q.add_argument("--out-dir", type=Path)
    q.add_argument("--arms", default=",".join(FT_ARMS))
    q.add_argument("--frozen-backbones", default="cnn10,cnn14,beats,wavlm_base_plus,wavlm_large")
    q.add_argument("--n-repeats", type=int, default=3)
    q.add_argument("--n-boot", type=int, default=2000)
    q.add_argument("--n-boot-c3", type=int, default=2000)
    q.add_argument("--n-perms", type=int, default=1000)
    q.add_argument("--n-jobs", type=int, default=-1)
    q.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cmd_frozen_dg(a) if a.cmd == "frozen-dg" else cmd_finetune(a)


if __name__ == "__main__":
    main()
