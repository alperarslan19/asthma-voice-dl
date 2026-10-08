"""
Faz 0 / EXP-003 — Kayıt bağlamı analizi: kodlama zinciri, günün saati, tarih, tasarım dengesi

BİLİMSEL AMAÇ
    Ses denetimi (AUD-001) iki şey gösterdi: (1) dosyalar iki farklı kodlama zincirinden geliyor,
    (2) hastalar ve sağlıklılar günün farklı saatlerinde kaydedilmiş. Bu script:
      A. Zincirleri karakterize eder (bitrate, bant genişliği, süre; katılımcı içi karışım;
         slot / deneme / ay ile ilişki) ve D-021'deki alçak geçiren kesim değerini veriden seçer.
      B. Katılımcı düzeyinde "kayıt bağlamı"nı çıkarır: kayıt tarihi (dosya zaman damgası öncelikli,
         yoksa CSV), seans başlangıç saati.
      C. EXP-003: SESİ KULLANMADAN bağlamın etiketi ne kadar tahmin ettiği (saat, tarih, ikisi, yaş).
      D. Değerlendirme tasarımlarının (E1, E1h, E2, E2h, E3, E3h) DENGE tablosu: her tasarımda
         saat / tarih / yaş hâlâ etiketi ne kadar ayırıyor (tabakalı AUC; 0.5 = denge) ve tasarımın
         kesinliği (gerçek AUC 0.75 varsayımıyla simülasyon).

      E. EXP-004: Günün saatinin KABA akustik izi var mı? Yalnız aynı etiket içinde (önce hastalar):
         denetim ölçümleri (gürültü tabanı, seviye, SNR, bant genişliği, süre, sessizlikler, bitrate,
         zincir) sabah ve öğleden sonra kayıtlarını ayırıyor mu? Etiket sabit olduğu için ayrım = saatin
         sese izi. (Derin gömmelerdeki ince izler Faz 2 probunda test edilir.)
         E-b. EXP-004b: Sabah ve öğleden sonra hastaları klinik olarak farklı mı? (Etiket içi saat
         kontrastının "yalnız bağlam" olarak okunabilmesi için ön koşul.)
      F. Her tasarımın "yalnız bağlam" referansı: bağlam-only modelin OOF skoru o tasarımın
         tabakaları içinde hangi AUC'yi alıyor? Bir ses modelinin o tasarımdaki sonucu 0.5 ile değil
         bu sayıyla karşılaştırılır.

GİRDİLER (Drive data_derived/): participants.csv, audio_inventory.csv
ÇIKTILAR
    <out_private>/participant_context.csv   katılımcı düzeyi (Drive'da kalır, git'e GİRMEZ)
    <out_report>/recording_context.{md,json} yalnız agrega; katılımcı ID'si İÇERMEZ (git'e girebilir)

ÇALIŞTIRMA
    python scripts/analyze_recording_context.py --participants .../participants.csv \
        --inventory .../audio_inventory.csv --out-private .../data_derived --out-report reports/recording_context
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LOCAL_TZ = "Europe/Istanbul"


def chain_of(encoder: pd.Series) -> pd.Series:
    return pd.Series(np.where(encoder.astype(str).str.startswith("Lavf"), "ffmpeg", "apple"), index=encoder.index)


def stratified_auc(df: pd.DataFrame, key: list[str], x: str) -> tuple[float, int]:
    """Yalnız aynı tabakadaki (key) astım–sağlıklı çiftlerini karşılaştıran AUC. 0.5 = x tabakalar içinde dengeli."""
    num = den = 0.0
    for _, g in df.dropna(subset=[x]).groupby(key):
        a, b = g.loc[g["label"] == 1, x].to_numpy(), g.loc[g["label"] == 0, x].to_numpy()
        if len(a) and len(b):
            num += (a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()
            den += len(a) * len(b)
    return (num / den if den else float("nan")), int(den)


def design_precision(groups: list[tuple[int, int]], true_auc: float = 0.75, n_sim: int = 2000, seed: int = 0) -> dict:
    """Tabakalı tasarımın kesinliği: gerçek AUC true_auc iken tahminin %95 aralığı (binormal simülasyon)."""
    rng = np.random.default_rng(seed)
    delta = np.sqrt(2) * norm.ppf(true_auc)
    sims = []
    for _ in range(n_sim):
        num = den = 0
        for n1, n0 in groups:
            a, b = rng.normal(delta, 1, n1), rng.normal(0, 1, n0)
            num += (a[:, None] > b[None, :]).sum()
            den += n1 * n0
        sims.append(num / den)
    sims = np.array(sims)
    return {"halfwidth_95": float((np.percentile(sims, 97.5) - np.percentile(sims, 2.5)) / 2),
            "p_above_0_5": float((sims > 0.5).mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--participants", required=True, type=Path)
    ap.add_argument("--inventory", required=True, type=Path)
    ap.add_argument("--out-private", required=True, type=Path)
    ap.add_argument("--out-report", required=True, type=Path)
    args = ap.parse_args()
    args.out_private.mkdir(parents=True, exist_ok=True)
    args.out_report.mkdir(parents=True, exist_ok=True)

    P = pd.read_csv(args.participants)
    inv = pd.read_csv(args.inventory)
    assert P["participant_id"].is_unique
    assert inv["participant_id"].notna().all() and inv["slot"].between(1, 7).all(), "envanter eşleşmemiş dosya içeriyor"
    inv["chain"] = chain_of(inv["tag_encoder"])
    R: dict = {"n_files": int(len(inv)), "n_participants_with_audio": int(inv["participant_id"].nunique())}

    # ---------------- A) Kodlama zincirleri ----------------
    A = {"files_per_chain": inv["chain"].value_counts().to_dict()}
    per_pid = inv.groupby("participant_id")["chain"].agg(lambda s: "+".join(sorted(set(s))))
    A["participants_by_chain_mix"] = per_pid.value_counts().to_dict()
    A["by_chain"] = {}
    for c, g in inv.groupby("chain"):
        A["by_chain"][c] = {k: g[k].describe(percentiles=[.01, .5, .99]).round(3).to_dict()
                            for k in ["audio_bit_rate", "bandwidth_khz", "duration_s", "trail_silence_s", "noise_floor_db"]}
        A["by_chain"][c]["sample_rates"] = g["sr"].value_counts().rename(lambda v: str(int(v))).to_dict()
    A["ffmpeg_share_by_slot"] = inv.groupby("slot")["chain"].apply(lambda s: round((s == "ffmpeg").mean(), 3)).to_dict()
    take = P.melt(id_vars="participant_id", value_vars=[f"slot{i}_take" for i in range(1, 8)], var_name="slot", value_name="k")
    take["slot"] = take["slot"].str.extract(r"(\d)")[0].astype(int)
    m = inv.merge(take, on=["participant_id", "slot"], how="left")
    ct = pd.crosstab(m["k"] > 0, m["chain"])  # DİKKAT: m.take değil (DataFrame.take metottur)
    A["chain_vs_retake_fisher_p"] = float(stats.fisher_exact(ct.values)[1]) if ct.shape == (2, 2) else None
    # D-021: alçak geçiren kesim = en düşük kodlayıcı kesiminin altı (0.25 kHz aşağı yuvarlanır)
    min_bw = float(inv["bandwidth_khz"].min())
    A["min_bandwidth_khz"] = min_bw
    A["min_bandwidth_khz_by_chain"] = inv.groupby("chain")["bandwidth_khz"].min().round(2).to_dict()
    A["recommended_lowpass_khz"] = float(np.floor((min_bw - 0.25) * 4) / 4)
    R["encoding_chains"] = A

    # ---------------- B) Katılımcı düzeyinde kayıt bağlamı ----------------
    inv["ts"] = pd.to_datetime(inv["creation_local"], utc=True).dt.tz_convert(LOCAL_TZ)
    s = inv.dropna(subset=["ts"]).groupby("participant_id")["ts"].agg(["min", "max", "count"])
    ctx = pd.DataFrame({
        "participant_id": s.index,
        "start_hour": (s["min"].dt.hour + s["min"].dt.minute / 60).to_numpy(),
        "session_minutes": ((s["max"] - s["min"]).dt.total_seconds() / 60).to_numpy(),
        "n_timestamped_files": s["count"].to_numpy(),
        "file_date": s["min"].dt.date.astype(str).to_numpy(),
    })
    ff = inv.groupby("participant_id")["chain"].apply(lambda c: (c == "ffmpeg").mean()).rename("ffmpeg_file_share")
    d = P.merge(ctx, on="participant_id", how="left").merge(ff, on="participant_id", how="left")
    d["has_audio"] = d["participant_id"].isin(inv["participant_id"])
    # D-024: kayıt tarihi = dosya zaman damgası (ses kaydının gerçek anı); yoksa CSV tarihi
    d["recording_date"] = d["file_date"].fillna(d["collection_day"])
    d["recording_date_source"] = np.where(d["file_date"].notna(), "file", np.where(d["collection_day"].notna(), "csv", "none"))
    mism = d[d["file_date"].notna() & d["collection_day"].notna() & (d["file_date"] != d["collection_day"])]
    delta = (pd.to_datetime(mism["file_date"]) - pd.to_datetime(mism["collection_day"])).dt.days
    R["date_checks"] = {
        "n_participants_file_vs_csv_date_differ": int(len(mism)),
        "their_labels": mism["label"].value_counts().rename({1: "asthma", 0: "healthy"}).to_dict(),
        "delta_days": sorted(delta.astype(int).tolist()),
        "n_csv_date_missing_filled_from_file": int((d["collection_day"].isna() & d["file_date"].notna() & d["has_audio"]).sum()),
        "n_with_audio_but_no_date": int((d["has_audio"] & d["recording_date"].isna()).sum()),
    }
    d["days"] = (pd.to_datetime(d["recording_date"]) - pd.Timestamp("2024-01-01")).dt.days
    d.to_csv(args.out_private / "participant_context.csv", index=False)

    # ---------------- C) EXP-003: bağlam-only baseline'lar ----------------
    a = d[d["has_audio"] & d["in_paper_cohort"]].copy()
    a["hour2"] = a["start_hour"] ** 2
    hour = {}
    for lab, g in a.dropna(subset=["start_hour"]).groupby("label"):
        h = g["start_hour"]
        hour["asthma" if lab == 1 else "healthy"] = {
            "n": int(len(h)), "median": float(h.median()), "q25": float(h.quantile(.25)), "q75": float(h.quantile(.75)),
            "frac_before_noon": float((h < 12).mean()), "min": float(h.min()), "max": float(h.max())}
    R["start_hour_by_label"] = hour
    R["n_with_start_hour"] = int(a["start_hour"].notna().sum())
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=0)
    feature_sets = {"hour": ["start_hour", "hour2"], "date": ["days"], "hour+date": ["start_hour", "hour2", "days"],
                    "age": ["age"], "age+hour": ["age", "start_hour", "hour2"], "age+hour+date": ["age", "start_hour", "hour2", "days"]}
    cohorts = {"audio_cohort": a, "time_window": a[a["in_time_overlap"]],
               "afternoon_only": a[a["start_hour"] >= 12]}
    exp003 = {}
    for cn, cdf in cohorts.items():
        for fn, feats in feature_sets.items():
            sub = cdf.dropna(subset=feats)
            if sub["label"].nunique() < 2 or sub["label"].value_counts().min() < 10:
                continue
            r = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced")),
                                sub[feats], sub["label"], cv=cv, scoring="roc_auc")
            exp003[f"{cn} | {fn}"] = {"n": int(len(sub)), "n_asthma": int(sub["label"].sum()),
                                      "n_healthy": int((sub["label"] == 0).sum()),
                                      "auc_mean": float(r.mean()), "auc_sd": float(r.std(ddof=1)),
                                      "auc_p2.5": float(np.percentile(r, 2.5)), "auc_p97.5": float(np.percentile(r, 97.5))}
    (args.out_report / "EXP-003_context_baselines.json").write_text(json.dumps(exp003, indent=2, ensure_ascii=False))

    # ---------------- D) Tasarım denge tablosu ----------------
    a["hb"] = np.floor(a["start_hour"])
    a["_all"] = 0
    designs = {
        "E1  tam kohort": (a, ["_all"]),
        "E1h tam kohort + aynı 1 saat dilimi": (a.dropna(subset=["hb"]), ["hb"]),
        "E2  zaman penceresi": (a[a["in_time_overlap"]], ["_all"]),
        "E2h zaman penceresi + aynı 1 saat dilimi": (a[a["in_time_overlap"]].dropna(subset=["hb"]), ["hb"]),
        "E3  aynı gün": (a.dropna(subset=["recording_date"]), ["recording_date"]),
        "E3h aynı gün + aynı 1 saat dilimi": (a.dropna(subset=["recording_date", "hb"]), ["recording_date", "hb"]),
    }
    bal = []
    for name, (df, key) in designs.items():
        g = df.groupby(key)["label"].agg(lambda s: ((s == 1).sum(), (s == 0).sum()))
        groups = [(n1, n0) for n1, n0 in g if n1 and n0]
        row = {"design": name, "n_asthma": int(sum(x for x, _ in groups)), "n_healthy": int(sum(y for _, y in groups)),
               "pairs": int(sum(x * y for x, y in groups))}
        for x, lab in [("start_hour", "hour"), ("days", "date"), ("age", "age")]:
            row[f"{lab}_auc"] = round(stratified_auc(df, key, x)[0], 3)
        row.update({k: round(v, 3) for k, v in design_precision(groups).items()})
        bal.append(row)
    R["design_balance"] = bal

    # ---------------- E) EXP-004: saatin kaba akustik izi (etiket içinde) ----------------
    feats = ["noise_floor_db", "speech_level_db", "snr_proxy_db", "rms_dbfs", "bandwidth_khz", "duration_s",
             "lead_silence_s", "trail_silence_s", "active_frac", "audio_bit_rate"]
    inv["chain_ffmpeg"] = (inv["chain"] == "ffmpeg").astype(float)
    agg = inv.groupby("participant_id")[feats + ["chain_ffmpeg"]].mean().reset_index()
    e = agg.merge(a[["participant_id", "label", "start_hour"]], on="participant_id").dropna(subset=["start_hour"])
    e["pm"] = (e["start_hour"] >= 12).astype(int)
    rng = np.random.default_rng(0)
    exp004 = {}
    for lab, name in [(1, "asthma"), (0, "healthy")]:
        g = e[e["label"] == lab]
        res = {"n": int(len(g)), "n_afternoon": int(g["pm"].sum()), "single_feature_auc": {}}
        for f in feats + ["chain_ffmpeg"]:
            auc = float(roc_auc_score(g["pm"], g[f]))
            null = [roc_auc_score(rng.permutation(g["pm"].to_numpy()), g[f]) for _ in range(500)]
            lo, hi = np.percentile(null, [0.3, 99.7])  # ~11 özellik için kaba Bonferroni
            res["single_feature_auc"][f] = {"auc": round(auc, 3), "beyond_chance": bool(auc < lo or auc > hi)}
        r = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)),
                            g[feats + ["chain_ffmpeg"]], g["pm"], cv=cv, scoring="roc_auc")
        res["all_features_cv_auc"] = {"mean": float(r.mean()), "sd": float(r.std(ddof=1))}
        exp004[name] = res
    R["exp004_time_of_day_acoustic_signature"] = exp004

    # E-b) EXP-004b: hastalarda sabah/öğleden sonra grupları KLİNİK olarak farklı mı?
    # Neden: "etiket içinde skor ↔ saat" ilişkisi (Spisak kısmi testi, N5 bağlam-vekili kontrolü) ancak
    # sabah ve öğleden sonra hastaları klinik olarak benzerse "kayıt bağlamı" diye okunabilir.
    pa = a[(a["label"] == 1)].dropna(subset=["start_hour"]).copy()
    pa["pm"] = (pa["start_hour"] >= 12).astype(int)
    cat = ["asthma_step", "sft_dx", "smoking", "sex", "gina_control", "ics_use", "allergic_rhinitis"]
    num = ["age", "act_total", "fev1_pct", "fvc_pct", "fev1_fvc"]
    uni = {}
    for c in cat:
        ct = pd.crosstab(pa[c].astype(str), pa["pm"])
        uni[c] = float(stats.chi2_contingency(ct.values)[1]) if ct.shape[0] > 1 else None
    for c in num:
        x = pa.dropna(subset=[c])
        uni[c] = float(stats.mannwhitneyu(x.loc[x["pm"] == 1, c], x.loc[x["pm"] == 0, c]).pvalue)
    ct = pd.crosstab(pa["period"].astype(str), pa["pm"])
    uni["period(early/late)"] = float(stats.chi2_contingency(ct.values)[1])
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    pc = pa.dropna(subset=num).copy()
    pc[cat] = pc[cat].astype(str)
    clin_model = make_pipeline(ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), cat), ("n", StandardScaler(), num)]),
                               LogisticRegression(max_iter=3000, class_weight="balanced"))
    r = cross_val_score(clin_model, pc[cat + num], pc["pm"], cv=cv, scoring="roc_auc")
    R["exp004b_clinical_profile_am_vs_pm_patients"] = {
        "n": int(len(pc)), "n_pm": int(pc["pm"].sum()), "univariate_p": {k: (round(v, 4) if v is not None else None) for k, v in uni.items()},
        "clinical_cv_auc": {"mean": float(r.mean()), "sd": float(r.std(ddof=1)),
                            "p2.5": float(np.percentile(r, 2.5)), "p97.5": float(np.percentile(r, 97.5))}}

    # ---------------- F) Her tasarımın "yalnız bağlam" referansı ----------------
    # Bir ses modeli bağlamı (saat/tarih/yaş) mükemmel kodlasa, sesten HİÇBİR astım bilgisi almadan
    # her tasarımda hangi tabakalı AUC'yi alırdı? Bağlam-only modelin OOF skoru tasarımın tabakaları
    # içinde değerlendirilir. Bu sayı, o tasarımın gerçek "şans çizgisi"dir (0.5 değil).
    ref = {}
    for fn in ["age", "hour+date", "age+hour+date"]:
        feats = feature_sets[fn]
        sub = a.dropna(subset=feats).copy()
        per_rep = {name: [] for name in designs}
        rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=0)
        oof = np.full((20, len(sub)), np.nan)
        for i, (tr, te) in enumerate(rcv.split(sub, sub["label"])):
            mdl = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced"))
            mdl.fit(sub.iloc[tr][feats], sub.iloc[tr]["label"])
            oof[i // 5, te] = mdl.predict_proba(sub.iloc[te][feats])[:, 1]
        assert not np.isnan(oof).any(), "her tekrarda her katılımcı tam bir kez test edilmeli"
        for k in range(20):
            sub["_score"] = oof[k]
            for name, (df, key) in designs.items():
                dd = sub[sub["participant_id"].isin(df["participant_id"])].dropna(subset=key)
                per_rep[name].append(stratified_auc(dd, key, "_score")[0])
        ref[fn] = {name: {"mean": round(float(np.mean(v)), 3), "sd": round(float(np.std(v, ddof=1)), 3)} for name, v in per_rep.items()}
    R["context_only_reference_by_design"] = ref

    (args.out_report / "recording_context.json").write_text(json.dumps(R, indent=2, ensure_ascii=False, default=str))
    L = ["# Kayıt bağlamı analizi (EXP-003) — otomatik üretildi, katılımcı ID'si içermez", "",
         "## A. Kodlama zincirleri", "",
         f"- Dosya: {A['files_per_chain']} · katılımcı içi karışım: {A['participants_by_chain_mix']}",
         f"- FFmpeg payı slota göre: {A['ffmpeg_share_by_slot']} · zincir ↔ yeniden deneme Fisher p = {A['chain_vs_retake_fisher_p']:.3g}",
         f"- En düşük bant genişliği: {A['min_bandwidth_khz_by_chain']} kHz → önerilen alçak geçiren kesim: **{A['recommended_lowpass_khz']} kHz** (D-021)"]
    for c, v in A["by_chain"].items():
        L.append(f"- {c}: bitrate medyan {v['audio_bit_rate']['50%']:.0f}, bant genişliği medyan {v['bandwidth_khz']['50%']:.2f} kHz, "
                 f"süre medyan {v['duration_s']['50%']:.2f} s, SR {v['sample_rates']}")
    L += ["", "## B. Tarih ve saat", "", f"- Tarih kontrolleri: {R['date_checks']}", f"- Saati bilinen katılımcı: {R['n_with_start_hour']}"]
    for k, v in hour.items():
        L.append(f"- {k}: seans başlangıcı medyan {v['median']:.2f} (IQR {v['q25']:.2f}–{v['q75']:.2f}), öğleden önce %{100*v['frac_before_noon']:.0f}, aralık {v['min']:.1f}–{v['max']:.1f} (n={v['n']})")
    L += ["", "## C. EXP-003 — sesi kullanmayan bağlam baseline'ları (lojistik regresyon, 5-fold × 20)", "",
          "| kohort | özellik | n (astım/sağlıklı) | AUC ort ± SD | fold %2.5–97.5 |", "|---|---|---|---|---|"]
    for k, v in exp003.items():
        c, f = k.split(" | ")
        L.append(f"| {c} | {f} | {v['n']} ({v['n_asthma']}/{v['n_healthy']}) | {v['auc_mean']:.3f} ± {v['auc_sd']:.3f} | {v['auc_p2.5']:.2f}–{v['auc_p97.5']:.2f} |")
    L += ["", "## D. Değerlendirme tasarımlarının dengesi", "",
          "Her sütun: o tasarımın tabakaları İÇİNDE değişkenin etiketi ayırma gücü (tabakalı AUC). 0.5 = tam denge; 0.5'ten uzak = artık confounding.", "",
          "| tasarım | astım/sağlıklı | çift | saat AUC | tarih AUC | yaş AUC | ±%95 (gerçek AUC 0.75) | P(tahmin > 0.5) |", "|---|---|---|---|---|---|---|---|"]
    for r in bal:
        L.append(f"| {r['design']} | {r['n_asthma']}/{r['n_healthy']} | {r['pairs']} | {r['hour_auc']} | {r['date_auc']} | {r['age_auc']} | {r['halfwidth_95']} | {r['p_above_0_5']} |")
    L += ["", "## E. EXP-004 — Saatin kaba akustik izi (yalnız aynı etiket içinde; sabah vs öğleden sonra)", ""]
    for name, res in exp004.items():
        sig = [f for f, v in res["single_feature_auc"].items() if v["beyond_chance"]]
        L.append(f"- {name} (n={res['n']}, öğleden sonra {res['n_afternoon']}): tüm ölçümler birlikte CV AUC "
                 f"{res['all_features_cv_auc']['mean']:.3f} ± {res['all_features_cv_auc']['sd']:.3f}; "
                 f"şanstan ayrılan tekil ölçüm: {sig if sig else 'yok'}")
    eb = R["exp004b_clinical_profile_am_vs_pm_patients"]
    small_p = {k: v for k, v in eb["univariate_p"].items() if v is not None and v < 0.05}
    L += ["", "### E-b. EXP-004b — Hastalarda sabah ve öğleden sonra grupları klinik olarak farklı mı?", "",
          f"- n = {eb['n']} hasta (öğleden sonra {eb['n_pm']}); klinik profil → öğleden sonra CV AUC "
          f"{eb['clinical_cv_auc']['mean']:.3f} ± {eb['clinical_cv_auc']['sd']:.3f} "
          f"(fold %2.5–97.5: {eb['clinical_cv_auc']['p2.5']:.2f}–{eb['clinical_cv_auc']['p97.5']:.2f})",
          f"- Tek değişkenli p < 0.05 (düzeltmesiz): {small_p if small_p else 'yok'}",
          f"- Tüm tek değişkenli p'ler: {eb['univariate_p']}"]
    L += ["", "## F. Her tasarımın 'yalnız bağlam' referansı (gerçek şans çizgisi)", "",
          "Bağlam-only lojistik modelin OOF skoru (5-fold × 20) her tasarımın tabakaları içinde değerlendirildi. "
          "Bağlamı mükemmel kodlayan ama hiç astım bilgisi taşımayan bir ses modeli bu AUC'leri alırdı; "
          "bir ses modelinin o tasarımdaki sonucu 0.5 ile değil bu sayıyla karşılaştırılmalı.", "",
          "| tasarım | " + " | ".join(ref.keys()) + " |", "|---|" + "---|" * len(ref)]
    for name in designs:
        L.append(f"| {name} | " + " | ".join(f"{ref[fn][name]['mean']:.3f} ± {ref[fn][name]['sd']:.3f}" for fn in ref) + " |")
    (args.out_report / "recording_context.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
