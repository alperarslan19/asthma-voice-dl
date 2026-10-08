"""
Faz 0 / Adım 2 — Klinik veri denetimi (clinical data audit)

BİLİMSEL AMAÇ
    Ses verisine dokunmadan önce, katılımcı tablosunun ne söylediğini sayılarla
    sabitlemek: kaç kişi, hangi gruplar, hangi değişkenler kimde var, ve en
    önemlisi: etiketi (astım / sağlıklı) SESİ HİÇ KULLANMADAN ne kadar tahmin
    edebiliyoruz? (confounder baseline'ları = EXP-001)

NE ÜRETİR
    1) <out_private>/participants.csv
         Katılımcı başına 1 satır. Split üretimi ve alt grup analizleri için
         ANA TABLO. Sağlık verisi içerir -> Google Drive'da kalır, GIT'E GİRMEZ.
    2) <out_report>/clinical_audit.md  ve  clinical_audit.json
         Yalnızca agrega sayılar. Git'e girebilir.
    3) <out_report>/EXP-001_confounder_baselines.json
         Sesi kullanmayan baseline sonuçları (deney kaydı).
    4) <out_report>/EXP-002_clinical_period_probe.json
         Yalnız hastalarda: klinik profil kayıt dönemini (erken/geç) ne kadar tahmin ediyor?
         Ses tabanlı "dönem probu"nun (D-017 N1, D-020) aşması gereken referans.

ÇALIŞTIRMA
    python scripts/audit_clinical.py \
        --csv  /content/drive/MyDrive/asthma-voice/data_raw/clinical_data.csv \
        --xlsx "/content/drive/MyDrive/asthma-voice/data_raw/astim-tarama_Data Report_20260314.xlsx" \
        --out-private /content/drive/MyDrive/asthma-voice/data_derived \
        --out-report  reports/clinical_audit

Tasarım notu: Bu script hiçbir şeyi "düzeltmez". Bulduğu tutarsızlıkları
raporlar; hangi katılımcının hangi analize gireceği kararı DECISIONS.md'de
alınır ve participants.csv'deki bayrak (flag) sütunlarıyla uygulanır.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Sabitler: CSV'deki Türkçe sütun adları (tek bir yerde tanımlı)
# ---------------------------------------------------------------------------
COL_ID = "Hasta"
COL_GROUP = "Gönüllü kategorisi"
REC_COLS = [f"Gönüllünün {i}. ses kaydı" for i in range(1, 8)]
COL_DATE = "Veri  toplama tarihi"  # DİKKAT: kaynakta iki boşluk var
COL_DOB = "Doğum Tarihi"
COL_AGE = "Yaş"
COL_SEX = "Cinsiyet"
COL_SMOKE = "Sigara Kullanımı"
PATIENT_SCREEN = "DahilEtme-1"      # hastalar için dahil etme kriteri
HEALTHY_SCREEN = "DahilEtme-1_1"    # sağlıklı gönüllüler için dahil etme kriteri

GROUP_MAP = {"hasta": 1, "sağlıklı gönüllü": 0}
PAPER_COHORT_MAX_ID = 101344  # Alagöz ve ark. 344 kişi = 101001..101344 (bkz. rapor)

# Yalnızca hastalarda tanımlı, alt grup analizi için tutulan değişkenler
PATIENT_ONLY = {
    "GINA": "gina_control",
    "Bir üst sorudaki semptom sayısına göre aşağıda uygun olan seçeneği işaretleyiniz": "gina_symptom_level",
    "Astım Basamak": "asthma_step",
    "Solunum Fonksiyon testi (SFT) Tanı": "sft_dx",
    "FEV1(Yüzde)": "fev1_pct",
    "FVC(Yüzde)": "fvc_pct",
    "FEV1/FVC (Yüzde)": "fev1_fvc",
    "Hasta Toplam Puanı": "act_total",
    "İnhaler steroid kullanıyor mu?": "ics_use",
    "Alerjik Rinit": "allergic_rhinitis",
}


NUMERIC_PATIENT_COLS = ["fev1_pct", "fvc_pct", "fev1_fvc", "act_total", "asthma_step"]


def to_num(s: pd.Series) -> pd.Series:
    """Türkçe ondalık virgülü ('91,8') noktaya çevirip sayıya dönüştür.
    Not: spirometri sütunlarında değerlerin ~%98'i virgüllü yazılmış; düz to_numeric bunları sessizce NaN yapıyordu."""
    out = pd.to_numeric(s.astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce")
    bad = s.notna() & out.isna()
    assert bad.sum() == 0, f"Sayıya çevrilemeyen değerler: {s[bad].head().tolist()}"
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    # Boş string -> NaN, kenar boşluklarını temizle
    df = df.apply(lambda c: c.str.strip()).replace({"": np.nan})
    # --- Yapısal kontroller (assert = "bu doğru değilse DUR") ---
    for c in [COL_ID, COL_GROUP, *REC_COLS, COL_DATE, COL_AGE, COL_SEX, COL_SMOKE]:
        assert c in df.columns, f"Beklenen sütun yok: {c!r}"
    assert df[COL_ID].notna().all(), "Boş participant ID var"
    assert df[COL_ID].is_unique, "Tekrarlanan participant ID var"
    assert df[COL_ID].str.fullmatch(r"\d{6}").all(), "ID'ler 6 haneli sayı değil"
    unknown = set(df[COL_GROUP].dropna()) - set(GROUP_MAP)
    assert not unknown, f"Beklenmeyen grup etiketi: {unknown}"
    assert df[COL_GROUP].notna().all(), "Grubu boş katılımcı var"
    return df


def compare_with_xlsx(df_csv: pd.DataFrame, xlsx: Path) -> dict:
    """CSV ile XLSX aynı veri mi? (XLSX'te 3 başlık satırı var: 2 grup başlığı + sütun adları)"""
    xl = pd.read_excel(xlsx, header=2, dtype=str)
    xl = xl.apply(lambda c: c.astype(str).str.strip()).replace({"nan": np.nan, "NaT": np.nan, "None": np.nan, "": np.nan})
    a = df_csv.astype(object).where(df_csv.notna(), np.nan).set_index(COL_ID)
    b = xl.set_index(COL_ID)
    out = {"same_shape": a.shape == b.shape, "same_columns": list(a.columns) == list(b.columns),
           "same_ids": set(a.index) == set(b.index)}
    n_diff_cells = None
    if out["same_columns"] and out["same_ids"]:
        b = b.loc[a.index]
        diff = ~((a == b) | (a.isna() & b.isna()))
        n_diff_cells = int(diff.values.sum())
    out["n_differing_cells"] = n_diff_cells
    out["identical"] = bool(out["same_shape"] and out["same_columns"] and out["same_ids"] and n_diff_cells == 0)
    return out


def build_participants(df: pd.DataFrame) -> pd.DataFrame:
    p = pd.DataFrame({"participant_id": df[COL_ID].astype(int)})
    p["group_raw"] = df[COL_GROUP].values
    p["label"] = df[COL_GROUP].map(GROUP_MAP).astype(int).values  # 1 = astım, 0 = sağlıklı
    p["collection_date"] = pd.to_datetime(df[COL_DATE], errors="coerce").values
    p["age"] = pd.to_numeric(df[COL_AGE], errors="coerce").values
    p["sex"] = df[COL_SEX].map({"kadın": "F", "erkek": "M"}).values
    p["smoking"] = df[COL_SMOKE].map({"hiç içmemiş": "never", "içiyor": "current",
                                      "içmiş bırakmış (en az 1 yıl)": "former"}).values
    # Ham değeri olup da eşleşmeyen kategori olmasın (sessiz NaN üretmeyelim)
    assert p["sex"].notna().sum() == df[COL_SEX].notna().sum(), "Tanınmayan cinsiyet kodu"
    assert p["smoking"].notna().sum() == df[COL_SMOKE].notna().sum(), "Tanınmayan sigara kodu"

    p["has_demographics"] = p[["age", "sex", "smoking"]].notna().all(axis=1)
    screen_col = np.where(p["label"] == 1, df[PATIENT_SCREEN], df[HEALTHY_SCREEN])
    p["screening_recorded"] = pd.Series(screen_col).notna().values
    p["in_paper_cohort"] = p["participant_id"] <= PAPER_COHORT_MAX_ID

    # Kayıt dönemi: son sağlıklı kayıt tarihinden sonra yalnızca hasta kaydedilmiş
    h = p[p.label == 0]["collection_date"]
    first_h, last_h = h.min(), h.max()
    p["period"] = np.select(
        [p.collection_date.isna(), p.collection_date <= last_h],
        ["unknown", "early"], default="late")
    p["in_time_overlap"] = (p.collection_date >= first_h) & (p.collection_date <= last_h)

    p["age_bin"] = pd.cut(p["age"], [0, 30, 45, 60, 200], labels=["<=30", "31-45", "46-60", ">60"]).astype(str)

    # Aynı gün kaydı: o gün karşı gruptan en az bir kişi de kaydedildi mi? (gün-içi karşılaştırma için)
    p["collection_day"] = p["collection_date"].dt.date.astype(str).replace("NaT", np.nan)
    days_h = set(p.loc[p.label == 0, "collection_day"].dropna())
    days_a = set(p.loc[p.label == 1, "collection_day"].dropna())
    p["same_day_as_other_group"] = np.where(p.label == 1, p.collection_day.isin(days_h), p.collection_day.isin(days_a))

    # Ses kaydı alanları: "<UUID>_<k>". UUID = veri toplama sistemindeki form alanı (her sütunda aynı),
    # k = muhtemelen kabul edilen denemenin sırası [HYPOTHESIS]. Dosyalar ise "<ID>_<slot>.<uzantı>"
    # olarak adlandırılmış (veri ekibi bilgisi) -> eşleme için bu alanlara GEREK YOK; k ikincil değişken.
    for i, c in enumerate(REC_COLS, start=1):
        parts = df[c].str.rsplit("_", n=1)
        p[f"slot{i}_field_uuid"] = parts.str[0].values
        p[f"slot{i}_take"] = pd.to_numeric(parts.str[1], errors="coerce").values
    take_cols = [f"slot{i}_take" for i in range(1, 8)]
    p["n_slots_retaken"] = (p[take_cols] > 0).sum(axis=1)
    p["sum_take_index"] = p[take_cols].sum(axis=1)

    for src, dst in PATIENT_ONLY.items():
        if src in df.columns:
            p[dst] = df[src].values
    for c in NUMERIC_PATIENT_COLS:
        if c in p:
            p[c] = to_num(p[c]).values
    return p.sort_values("participant_id").reset_index(drop=True)


def median_iqr(s: pd.Series) -> str:
    s = s.dropna()
    if len(s) == 0:
        return "NA"
    return f"{s.median():.1f} ({s.quantile(.25):.1f}–{s.quantile(.75):.1f}), n={len(s)}"


def hanley_mcneil_ci(auc: float, n_pos: int, n_neg: int) -> tuple[float, float]:
    """Tek bir AUC tahmininin yaklaşık %95 CI genişliği (Hanley & McNeil 1982)."""
    q1, q2 = auc / (2 - auc), 2 * auc**2 / (1 + auc)
    se = math.sqrt((auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2) + (n_neg - 1) * (q2 - auc**2)) / (n_pos * n_neg))
    return se, 1.96 * se


PERIOD_PROBE_CAT = ["asthma_step", "sft_dx", "smoking", "sex", "gina_control", "ics_use", "allergic_rhinitis"]
PERIOD_PROBE_NUM = ["age", "act_total", "fev1_pct", "fvc_pct", "fev1_fvc"]


def clinical_period_probe(p: pd.DataFrame, n_repeats: int = 20, seed: int = 0) -> dict:
    """EXP-002: Yalnız hastalarda (etiket sabit!) klinik profil -> kayıt dönemi (geç=1).
    Yüksek AUC = hasta profili dönemler arasında değişmiş. Sesten dönem tahmini bu sayıyla
    KARŞILAŞTIRILMALI: ses, klinik profilin ötesinde dönem bilgisi taşıyorsa kayıt koşulları değişmiştir."""
    d = p[p.in_paper_cohort & (p.label == 1) & p.period.isin(["early", "late"])].copy()
    d["y_late"] = (d.period == "late").astype(int)
    d[PERIOD_PROBE_CAT] = d[PERIOD_PROBE_CAT].astype(str)  # eksik değer kendi kategorisi ('nan')
    d = d.dropna(subset=PERIOD_PROBE_NUM)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=seed)

    def run(cat, num):
        tr = ([("c", OneHotEncoder(handle_unknown="ignore"), cat)] if cat else []) + \
             ([("n", StandardScaler(), num)] if num else [])
        model = make_pipeline(ColumnTransformer(tr), LogisticRegression(max_iter=3000, class_weight="balanced"))
        auc = cross_validate(model, d, d.y_late, cv=cv, scoring="roc_auc")["test_score"]
        return {"auc_mean": float(auc.mean()), "auc_sd": float(auc.std(ddof=1)),
                "auc_p2.5": float(np.percentile(auc, 2.5)), "auc_p97.5": float(np.percentile(auc, 97.5))}

    out = {"n_early": int((d.y_late == 0).sum()), "n_late": int((d.y_late == 1).sum()), "models": {}}
    out["models"]["all_clinical"] = run(PERIOD_PROBE_CAT, PERIOD_PROBE_NUM)
    out["models"]["asthma_step_only"] = run(["asthma_step"], [])
    out["models"]["sft_dx_only"] = run(["sft_dx"], [])
    out["models"]["smoking_only"] = run(["smoking"], [])
    out["models"]["measured_spirometry_only"] = run([], ["fev1_pct", "fvc_pct", "fev1_fvc"])
    out["models"]["all_clinical_without_step"] = run([c for c in PERIOD_PROBE_CAT if c != "asthma_step"], PERIOD_PROBE_NUM)
    # Tedavi basamağı içinde tabaka büyüklükleri: "aynı basamakta ses dönemi ayırıyor mu?" tasarımı için
    out["step_by_period"] = pd.crosstab(d.asthma_step, d.period).to_dict("index")
    return out


def confounder_baselines(p: pd.DataFrame, n_repeats: int = 20, seed: int = 0) -> dict:
    """EXP-001: Sesi hiç kullanmadan etiketi tahmin etmek. Hiperparametre seçimi yok
    -> nested CV gerekmez. Ölçekleme pipeline içinde, her fold'un train kısmına fit edilir."""
    d = p.copy()
    d["male"] = (d.sex == "M").astype(float).where(d.sex.notna())
    d["smoke_current"] = (d.smoking == "current").astype(float).where(d.smoking.notna())
    d["smoke_former"] = (d.smoking == "former").astype(float).where(d.smoking.notna())
    d["days_since_2024"] = (d.collection_date - pd.Timestamp("2024-01-01")).dt.days
    feature_sets = {
        "age": ["age"],
        "sex": ["male"],
        "smoking": ["smoke_current", "smoke_former"],
        "age+sex+smoking": ["age", "male", "smoke_current", "smoke_former"],
        "collection_date": ["days_since_2024"],
        # Meta veri sızıntısı kontrolü: ID numarası kayıt sırasıyla artıyorsa tarih vekilidir
        "participant_id_number": ["participant_id"],
    }
    cohorts = {
        "paper_cohort_344": d[d.in_paper_cohort],
        "time_overlap": d[d.in_paper_cohort & d.in_time_overlap],
    }
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=seed)
    out = {}
    for cname, cdf in cohorts.items():
        for fname, feats in feature_sets.items():
            sub = cdf.dropna(subset=feats)
            X, y = sub[feats].to_numpy(float), sub["label"].to_numpy(int)
            model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced"))
            r = cross_validate(model, X, y, cv=cv, scoring=["roc_auc", "balanced_accuracy"])
            auc, bac = r["test_roc_auc"], r["test_balanced_accuracy"]
            out[f"{cname} | {fname}"] = {
                "n_participants": int(len(sub)), "n_asthma": int(y.sum()), "n_healthy": int((y == 0).sum()),
                "auc_mean": float(auc.mean()), "auc_sd": float(auc.std(ddof=1)),
                "auc_p2.5": float(np.percentile(auc, 2.5)), "auc_p97.5": float(np.percentile(auc, 97.5)),
                "balacc_mean": float(bac.mean()), "balacc_sd": float(bac.std(ddof=1)),
                "n_folds_total": int(len(auc)),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--xlsx", type=Path, default=None)
    ap.add_argument("--out-private", required=True, type=Path)
    ap.add_argument("--out-report", required=True, type=Path)
    args = ap.parse_args()
    args.out_private.mkdir(parents=True, exist_ok=True)
    args.out_report.mkdir(parents=True, exist_ok=True)

    report: dict = {"inputs": {"clinical_csv_sha256": sha256(args.csv)}}
    df = load_csv(args.csv)
    report["inputs"]["n_rows"], report["inputs"]["n_cols"] = int(df.shape[0]), int(df.shape[1])
    if args.xlsx is not None:
        report["inputs"]["xlsx_sha256"] = sha256(args.xlsx)
        report["csv_vs_xlsx"] = compare_with_xlsx(df, args.xlsx)

    p = build_participants(df)
    # ---- Temel kontroller ----
    assert len(p) == len(df)
    take_cols = [f"slot{i}_take" for i in range(1, 8)]
    assert p[take_cols].notna().all().all(), "Bazı katılımcılarda ses kaydı alanı boş"
    for i in range(1, 8):
        assert p[f"slot{i}_field_uuid"].nunique() == 1, f"slot{i}: birden çok UUID var — varsayım bozuldu"

    R = report
    R["counts"] = {
        "all": p.label.value_counts().rename({1: "asthma", 0: "healthy"}).to_dict(),
        "paper_cohort_ids_le_101344": p[p.in_paper_cohort].label.value_counts().rename({1: "asthma", 0: "healthy"}).to_dict(),
        "ids_gt_101344": p[~p.in_paper_cohort][["participant_id", "label", "has_demographics"]].to_dict("records"),
        "healthy_without_any_demographics": p[(p.label == 0) & p[["age", "sex", "smoking", "collection_date"]].isna().all(axis=1)].participant_id.tolist(),
        "screening_not_recorded": p[~p.screening_recorded][["participant_id", "label"]].to_dict("records"),
        # Yalnız ID: git'e girebilecek raporda katılımcıya özgü klinik değer yazılmaz (D-014)
        "patients_outside_18_65": p[(p.label == 1) & ((p.age < 18) | (p.age > 65))].participant_id.tolist(),
    }
    R["recording_fields"] = {
        "slot_uuids": {f"slot{i}": p[f"slot{i}_field_uuid"].iloc[0] for i in range(1, 8)},
        "take_index_counts": {f"slot{i}": p[f"slot{i}_take"].value_counts().sort_index().to_dict() for i in range(1, 8)},
    }
    ct = pd.crosstab(p.label, p.n_slots_retaken > 0)
    R["retakes_by_group"] = {
        "any_retake_rate_asthma": float((p[p.label == 1].n_slots_retaken > 0).mean()),
        "any_retake_rate_healthy": float((p[p.label == 0].n_slots_retaken > 0).mean()),
        "fisher_p": float(stats.fisher_exact(ct.values)[1]),
    }
    R["demographics"] = {}
    for coh_name, coh in {"all": p, "paper_cohort": p[p.in_paper_cohort]}.items():
        for lab, g in coh.groupby("label"):
            key = f"{coh_name} | {'asthma' if lab == 1 else 'healthy'}"
            R["demographics"][key] = {
                "n": int(len(g)), "age_median_iqr": median_iqr(g.age),
                "sex": g.sex.value_counts(dropna=False).rename(lambda x: str(x)).to_dict(),
                "smoking": g.smoking.value_counts(dropna=False).rename(lambda x: str(x)).to_dict(),
                "age_bin": g.age_bin.value_counts().to_dict(),
            }
    pc = p[p.in_paper_cohort]
    R["age_test_paper_cohort_mannwhitney_p"] = float(stats.mannwhitneyu(pc[pc.label == 1].age.dropna(), pc[pc.label == 0].age.dropna()).pvalue)
    month = p.collection_date.dt.to_period("M").astype(str).replace("NaT", "missing")
    R["month_by_group"] = pd.crosstab(month, p.label).rename(columns={1: "asthma", 0: "healthy"}).to_dict("index")
    h = p[p.label == 0].collection_date
    dd = p.dropna(subset=["collection_date"])
    R["id_vs_date_spearman"] = float(stats.spearmanr(dd.participant_id, dd.collection_date.map(pd.Timestamp.toordinal)).statistic)
    R["temporal"] = {
        "healthy_first_date": str(h.min().date()), "healthy_last_date": str(h.max().date()),
        "patients_recorded_after_last_healthy": int(((p.label == 1) & (p.period == "late")).sum()),
        "time_overlap_counts": p[p.in_paper_cohort & p.in_time_overlap].label.value_counts().rename({1: "asthma", 0: "healthy"}).to_dict(),
    }
    # Hastalarda erken vs geç dönem klinik profili: dönem probunu yorumlamak için gerekli
    pat = pc[(pc.label == 1) & pc.period.isin(["early", "late"])]
    mix = {}
    for col in ["sex", "smoking", "gina_control", "asthma_step", "sft_dx", "ics_use", "allergic_rhinitis"]:
        if col in pat:
            ct = pd.crosstab(pat.period, pat[col].astype(str))
            mix[col] = {"proportions": ct.div(ct.sum(axis=1), axis=0).round(3).to_dict("index"),
                        "chi2_p": float(stats.chi2_contingency(ct).pvalue)}
    for col in ["age", "fev1_pct", "fvc_pct", "fev1_fvc", "act_total"]:
        if col in pat:
            a, b = pat[pat.period == "early"][col].dropna(), pat[pat.period == "late"][col].dropna()
            mix[col] = {"median_early": float(a.median()), "median_late": float(b.median()),
                        "n_early": int(len(a)), "n_late": int(len(b)),
                        "mannwhitney_p": float(stats.mannwhitneyu(a, b).pvalue)}
    R["patients_early_vs_late_case_mix"] = mix
    # Aynı gün kaydedilmiş hasta-sağlıklı yapısı (gün-içi tabakalı AUC için)
    dd2 = pc.dropna(subset=["collection_day"])
    per_day = dd2.groupby(["collection_day", "label"]).size().unstack(fill_value=0)
    both = per_day[(per_day.get(0, 0) > 0) & (per_day.get(1, 0) > 0)]
    # Gün ve hafta düzeyinde grup dağılımı: "her gün iki grup da var mı?" sorusunun cevabı
    wk = dd2.groupby([pd.to_datetime(dd2["collection_day"]).dt.to_period("W"), "label"]).size().unstack(fill_value=0)
    R["day_week_structure"] = {
        "n_days": int(len(per_day)),
        "n_days_patients_only": int(((per_day.get(1, 0) > 0) & (per_day.get(0, 0) == 0)).sum()),
        "n_days_healthy_only": int(((per_day.get(0, 0) > 0) & (per_day.get(1, 0) == 0)).sum()),
        "n_days_both": int(len(both)),
        "n_asthma_on_patient_only_days": int(per_day[per_day.get(0, 0) == 0][1].sum()),
        "n_weeks": int(len(wk)), "n_weeks_with_healthy": int((wk.get(0, 0) > 0).sum()),
        "n_weeks_patients_only": int(((wk.get(0, 0) == 0) & (wk.get(1, 0) > 0)).sum()),
    }
    # "Veri toplama tarihi" gerçek ziyaret tarihi mi? (SFT tarihi ve yaşla tutarlılık)
    sft = pd.to_datetime(df["Solunum Fonksiyon testi (SFT) Yapılma Tarihi"], errors="coerce")
    col = pd.to_datetime(df[COL_DATE], errors="coerce")
    diff = (col - sft).dt.days[(p.label == 1).values].dropna()
    dob_ = pd.to_datetime(df[COL_DOB], errors="coerce")
    age_calc = np.floor((col - dob_).dt.days / 365.2425)
    age_rep = pd.to_numeric(df[COL_AGE], errors="coerce")
    m = age_calc.notna() & age_rep.notna()
    R["date_field_validity"] = {
        "patients_collection_equals_sft_date_frac": float((diff == 0).mean()),
        "patients_within_7_days_of_sft_frac": float((diff.abs() <= 7).mean()),
        "n_patients_compared": int(len(diff)),
        "age_consistent_with_collection_date": f"{int((age_calc[m] == age_rep[m]).sum())}/{int(m.sum())}",
    }
    R["same_day_design"] = {
        "n_days_with_both_groups": int(len(both)),
        "n_asthma_on_shared_days": int(both[1].sum()), "n_healthy_on_shared_days": int(both[0].sum()),
        "n_within_day_pairs": int((both[0] * both[1]).sum()),
    }

    # Aynı doğum tarihi + cinsiyet: aynı kişi iki ID ile kayıtlı olabilir mi? (kimlik sızıntısı)
    dob = pd.to_datetime(df[COL_DOB], errors="coerce")
    tmp = pd.DataFrame({"id": p.participant_id.values, "label": p.label.values, "dob": dob.values, "sex": p.sex.values}).dropna(subset=["dob", "sex"])
    pairs = []
    for _, g in tmp.groupby(["dob", "sex"]):
        if len(g) > 1:
            pairs.append({"ids": g.id.tolist(), "labels": g.label.tolist(),
                          "dob_is_jan_1": bool(g.dob.iloc[0].month == 1 and g.dob.iloc[0].day == 1)})
    R["same_dob_and_sex_groups"] = pairs  # Doğum tarihinin kendisi rapora YAZILMAZ
    # Hangi değişkenler hangi grupta dolu? (eksiklik deseni = potansiyel etiket vekili)
    fill = pd.DataFrame({
        "asthma": df[p.label.values == 1].notna().mean(),
        "healthy": df[p.label.values == 0].notna().mean()})
    R["variables_filled_in_both_groups_ge_80pct"] = fill[(fill.asthma >= .8) & (fill.healthy >= .8)].index.tolist()
    R["variables_patient_only_ge_80pct"] = fill[(fill.asthma >= .8) & (fill.healthy == 0)].index.tolist()
    R["patient_clinical_distributions"] = {k: pc[pc.label == 1][k].value_counts(dropna=False).rename(lambda x: str(x)).to_dict()
                                           for k in ["gina_control", "gina_symptom_level", "asthma_step", "sft_dx"] if k in p}
    # Örneklem büyüklüğünün getirdiği belirsizlik (beklenen AUC CI yarı genişliği)
    R["expected_auc_ci_halfwidth_at_auc_0.80"] = {
        "paper_cohort_284v60": hanley_mcneil_ci(0.80, 284, 60)[1],
        "time_overlap_99v57": hanley_mcneil_ci(0.80, 99, 57)[1],
        "single_test_fold_57v12": hanley_mcneil_ci(0.80, 57, 12)[1],
    }

    # ---- EXP-001 ----
    exp001 = confounder_baselines(p)
    (args.out_report / "EXP-001_confounder_baselines.json").write_text(json.dumps(exp001, indent=2, ensure_ascii=False))
    # ---- EXP-002 ----
    exp002 = clinical_period_probe(p)
    (args.out_report / "EXP-002_clinical_period_probe.json").write_text(json.dumps(exp002, indent=2, ensure_ascii=False, default=str))

    # ---- Çıktılar ----
    priv = args.out_private / "participants.csv"
    p.to_csv(priv, index=False)
    R["outputs"] = {"participants_csv": str(priv), "participants_csv_sha256": sha256(priv)}
    (args.out_report / "clinical_audit.json").write_text(json.dumps(R, indent=2, ensure_ascii=False, default=str))

    lines = ["# Clinical audit (otomatik üretildi — elle düzenlemeyin)", "",
             f"- CSV sha256: `{R['inputs']['clinical_csv_sha256']}`",
             f"- CSV == XLSX: {R.get('csv_vs_xlsx', {}).get('identical')}",
             f"- Tüm kohort: {R['counts']['all']}",
             f"- Makale kohortu (ID ≤ {PAPER_COHORT_MAX_ID}): {R['counts']['paper_cohort_ids_le_101344']}",
             f"- Demografisi tamamen boş sağlıklılar: {R['counts']['healthy_without_any_demographics']}",
             f"- Tarama kriterleri kaydedilmemiş: {[r['participant_id'] for r in R['counts']['screening_not_recorded']]}",
             f"- 18–65 dışı hasta: {R['counts']['patients_outside_18_65']}",
             f"- Son sağlıklı kayıttan sonra kaydedilen hasta: {R['temporal']['patients_recorded_after_last_healthy']}",
             f"- Zaman-örtüşen alt kohort: {R['temporal']['time_overlap_counts']}",
             f"- Tekrar-deneme oranı (astım / sağlıklı): {R['retakes_by_group']['any_retake_rate_asthma']:.3f} / "
             f"{R['retakes_by_group']['any_retake_rate_healthy']:.3f} (Fisher p={R['retakes_by_group']['fisher_p']:.3f})",
             f"- Yaş farkı (makale kohortu) Mann-Whitney p = {R['age_test_paper_cohort_mannwhitney_p']:.2e}",
             f"- Katılımcı ID ↔ kayıt tarihi Spearman ρ = {R['id_vs_date_spearman']:.3f}",
             "", "## Demografi", ""]
    for k, v in R["demographics"].items():
        lines.append(f"- **{k}**: n={v['n']}, yaş medyan (IQR) {v['age_median_iqr']}, cinsiyet {v['sex']}, sigara {v['smoking']}")
    lines += ["", "## Hastalarda erken vs geç dönem klinik profili", ""]
    for k, v in R["patients_early_vs_late_case_mix"].items():
        if "chi2_p" in v:
            lines.append(f"- {k}: p={v['chi2_p']:.3g} · {v['proportions']}")
        else:
            lines.append(f"- {k}: medyan erken {v['median_early']:.1f} (n={v['n_early']}) / geç {v['median_late']:.1f} (n={v['n_late']}), p={v['mannwhitney_p']:.3g}")
    dw, dv = R["day_week_structure"], R["date_field_validity"]
    lines += ["", "## Gün / hafta yapısı ve tarih alanının geçerliliği", "",
              f"- {dw['n_days']} kayıt günü: yalnız hasta {dw['n_days_patients_only']}, yalnız sağlıklı {dw['n_days_healthy_only']}, "
              f"ikisi birden {dw['n_days_both']}; yalnız-hasta günlerinde kaydedilen hasta: {dw['n_asthma_on_patient_only_days']}",
              f"- {dw['n_weeks']} hafta: sağlıklı olan {dw['n_weeks_with_healthy']}, yalnız hasta {dw['n_weeks_patients_only']}",
              f"- Hastalarda veri toplama tarihi = SFT tarihi: %{100*dv['patients_collection_equals_sft_date_frac']:.1f} "
              f"(±7 gün: %{100*dv['patients_within_7_days_of_sft_frac']:.1f}); yaş bu tarihe göre tutarlı: {dv['age_consistent_with_collection_date']}"]
    lines += ["", f"## EXP-002 — Klinik profil → kayıt dönemi, yalnız hastalar (erken {exp002['n_early']} / geç {exp002['n_late']}; 5-fold × 20)", "",
              "| model | AUC ort ± SD | fold %2.5–97.5 |", "|---|---|---|"]
    for k, v in exp002["models"].items():
        lines.append(f"| {k} | {v['auc_mean']:.3f} ± {v['auc_sd']:.3f} | {v['auc_p2.5']:.2f}–{v['auc_p97.5']:.2f} |")
    lines.append(f"\nTedavi basamağı × dönem: {exp002['step_by_period']}")
    sd = R["same_day_design"]
    lines += ["", f"## Aynı gün tasarımı: {sd['n_days_with_both_groups']} gün, {sd['n_asthma_on_shared_days']} astım / "
                  f"{sd['n_healthy_on_shared_days']} sağlıklı, {sd['n_within_day_pairs']} gün-içi çift"]
    lines += ["", "## Ay × grup", "", "| ay | astım | sağlıklı |", "|---|---|---|"]
    for m, row in R["month_by_group"].items():
        lines.append(f"| {m} | {row.get('asthma', 0)} | {row.get('healthy', 0)} |")
    lines += ["", "## EXP-001 — Sesi kullanmayan baseline'lar (5-fold × 20 tekrar, lojistik regresyon)", "",
              "| kohort | özellik | n (astım/sağlıklı) | AUC ort ± SD | AUC fold %2.5–97.5 | BalAcc ort ± SD |", "|---|---|---|---|---|---|"]
    for k, v in exp001.items():
        c, f = k.split(" | ")
        lines.append(f"| {c} | {f} | {v['n_participants']} ({v['n_asthma']}/{v['n_healthy']}) | "
                     f"{v['auc_mean']:.3f} ± {v['auc_sd']:.3f} | {v['auc_p2.5']:.2f}–{v['auc_p97.5']:.2f} | "
                     f"{v['balacc_mean']:.3f} ± {v['balacc_sd']:.3f} |")
    lines += ["", "## Beklenen AUC belirsizliği (AUC=0.80'de %95 CI yarı genişliği, Hanley–McNeil)", ""]
    for k, v in R["expected_auc_ci_halfwidth_at_auc_0.80"].items():
        lines.append(f"- {k}: ±{v:.3f}")
    lines += ["", "## Aynı doğum tarihi + cinsiyet grupları (ID'ler; tarih yazılmaz)", ""]
    for g in pairs:
        lines.append(f"- {g['ids']} etiketler={g['labels']} 1-Ocak-tarihi={g['dob_is_jan_1']}")
    (args.out_report / "clinical_audit.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
