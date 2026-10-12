"""(C) Makale reposundaki klinik dosyaların katılımcı kimliğine göre hizalanma denetimi (yalnız toplu sayılar basar).

Karşılaştırılan kaynaklar (hepsi kimlikli):
  OURS  = clinical_data.csv (bizim 2026-03-14 dışa aktarımımız)
  K0704 = astim-tarama_Data_Report_20250704_cleaned.csv (makale reposu, 04072025 hattının girdisi)
  MASTER= master_acoustic_features_v2.csv (K0704 + ses özellikleri; 04072025 hattının çıktısı)
  TEZ   = Ast_m_tezAnaliz_Data_noLabels_21072025.csv (yalnız astım; 18-24072025 hatlarının etiket kaynağı)
Soru: aynı katılımcı kimliğinde demografi (etiket, yaş, cinsiyet) ve klinik blok (spirometri, basamak, GINA ...) aynı mı?
Kullanım: python -I alignment_check.py <paper_repo_csv_dir>   (dosyalar '-<ad>.csv' ile biter)"""
import sys, glob
import numpy as np
import pandas as pd

D = sys.argv[1]


def f(name, must=None):
    ps = [p for p in glob.glob(D + "/*.csv") if p.endswith("-" + name) and (must is None or must in p)]
    assert len(ps) >= 1, name
    return ps[0]


def num(s):
    return pd.to_numeric(s.astype(object).astype(str).str.replace(",", ".").str.strip(), errors="coerce")


OURS = pd.read_csv(f("clinical_data.csv"), sep=None, engine="python")
OURS["A"] = (OURS["Gönüllü kategorisi"] == "hasta").astype(int)
K = pd.read_csv(f("astim-tarama_Data_Report_20250704_cleaned.csv", "d7b397be"), sep=None, engine="python")
K = K.rename(columns={"﻿Hasta": "Hasta"})
MASTER = pd.read_csv(f("master_acoustic_features_v2.csv")).drop_duplicates("SubjectNumber")
TEZ = pd.read_csv(f("Ast_m_tezAnaliz_Data_noLabels_21072025.csv"), sep=None, engine="python").rename(columns={"﻿ID": "ID"})
print(f"satır: OURS {len(OURS)}, K0704 {len(K)}, MASTER katılımcı {len(MASTER)}, TEZ {len(TEZ)}")

# 1) Demografi hizası
J = K.merge(OURS, on="Hasta", suffixes=("_k", "_o"))
assert len(J) == len(K) == len(OURS)
lab_eq = (J["Gönüllü kategorisi_k"] == J["A"]).mean()
age_eq = (num(J["Yaş_k"]) == num(J["Yaş_o"])).mean()
sex_eq = (J["Cinsiyet_k"].map({1.0: "kadın", 0.0: "erkek"}) == J["Cinsiyet_o"]).mean()
print(f"[K0704 vs OURS] aynı kimlikte etiket eşit {lab_eq:.3f} | yaş eşit {age_eq:.3f} | cinsiyet eşit {sex_eq:.3f}")
JM = MASTER.merge(OURS, left_on="SubjectNumber", right_on="Hasta")
print(f"[MASTER vs OURS] eşleşen {len(JM)} | etiket eşit {(JM.DiseaseCategory == JM.A).mean():.3f}")

# 2) Klinik blok hizası: spirometri üçlüsü (FVC%, FEV1%, FEV1/FVC) aynı kimlikte mi, başka kimlikte mi?
cols = ["FVC(Yüzde)", "FEV1(Yüzde)", "FEV1/FVC (Yüzde)"]


def keys(T):
    A = np.column_stack([num(T[c]).to_numpy(float) for c in cols])
    ok = np.isfinite(A).all(1)
    return [tuple(np.round(r, 1)) if o else None for r, o in zip(A, ok)]


cmap = {}
for k, h in zip(keys(OURS), OURS.Hasta):
    if k is not None:
        cmap.setdefault(k, []).append(h)
same = other = miss = 0
other_owner_cat = {}
other_row_cat = {}
own = OURS.set_index("Hasta")["A"]
for k, h, cat in zip(keys(K), K.Hasta, K["Gönüllü kategorisi"]):
    if k is None:
        continue
    hs = cmap.get(k)
    if hs is None:
        miss += 1
    elif h in hs:
        same += 1
    else:
        other += 1
        o = int(own[hs[0]])
        other_owner_cat[o] = other_owner_cat.get(o, 0) + 1
        other_row_cat[int(cat)] = other_row_cat.get(int(cat), 0) + 1
print(f"[K0704 spirometri üçlüsü] dolu {same + other + miss}: aynı kimlikte {same}, BAŞKA kimlikte {other}, OURS'ta yok {miss}")
print(f"   başka kimliğe gidenlerin OURS'taki sahibi (1=astım): {other_owner_cat} | K0704'te düştüğü satırın etiketi: {other_row_cat}")
for lab, nm in [(0, "sağlıklı"), (1, "astım")]:
    k = K[K["Gönüllü kategorisi"] == lab]
    o = OURS[OURS.A == lab]
    print(f"   {nm}: spirometri dolu K0704 {int(num(k['FEV1(Yüzde)']).notna().sum())}/{len(k)} | OURS {int(num(o['FEV1(Yüzde)']).notna().sum())}/{len(o)}"
          f" ; astım basamağı dolu K0704 {int(k['Astım Basamak'].notna().sum())} | OURS {int(o['Astım Basamak'].notna().sum())}")
both = J["Astım Basamak_k"].notna() & J["Astım Basamak_o"].notna()
st = J.loc[both, "Astım Basamak_o"].value_counts(normalize=True)
print(f"   basamak ikisi de dolu {int(both.sum())}: aynı kimlikte eşit {(J.loc[both, 'Astım Basamak_k'] == J.loc[both, 'Astım Basamak_o']).mean():.3f}"
      f" (rastgele eşleşme beklentisi Σp² = {(st ** 2).sum():.3f})")

# 3) TEZ dosyası hizası (18-24072025 hatları)
JT = TEZ.merge(OURS, left_on="ID", right_on="Hasta")
for a, b in [("age", "Yaş"), ("fev1perc", "FEV1(Yüzde)"), ("fvcperc", "FVC(Yüzde)"), ("fev1fvcperc", "FEV1/FVC (Yüzde)"), ("asthmastep", "Astım Basamak")]:
    x, y = num(JT[a]), num(JT[b])
    m = x.notna() & y.notna()
    print(f"[TEZ vs OURS] {a:12s} n={int(m.sum())} eşit {(abs(x[m] - y[m]) < 0.01).mean():.3f}")
print(f"[TEZ vs OURS] cinsiyet eşit {(JT.sex.map({1: 'kadın', 0: 'erkek'}) == JT.Cinsiyet).mean():.3f}")

# 4) Makalenin astım özellikleri tablosu hangi kaynakla tutarlı?
print("makale (astım): FVC% 86.0, FEV1% 78.15, FEV1/FVC 75.89, IgE 52.95, Eos% 2.20")
for c in ["FVC(Yüzde)", "FEV1(Yüzde)", "FEV1/FVC (Yüzde)", "Total Ig E Sonuç:", "Eozinofil (%): "]:
    print(f"   {c:20s} OURS medyan {num(OURS.loc[OURS.A == 1, c]).median():.2f} | K0704 medyan {num(K.loc[K['Gönüllü kategorisi'] == 1, c]).median():.2f}")
# 5) Cinsiyet dengesi (toplu)
ct = pd.crosstab(OURS.A, OURS.Cinsiyet, normalize="index").round(3)
print("cinsiyet oranı (satır = 0 sağlıklı / 1 astım):", ct.to_dict())
