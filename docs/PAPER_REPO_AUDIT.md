# Makale reposunun denetimi (EXP-031)

**Konu:** Alagöz ve ark., "Voice Analysis in Asthmatic and Healthy Individuals Using Machine Learning Models" (BMC Pulm Med, baskıda; doi:10.1186/s12890-026-04654-9). Yazarların kod ve ara dosya reposu (`Voice_Analysis_in_Asthmatic_repo`, Claude projesine 2026-10-12'de yüklendi).

**Tür:** Post hoc, betimsel denetim (kullanıcı isteği, 2026-10-12). Önceden yazılmış bir hipotez testi değil. Karar kapısı değil.

**Veri kuralı (D-014):** Bu belge ve `reports/paper_repo_audit/` yalnız toplu sayılar içerir. Katılımcı düzeyi dosyalar git'te yok. Repodaki kod çalıştırılmadı; yalnız okundu. Yeniden üretimler bizim yazdığımız betiklerle, ayrı bir dizinde, `python -I` ile yapıldı.

---

## 0. Özet

1. **Ne yaptılar:** Astım–sağlıklı ayrımı (makalenin Görev 1'i) + astım içinde 30'a yakın klinik hedef (GINA / AKT kontrolü, astım basamağı, alerjik rinit, SFT tanısı, semptomlar, ilaçlar, sigara, cinsiyet) + spirometri ve yaş için regresyon. Hepsi aynı 44 özellik (12 MFCC + Δ + ΔΔ ortalamaları + 8 spektral istatistik) ve ses birimi başına ayrı model.
2. **Makalenin sayıları bu repodan birebir çıkmıyor [FACT].** Aynı kod ve aynı özellik dosyalarıyla en iyi AUC'ler 0.64–0.72. Makalede 0.66–0.77 (Tablo 4) ve 0.65–0.74 (Tablo 2). Örnek: "gelecek" bizde 0.710, makalede 0.769 / 0.717.
3. **Kritik veri bulgusu [FACT]:** 4 Temmuz 2025 tarihli temizlenmiş klinik dosyada **klinik değişken bloğu katılımcı kimliğiyle hizasız**. Etiket, yaş ve cinsiyet doğru. Ama spirometri, astım basamağı, GINA, AKT, alerjik rinit, SFT başka katılımcılara ait: spirometri değerlerinin 284'ünden 283'ü başka bir kimliğin satırında. 46 sağlıklı gönüllünün satırında astım hastalarının spirometrisi, 54'ünde astım basamağı var.
   - Bu dosyayı kullanan 04072025 analizleri (5 klinik hedef) **karışık etiketle** yapılmış; sonuçları yorumlanamaz.
   - 18–24 Temmuz analizleri ve makalenin klinik özellikler tablosu **doğru hizalı** kaynakla tutarlı. Bizim `clinical_data.csv` de doğru hizalı kaynakla birebir uyumlu. **Bizim analizlerimiz etkilenmiyor.**
4. **Seçim iyimserliği somut olarak görülüyor [FACT + INFERENCE].** Karışık etiketli (fiilen rastgele) analizde bile "10 modelin en iyisi" dengeli doğruluğu %47–69'a çıkıyor. Doğru hizalı analizde her ses için en iyi model, 12 temel modelin medyanından +0.04 ile +0.09 AUC yukarıda.
5. **Bağlam kontrolü, onların kendi özellikleriyle [FACT, post hoc]:**
   - Yalnız kayıt tarihi AUC 0.85–0.89; tarih + yaş 0.89–0.92.
   - Onların özellikleriyle en iyi ses modeli (7 sesin füzyonu) 0.75–0.79. Gün-gruplu CV'de her model 0.02–0.05 düşüyor.
   - Ses skoru tarih + yaşa eklenince hâlâ etiketle ilişkili (OR/SD ≈ 2.0, gün-kümeli jackknife p 0.006–0.029). Ama bu bağlamda **saat yok**. Bizim ana analizimizde (saatli) aynı tür MFCC skoru gün-kümeli testte anlamlı değildi. Yani bu sonuç bizimkini çürütmez; saatin önemini gösterir [INFERENCE].
6. **Astım içinde ses → klinik durum ilişkisi, doğru hizalı etiketlerle de yok [FACT, bu özelliklerle].**
   - Cinsiyet AUC 0.88–0.96 (pozitif kontrol çalışıyor).
   - Diğer bütün klinik hedefler medyan AUC 0.50–0.56.
   - FEV1%, FVC%, FEV1/FVC regresyonu R² < 0. Yaş R² ≤ 0.05.
7. **Açık sorularımızdan birkaçı cevaplandı** (bölüm 7): GINA 47 / 237 farkı, makalenin kodu, seçim kaydı, 344 / 342 farkı, SFT'nin aynı gün yapılıp yapılmadığı.

---

## 1. Reponun haritası (ne, ne zaman, hangi girdiyle)

| Klasör / betik | Girdi | Ne yapıyor | Çıktı | Makaleyle ilişkisi |
|---|---|---|---|---|
| `04072025/create_master_features.py` | `astim-tarama_Data_Report_20250704_cleaned.csv` + `.m4a` | 22.05 kHz, `librosa.effects.trim(top_db=20)` (yalnız baş / son sessizlik), 44 özellik; klinik satırı kimlikle birleştirir | `master_acoustic_features_v2.csv` (kimlikli; 342 katılımcı, 283 / 59) | Ara dosya |
| `04072025/exploratory_analyzer.py` | master | Yalnız astım; tek RF; 5 hedef (GINA, AKT, alerjik rinit, basamak 1–3 / 4–5, SFT normal / anormal); dengeli doğruluk | `exploratory_analysis_summary.csv` | Makalede yok |
| `04072025/comprehensive_analyzer.py` | master | Aynı 5 hedef; 10 model; **en iyi dengeli doğruluk** seçilip raporlanıyor | `comprehensive_analysis_summary.csv` | Makalede yok |
| `04072025/model_trainer_updated.py` | master | Astım içi GINA; karıştırmalı StratifiedKFold | — | Görev 2'nin erken hali olabilir |
| `18072025/feature_extraction.py` | `.m4a` | **Spektral çıkarma** (ilk 0.2 s gürültü tahmini), Hamming 2048 / 512, 44 özellik; ses başına dosya, **kimlik yok**, sıra `os.listdir` | `<ses>_features.csv` (342 satır; ordu 341) | **Görev 1'in girdisi** [INFERENCE: kod yapısı makaleyle aynı] |
| `18072025/model_training.py` | `<ses>_features.csv` | 12 model; `cross_validate(cv=5)` (**karıştırmasız**); en iyi 5 modelden Voting / Stacking / Weighted Voting | sonuç tabloları | **Görev 1 hattı** |
| `model_training_medicalData_CO.py` | `features_with_all_medical_labels.csv` (kimlikli, yalnız astım) | Aynı 12 model, 26 ikili klinik hedef | `medical_data_full_cv_report.csv` (2002 satır), `best_model_summary_report.csv` | Makalede yok |
| `21072025/`, `24072025/` | `features_with_all_medical_labels_21072025_wGINA.csv`, `features_onlyGINA_24072025.csv`; etiket kaynağı `Ast_m_tezAnaliz_*`, `Ast_m_hastalari_GINAcat.csv` | Yalnız astım; sınıflama (GINA_class_new, AKT, semptom, sigara, komorbidite …) + regresyon (FEV1%, FVC%, FEV1/FVC, yaş); `roc_auc_ovr`, ağırlıklı metrikler | `medical_data_full_cv_report_21072025*.csv`, `*GINA_class_new*` | Görev 2'nin (GINA) kaynağı [INFERENCE] |
| `31072025/RobMulla-styleTestGINA.py` | — | Yalnız import satırları; yarım kalmış | — | — |
| `Spectogram-tutorial.py` | — | Eğitim amaçlı örnek | — | İlgisiz |

Not: Makale "enerji tabanlı VAD ile sessizlik kırpma" diyor. Kodda iki farklı ön işleme var: 4 Temmuz'da baş / son kırpma, 18 Temmuz'da spektral çıkarma. Makaledeki tabloların hangisiyle üretildiği koddan kesin çıkmıyor [NEEDS VERIFICATION].

---

## 2. Yeniden üretim A: Görev 1 hattı, birebir

**Ne yapıldı:** `18072025/model_training.py` ile aynı 12 model ve ayarlar, aynı pipeline (StandardScaler → SMOTE → model), aynı `cross_validate(cv=5)` (karıştırmasız), en iyi 5 modelden aynı üç ensemble. Girdi: repodaki `<ses>_features.csv`. Betik: `scripts/paper_repo_audit/repro_pipeline.py`.

| Ses | Makale Tablo 2 | Makale Tablo 4 | Bizim en iyi (15 model) | LR | 12 temel modelin medyanı | En iyinin özgüllüğü |
|---|---|---|---|---|---|---|
| aaa | Voting 0.649 | MLP 0.700 | Stacking 0.648 ± 0.069 | 0.628 | 0.590 | 0.21 |
| ana | Voting 0.650 | MLP 0.690 | LR 0.682 ± 0.072 | 0.682 | 0.585 | 0.52 |
| araba | Voting 0.704 | LR 0.656 | Karar ağacı 0.644 ± 0.099 | 0.576 | 0.554 | 0.51 |
| gelecek | Voting 0.717 | Voting 0.769 | Weighted Voting 0.710 ± 0.050 | 0.675 | 0.633 | 0.22 |
| ordu | SVM 0.686 | XGBoost 0.674 | SVM 0.636 ± 0.073 | 0.613 | 0.596 | 0.24 |
| titiz | Voting 0.736 | Voting 0.723 | Stacking 0.653 ± 0.055 | 0.610 | 0.613 | 0.21 |
| ünlem | Voting 0.726 | XGBoost 0.749 | LR 0.720 ± 0.100 | 0.720 | 0.631 | 0.64 |

(± = 5 fold SD, makaledeki gibi.)

**Okuma:**
- **[FACT]** Repo kodu ve repo özellikleriyle makalenin iki tablosundan hiçbiri birebir çıkmıyor. Bizim en iyiler çoğunlukla daha düşük (titiz −0.07 / −0.08, gelecek −0.06 / −0.01). Hat karıştırmasız ve tohumlu olduğu için koşudan koşuya değişmemeli.
- **[INFERENCE]** Olası nedenler: makaledeki tablolar başka bir özellik sürümüyle ya da kodun başka bir haliyle üretildi; kütüphane sürümleri (sklearn / xgboost / catboost) farklı. Hangisi olduğunu repodan ayıramayız [NEEDS VERIFICATION, yazarlardan].
- **[FACT]** Her seste "en iyi" model, 12 temel modelin medyanından +0.04 ile +0.09 yukarıda. Bu, aynı test fold'larıyla seçimin üst sınırı gibi okunabilir. Seçim için iç içe CV yok.
- **[FACT]** En iyi modellerin çoğunda özgüllük 0.21–0.24: model çoğu kişiye "astım" diyor. Yüksek doğruluk (0.78–0.80) büyük ölçüde %83 astım oranından geliyor.

---

## 3. Veri hizalama bulgusu (C)

**Nasıl fark edildi:** Master dosyasında sağlıklı gönüllülerin (n = 59) 53'ünde astım basamağı (3, 4, 5) ve GINA / AKT değerleri vardı. Bizim dosyamızda sağlıklıların bu alanları boş.

**Kontrol (betik: `scripts/paper_repo_audit/alignment_check.py`; çıktı: `reports/paper_repo_audit/alignment_check.log`):**

| Karşılaştırma (aynı katılımcı kimliğinde) | Sonuç |
|---|---|
| 20250704 dosyası vs bizim dosya: etiket / yaş / cinsiyet | 1.000 / 0.980 / 0.986 eşit |
| Spirometri üçlüsü (FVC%, FEV1%, FEV1/FVC): 284 dolu satır | **1'i aynı kimlikte, 283'ü başka bir kimliğin satırında**; hepsi bizim dosyada bir astım hastasına ait |
| Bu üçlülerin düştüğü satırlar | 237 astım, **46 sağlıklı** satırı |
| Astım basamağı (ikisi de dolu, n = 228) | Aynı kimlikte eşitlik 0.325; rastgele eşleşme beklentisi 0.368 |
| Tez dosyası (`Ast_m_tezAnaliz_Data_noLabels_21072025`) vs bizim dosya | yaş 1.000, FEV1% 0.993, FEV1/FVC 1.000, basamak 1.000, cinsiyet 1.000 |
| 18–24 Temmuz özellik + etiket dosyaları vs bizim dosya | yaş, cinsiyet, basamak 1.000; FEV1% 0.993 |
| Makalenin astım özellikleri tablosu (FEV1% 78.15, FEV1/FVC 75.89, IgE 52.95) | Bizim dosyanın medyanlarıyla **birebir**; 20250704 dosyasıyla değil (76.40 / 74.67 / 51.80) |

**Yorum:**
- **[FACT]** 20250704 temizlenmiş dosyasında klinik blok, kimlik / etiket / yaş / cinsiyet sütunlarından bağımsız olarak yer değiştirmiş. Basit bir satır kayması değil (±6 satır kaydırma eşleşmeyi artırmıyor).
- **[INFERENCE, güçlü]** Hatalı olan 20250704 dosyası; bizim dışa aktarımımız doğru. Dayanak: (1) sağlıklıda astım basamağı klinik olarak anlamsız; (2) üç bağımsız kaynak (bizim dosya, tez dosyaları, makalenin tablosu) birbiriyle uyumlu, aykırı olan yalnız bu dosya. Kesin teyit veri ekibinden [NEEDS VERIFICATION].
- **Etkisi:**
  - 04072025 klasöründeki 5 klinik hedef analizi (exploratory, comprehensive, model_trainer_updated) karışık etiketle yapılmış. Sonuçları bilgi taşımıyor.
  - Makalenin Görev 1'i (astım–sağlıklı) etkilenmiyor: etiket doğru.
  - Makalenin Görev 2'si (GINA) büyük olasılıkla etkilenmiyor: 47 / 237 sayısı doğru hizalı `GINA_class_new` ile birebir [INFERENCE].
  - **Bizim projemiz etkilenmiyor:** etiket, yaş ve tarih kendi dışa aktarımımızdan; o da doğru hizalı kaynaklarla uyumlu [FACT].

**Yan ürün, seçim iyimserliğinin canlı örneği [INFERENCE]:** Karışık etiketle yapılan "10 modelin en iyisi" tablosunda dengeli doğruluk %46.7–69.2 (medyan ≈ %57). Tek model (RF) aynı hedeflerde %39–62 (medyan ≈ %48). Etiket fiilen rastgeleyken bile "en iyi model" seçimi %60'ların üstünde sayılar üretebiliyor.

---

## 4. Bağlam analizi B: onların özellikleri, bizim çerçevemiz

**Ne yapıldı (post hoc):** Master dosyasındaki 44 özellik (4 Temmuz sürümü). Etiket ve tarih bizim dosyamızdan (kimlikle). 340 katılımcı (283 / 57; tarihi olmayan 2 kişi çıktı), 67 gün. Makalenin ayarlarıyla LR, SVM, RF (StandardScaler → SMOTE). 7 sesin olasılık ortalaması = füzyon. İki CV düzeni, her biri 5 tekrar:
- katılımcı düzeyi tabakalı 5-fold (makaledeki gibi, karıştırmalı);
- kayıt gününe göre gruplu 5-fold (bizim `grouped_folds`).

Referanslar aynı fold'larda: yalnız tarih (`cr(gün, 4)`), yalnız yaş, tarih + yaş. **Saat bu veride yok.** Betik: `scripts/paper_repo_audit/context_check.py`.

| Kol | Katılımcı CV | Gün-gruplu CV | Fark |
|---|---|---|---|
| Yalnız tarih | 0.886 | 0.851 | −0.035 |
| Tarih + yaş | 0.917 | 0.894 | −0.023 |
| Yalnız yaş | 0.677 | 0.658 | −0.019 |
| LR füzyon | 0.791 | 0.754 | −0.037 |
| SVM füzyon | 0.758 | 0.708 | −0.050 |
| RF füzyon | 0.737 | 0.721 | −0.016 |
| Tek ses (21 kol) | 0.61–0.71 | 0.58–0.68 | −0.06 … +0.01 |

(5 tekrarın ortalaması; tekrarlar arası SD 0.002–0.037: `reports/paper_repo_audit/context_auc.csv`.)

T1 (bağlam = `cr(gün, 4)` + yaş; Firth PLR + gün-kümeli jackknife CV3; OR/SD için katılımcı bootstrap, B = 1000):

| Düzen | Kol | OR/SD [%95 bootstrap] | PLR p | CV3 p |
|---|---|---|---|---|
| Gün-gruplu | LR füzyon | 2.02 [1.34, 3.82] | 0.0009 | 0.029 |
| Gün-gruplu | SVM füzyon | 1.94 [1.32, 3.28] | 0.003 | 0.016 |
| Gün-gruplu | RF füzyon | 2.01 [1.37, 3.74] | 0.0009 | 0.006 |
| Katılımcı | LR / SVM / RF | 2.3–2.7 | < 0.001 | < 0.005 |

**Okuma:**
- **[FACT]** Yalnız kayıt tarihi, onların özellikleriyle kurulan her ses modelinden daha iyi ayırıyor (0.85–0.89 vs ≤ 0.79). Bizim verimizdeki tablo onların özellikleriyle de aynı.
- **[FACT]** Bizim önceki MFCC replikasyonumuzla uyumlu: katılımcı-gruplu füzyon 0.77–0.79, gün-gruplu 0.72–0.73 idi; burada 0.74–0.79 ve 0.71–0.75.
- **[INFERENCE]** T1 burada anlamlı, bizim ana MFCC analizimizde gün-kümeli testte değildi (CV3 0.10 / 0.15). En olası fark: **bu bağlamda saat yok**. Ana analizimizde saat, bağlamın güçlü bir parçasıydı (saat + tarih 0.93–0.94; burada tarih 0.85–0.89). Saat dışarıda kalınca onun açıklayacağı kısım ses skoruna geçebilir. Bu yüzden bu T1 bizim ana sonucumuzun yerine geçmez; tamamlayıcı bir kontroldür.
- **[INFERENCE]** Makalenin "sesler astımı AUC ~0.70–0.77 ile ayırıyor" bulgusu, kendi özellikleriyle de gün gruplamasına karşı birkaç puan duyarlı ve bağlamdan ayrılamıyor.

---

## 5. Astım içi klinik hedefler (doğru hizalı 18–24 Temmuz analizleri)

Kaynak: repodaki rapor dosyaları; yeniden çalıştırılmadı. 44 özellik, 12 sınıflayıcı / 4 regresör, ses başına ayrı model, 5-fold. Etiketler doğru hizalı (bölüm 3).

| Hedef | Sonuç (7 ses × model) |
|---|---|
| Cinsiyet (pozitif kontrol) | AUC medyan 0.92, en yüksek 0.96 |
| GINA_class_new (47 / 237) | medyan 0.53, en yüksek 0.66 |
| AKT kontrol durumu | medyan 0.53, en yüksek 0.69 |
| Hırıltı, nefes darlığı, öksürük, alerjik rinit, ICS / nazal steroid, komorbidite | medyan 0.48–0.53, en yüksek 0.56–0.65 |
| Sigara | medyan 0.56, en yüksek 0.63 |
| 26 ikili hedef (ilaçlar, semptomlar; 18072025 sürümü) | cinsiyet dışı medyan 0.56; hücrelerin %21'i > 0.60 (en yüksek 0.72) |
| FEV1% / FVC% / FEV1/FVC (regresyon) | R² medyan −0.15 / −0.17 / −0.18; en yüksek −0.01 / 0.01 / −0.06 |
| Yaş (regresyon) | R² medyan −0.02, en yüksek 0.05 |

**Okuma:**
- **[FACT]** Hat çalışıyor (cinsiyet 0.92–0.96). Ama 44 özellik, astım hastaları arasında kontrol düzeyini, şiddeti, semptomları, ilaçları ya da akciğer fonksiyonunu ayırt etmiyor. En yüksek değerler, 84 karşılaştırmanın en iyisi olarak beklenen gürültü düzeyinde (≈ 0.60–0.65) [INFERENCE].
- **[INFERENCE]** Yaşın da bu özelliklerden çıkmaması (R² ≈ 0) dikkat çekici. Bu özellikler daha çok cinsiyet / ses tınısı ve kayıt koşulu taşıyor olabilir [HYPOTHESIS].
- **[INFERENCE, zayıf]** Ses astım fizyolojisini (hava yolu daralması) doğrudan yansıtsaydı, astım içinde FEV1% ile bir ilişki beklenirdi. Bu özelliklerle yok. Bu, astım–sağlıklı farkının hava yolundan çok bağlam / demografi kaynaklı olduğu okumasıyla tutarlı; ama kanıtı değil. Nedenleri: sınırlı aralık, tek ölçüm, özelliklerin kaba olması. Derin temsillerle test edilmedi.

---

## 6. Makaledeki diğer tutarsızlıklar (yeni)

- **[FACT, aritmetik]** Tablo 3'te (GINA) özgüllük değerleri imkânsız. Örnek "aaa": doğruluk 0.736, duyarlılık 0.173, özgüllük 0.208. Doğruluk, duyarlılık ve özgüllüğün ağırlıklı ortalaması olduğundan en fazla 0.208 olabilir. Tablo 5'te aynı görevin özgüllükleri tutarlı (0.70–0.95). Tablo 3 büyük olasılıkla eski bir sürümün metrik hatası [INFERENCE].
- **[FACT]** Tablo 5'te "titiz" için "en iyi" model AUC 0.463. 12 model arasında en yüksek AUC'nin 0.5'in altında olması beklenmez; seçim AUC dışında bir ölçütle yapılmış olabilir.
- **[FACT]** Tablo 1 (split denetimi) 344 katılımcı gösteriyor (275 + 69). Repodaki ses dosyaları 342 (283 / 59; ordu 341). Bizim ses kohortumuzla (59 sağlıklı) aynı.
- **[FACT]** GINA 47 / 237 (makale) ile bizim dosyadaki 41 / 242 arasındaki fark çözüldü: makale ayrı bir dosyadaki `GINA_class_new` sınıfını kullanmış. Bu sınıf, ham GINA alanından 6 kişide farklı (5 "kontrol altında değil" + 1 boş → "kontrol altında"). Türetme kuralı bilinmiyor [NEEDS VERIFICATION].
- Önceden bilinenler (eleştirel değerlendirme belgesi, bölüm 3): iki ayrı "Tablo 1 / Tablo 2", aynı görev için farklı en iyi model ve AUC.

---

## 7. Mevcut sorularımıza cevaplar

| Bizim soru | Cevap | Etiket |
|---|---|---|
| Yayımlanmış MFCC çalışmasında split / bağlam kontrolü nasıl? (PHASE3_CONCLUSION, NEEDS VERIFICATION) | Ses başına kişi başına 1 satır; katılımcı sızıntısı olası değil. Ama gün gruplaması ve bağlam kontrolü yok. Kendi özellikleriyle gün-gruplu CV −0.02 / −0.05; tarih tek başına daha iyi | FACT |
| Makaledeki "Tablo 1 split denetimi" koddan üretilebilir mi? | Hayır. Görev 1 dosyalarında kimlik yok; sayılar da tutmuyor (344 vs 342) | FACT |
| Seçim iyimserliği ne kadar? | En iyi model, medyan modelden +0.04 / +0.09 AUC; karışık etiketle bile "en iyi" dengeli doğruluk %69'a çıkıyor | FACT / INFERENCE |
| Makalenin sayıları yeniden üretilebilir mi? | Repo kodu + repo özellikleriyle hayır (en iyiler 0.64–0.72 vs 0.66–0.77) | FACT |
| GINA 47 / 237 vs 41 / 242 | `GINA_class_new` (ayrı dosya); 6 kişi farklı | FACT; kural NEEDS VERIFICATION |
| Klinik dosya 64 sağlıklı, ses kohortu 59 | Yazarların ses dosyaları da 59 sağlıklı (342 katılımcı). Makale 60 / 344 diyor | FACT |
| Aşama C (astım içi klinik) umut verici mi? | Bu 44 özellikle hiçbir klinik hedef ayrılmıyor; FEV1% R² < 0. Derin temsiller test edilmedi | FACT (MFCC) / HYPOTHESIS (derin) |
| SFT ses kaydıyla aynı gün mü? (G3 prosedür yolu) | Bizim dosyada astımlıların 277 / 284'ünde SFT tarihi = kayıt tarihi; sağlıklılarda SFT kaydı yok. Sıra (önce / sonra) ve bronkodilatör hâlâ bilinmiyor | FACT; sıra NEEDS VERIFICATION |
| Cinsiyet karıştırıcı mı? | Özellikler cinsiyeti çok iyi taşıyor (AUC 0.95). Ama gruplar arası oran yakın (sağlıklı %64, astım %69 kadın); büyük bir karıştırıcı olması beklenmez. T1 bağlamımızda yok; duyarlılık analizi olarak eklenebilir | FACT / INFERENCE |

Not: Önceki çalışma notlarımızda geçen "astım içi klinik hedefler şansta → astım sinyali hava yolu fizyolojisi değil" okuması, karışık etiketli 04072025 sonuçlarını da içeriyordu. Doğru hizalı 18–24 Temmuz sonuçları da aynı yönde, bu yüzden okuma değişmiyor; ama dayanağı yalnız doğru hizalı analizler olmalı ve "zayıf çıkarım" düzeyinde kalmalı.

---

## 8. Ne değişir? (öneriler, ÖNERİLDİ)

- **Aşama A (yazım; D-044):** Bu denetim, ana mesajı karşı tarafın kendi özellikleriyle destekliyor: "tarih tek başına sesten iyi; gün gruplaması AUC'yi düşürüyor; en iyi model seçimi şişiriyor". Rapora "dış hat kontrolü" olarak girer.
- **Veri ekibine geri bildirim (öncelikli; yazılı):**
  1. 20250704 temizlenmiş klinik dosyasında hizalama sorunu. O dosyadan üretilmiş başka analiz ya da sunum varsa kontrol edilmeli.
  2. `GINA_class_new` türetme kuralı.
  3. 344 / 60 vs 342 / 59 farkı.
  4. Sağlıklılarda SFT yapıldı mı, yapıldıysa değerleri nerede? SFT ile ses kaydının sırası ve bronkodilatör.
  5. Makale tablolarını üreten özellik sürümü ve kod (Tablo 2 vs 4, Tablo 3 vs 5).
- **Aşama C:** Bu 44 özellik için önsel düşük. Derin temsillerle tek, önceden yazılı bir test (ör. astım içinde FEV1%, cinsiyet pozitif kontrolüyle) ucuz: GPU gerekmez, gömmeler hazır. Değer: olumlu çıkarsa veri setindeki tek bağlamdan görece bağımsız ses–fizyoloji kanıtı olur. Karar ayrı bir tasarım belgesiyle.

---

## 9. Sınırlılıklar

- Repo tam değil: makale tablolarını üreten son kod / özellik sürümü yok.
- Kütüphane sürümleri bilinmiyor; A'daki farkın bir kısmı sürümden gelebilir.
- B'de saat yok; T1 bizim ana analizimizden daha zayıf bir bağlam kontrolü.
- Bölüm 5'teki sonuçlar onların rapor dosyalarından okundu; yeniden çalıştırılmadı.
- Tüm analizler post hoc ve betimsel; çoklu karşılaştırma düzeltmesi yok.

---

## 10. Yeniden üretim

Girdi dizini: Claude projesindeki repo dosyaları ve `clinical_data.csv`, ayrı bir dizine kopyalanmış (dosya adları `-<ad>.csv` ile biter).

```
python -I scripts/paper_repo_audit/alignment_check.py <dizin>                 # bölüm 3 (saniyeler)
python -I scripts/paper_repo_audit/context_check.py <dizin> <çıktı_öneki>     # bölüm 4 (~6 dk, 2 CPU)
python -I scripts/paper_repo_audit/repro_pipeline.py <dizin> <çıktı.csv> 0    # bölüm 2 (~13 dk, 2 CPU)
```

Çıktılar (toplu): `reports/paper_repo_audit/{alignment_check.log, context_auc.csv, context_t1.csv, repro_pipeline.csv}`.
