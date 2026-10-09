# EXPERIMENTS.md — Bilimsel deney günlüğü

**Nasıl kullanılır**
- Her deney benzersiz bir ID alır: `EXP-001`, `EXP-002`, … ID'ler **yeniden kullanılmaz**, başarısız ya da yarıda kalan deneyler **silinmez** (DURUM: başarısız / iptal + neden).
- Bu dosya insanlar için: soru, tasarım, yorum, kısıtlar. Sayıların makinece okunan kopyası `results/registry.csv`'de (Google Sheets bunun içe aktarılmış görünümü).
- Deney **başlamadan önce** "Soru", "Tasarım" ve "Beklenti" yazılır; sonuçlar sonra eklenir. Böylece sonuca bakıp soruyu değiştirmeyiz.
- Her deney hangi araştırma sorusuna (RQ) hizmet ettiğini yazar. RQ listesi en altta.

---

## EXP-001 — Sesi kullanmayan confounder baseline'ları
| Alan | Değer |
|---|---|
| Tarih | 2026-10-08 |
| DURUM | Tamamlandı |
| Araştırma sorusu | RQ7 (model astımı mı öğreniyor, confounder'ı mı?) — ses modelleri için alt sınır |
| Veri sürümü | `clinical_data.csv` sha256 `cab18a356920df713a22d0084e4e366f7ce0a2f290f640d0f9b565d523a3af9b` |
| Katılımcı | 344 (makale kohortu) — özelliği eksik olanlar ilgili satırda düşer; zaman-örtüşen alt kohort 156 |
| Hasta / kontrol | 284 / 60 (alt kohort 99 / 57) |
| Kayıt türleri | Yok (ses kullanılmadı) |
| Split stratejisi | Katılımcı düzeyinde `RepeatedStratifiedKFold(5, 20 tekrar, random_state=0)` = 100 fold |
| Girdi | Ayrı ayrı: yaş; cinsiyet; sigara (one-hot); yaş+cinsiyet+sigara; kayıt tarihi (gün); katılımcı ID numarası |
| Sampling rate / ön işleme | — / StandardScaler (pipeline içinde, yalnız train fold'una fit) |
| Model | Lojistik regresyon, `class_weight="balanced"` |
| Pretrained / dondurulmuş katman / başlık | — |
| Kayıp / dengesizlik | Log-loss / sınıf ağırlıkları |
| Augmentation | Yok |
| Hiperparametre | sklearn varsayılanı (C=1.0); ayar yapılmadı → nested CV gerekmez |
| Tohum | 0 |
| Metrikler | Katılımcı düzeyi ROC-AUC, dengeli doğruluk; fold ort. ± SD, fold %2.5–97.5 aralığı |
| Donanım / süre | CPU, < 1 dk |
| Git commit | Repo henüz kurulmadı — ilk commit'te güncellenecek |
| Script | `scripts/audit_clinical.py` |
| Sonuç konumu | `reports/clinical_audit/EXP-001_confounder_baselines.json`, `clinical_audit.md` |

**Sonuçlar (AUC ort. ± SD, fold aralığı %2.5–97.5)**

| Kohort | Özellik | AUC | Dengeli doğr. |
|---|---|---|---|
| 344 | yaş | 0.681 ± 0.077 (0.51–0.83) | 0.624 ± 0.070 |
| 344 | cinsiyet | 0.516 ± 0.072 | 0.516 ± 0.072 |
| 344 | sigara | 0.593 ± 0.066 | 0.580 ± 0.061 |
| 344 | yaş+cinsiyet+sigara | 0.677 ± 0.083 | 0.634 ± 0.075 |
| 344 | kayıt tarihi | **0.797 ± 0.048** (0.71–0.88) | 0.734 ± 0.069 |
| 344 | katılımcı ID numarası | **0.800 ± 0.053** (0.70–0.90) | 0.694 ± 0.080 |
| zaman-örtüşen | yaş | 0.674 ± 0.081 | 0.615 ± 0.080 |
| zaman-örtüşen | cinsiyet | 0.489 ± 0.070 | 0.489 ± 0.070 |
| zaman-örtüşen | sigara | 0.653 ± 0.089 | 0.605 ± 0.084 |
| zaman-örtüşen | yaş+cinsiyet+sigara | 0.687 ± 0.090 | 0.652 ± 0.081 |
| zaman-örtüşen | kayıt tarihi | 0.567 ± 0.116 | 0.552 ± 0.079 |
| zaman-örtüşen | katılımcı ID numarası | 0.578 ± 0.116 | 0.570 ± 0.075 |

**Yorum.** Kayıt tarihi ve ona bağlı ID numarası, yayınlanmış en iyi ses modelinden (AUC 0.769) daha iyi ayırıyor. Bu, o modelin tarihi kullandığını kanıtlamaz; ama tam kohortta alınan bir ses sonucunun zaman kontrolü olmadan yorumlanamayacağını gösterir. Yaş etkisi zaman kontrolünden bağımsız sürüyor. Zaman-örtüşen alt kohortta sigara daha güçlü (0.653 vs 0.593): zamanı kontrol etmek başka bir confounder'ı öne çıkarabiliyor. Fold'dan fold'a AUC yayılımı (~0.3) tek split sonuçlarının ne kadar oynak olduğunu gösteriyor.

**Bilinen kısıtlar.** Tarih ve yaş 3–7 katılımcıda eksik (complete-case). Lojistik regresyon doğrusal ilişki varsayar; daha esnek bir model tarih için daha da yüksek AUC verebilir (eşik etkisi). Bu tablo bir alt sınırdır.

---

## EXP-002 — Klinik profil → kayıt dönemi (yalnız hastalar)
| Alan | Değer |
|---|---|
| Tarih | 2026-10-08 |
| DURUM | Tamamlandı |
| Araştırma sorusu | RQ7 — "dönem sinyali" ne kadar hasta profilinden gelir? Sesten dönem tahmininin (D-017 N1, D-020) referans çizgisi |
| Beklenti (önceden) | Tedavi basamağı ve SFT kategorisinin dönemler arasında farklı olduğu biliniyordu (χ² p ≈ 3×10⁻¹² ve 0.001) → AUC belirgin şekilde 0.5'in üstünde [HYPOTHESIS] |
| Veri sürümü | `clinical_data.csv` sha256 `cab18a35…a3af9b` |
| Katılımcı | 284 hasta (erken 100 / geç 184); sağlıklı yok (etiket sabit tutuldu) |
| Split | Katılımcı düzeyinde `RepeatedStratifiedKFold(5, 20 tekrar, random_state=0)`; hedef: geç dönem (2024-04-02 sonrası) |
| Girdi | Kategorik: tedavi basamağı, SFT tanısı, sigara, cinsiyet, GINA kontrol, inhaler steroid, alerjik rinit (one-hot, eksik = ayrı kategori); sayısal: yaş, ACT toplamı, FEV1%, FVC%, FEV1/FVC (ölçekleme fold içinde) |
| Model | Lojistik regresyon, sınıf ağırlıklı, varsayılan C |
| Script / sonuç | `scripts/audit_clinical.py` → `reports/clinical_audit/EXP-002_clinical_period_probe.json` |

| Model | AUC ort ± SD | Fold %2.5–97.5 |
|---|---|---|
| Tüm klinik profil | **0.855 ± 0.048** | 0.76–0.93 |
| Yalnız tedavi basamağı | 0.744 ± 0.048 | 0.65–0.82 |
| Tedavi basamağı hariç tümü | 0.699 ± 0.068 | 0.58–0.83 |
| Yalnız SFT tanısı | 0.620 ± 0.061 | 0.50–0.73 |
| Yalnız sigara | 0.591 ± 0.063 | 0.46–0.70 |
| Yalnız ölçülen spirometri (FEV1%, FVC%, oran) | 0.536 ± 0.066 | 0.40–0.65 |

Tedavi basamağı × dönem (erken / geç): 1: 2/1 · 2: 1/32 · 3: 11/68 · **4: 81/66** · 5: 4/17.

**Yorum.** Hasta profili dönemler arasında güçlü biçimde değişmiş (AUC 0.855). Ancak değişim **kodlanmış/kategorik** değişkenlerde (tedavi basamağı, SFT kategorisi, ACT) yoğun; **ölçülen** akciğer fonksiyonu ise neredeyse hiç ayırmıyor (0.536). Bu, iki dönemde farklı bir hasta kaynağı ya da farklı bir kodlama pratiği olduğunu düşündürüyor. [INFERENCE] Sonuç: sesten yapılacak bir dönem tahmini yüksek çıkarsa bu tek başına "kayıt koşulları değişti" demek değildir. Doğru test, sesin klinik profilin **ötesinde** dönem bilgisi taşıyıp taşımadığıdır (D-020).

**Bilinen kısıtlar.** Yalnız hastalar; kategorik kodlama dönemler arasında değişmiş olabilir (veri ekibine soruldu); doğrusal model.

---

## AUD-001 — Ses verisi denetimi (deney değil, denetim kaydı)
| Alan | Değer |
|---|---|
| Tarih | 2026-10-08 (Colab, CPU) |
| Script | `scripts/audit_audio.py` (test: `tests/test_audit_audio.py`) |
| Çıktılar | `reports/audio_audit/audio_audit.{md,json}`; Drive `data_derived/{audio_inventory,recording_map,near_duplicate_pairs}.csv` |
| Sonuç | 2 393 dosya (beklenen 2 408), 342 sesli katılımcı (283 / 59), 341'i 7/7; AAC-LC mono, 48 kHz (1 dosya 44.1 kHz); kopya yok; konum etiketi yok; iki kodlama zinciri (Apple 1 759 / FFmpeg 634); Apple dosyalarında dosya tarihi = CSV tarihi %98.7 |
| Zincir ayrıntısı (5. tur) | Zincir dosya düzeyinde karışık (285 katılımcıda ikisi birlikte); FFmpeg dosyaları yeniden kodlanmış (bitrate medyan 159 vs 185 kbps, bant genişliği çoğunlukla 24 vs 12–14 kHz); en düşük kesim 11.27 kHz |
| Confounder bulguları | Günün saati ↔ etiket AUC 0.13–0.17 (güçlü; D-022); hastalarda dönemle süre/sessizlik/saat AUC 0.35–0.36 / 0.63–0.64; kodlama zinciri dönemle ilişkili (p = 6×10⁻¹⁰), etiketle değil (p = 0.22) |
| Yorum | Teknik olarak temiz veri; zamansal confounder'ın akustik yolu daha çok kayıt prosedürü ve günün saati, kodlayıcı değil. Ayrıntı: rapor 2.3 ve 6.7 |

---

## EXP-003 — Kayıt bağlamı baseline'ları ve tasarım dengesi (sesi kullanmadan)
| Alan | Değer |
|---|---|
| Tarih | 2026-10-08 |
| DURUM | Tamamlandı |
| Araştırma sorusu | RQ7 — kayıt bağlamı (tarih, günün saati) etiketi ne kadar ele veriyor; hangi değerlendirme tasarımı onu dengeliyor? |
| Beklenti (önceden) | AUD-001'de dosya düzeyinde saat ↔ etiket AUC 0.13–0.17 → katılımcı düzeyinde güçlü bir saat etkisi [HYPOTHESIS] |
| Veri | `participants.csv` + `audio_inventory.csv` (Drive); 342 sesli katılımcı, saat 340'ında (seans başlangıcı = en erken zaman damgalı dosya) |
| Split | Katılımcı düzeyinde `RepeatedStratifiedKFold(5, 20, random_state=0)` |
| Model | Lojistik regresyon (sınıf ağırlıklı); saat = saat + saat² |
| Script / sonuç | `scripts/analyze_recording_context.py` → `reports/recording_context/` (katılımcı ID'si içermez) |

| Özellik | Ses kohortu (282/58) | Zaman penceresi (97/57) | Yalnız öğleden sonra (86/46) |
|---|---|---|---|
| Saat | 0.841 ± 0.061 | 0.873 ± 0.053 | 0.767 ± 0.095 |
| Tarih | 0.806 ± 0.055 | 0.558 ± 0.118 | 0.862 ± 0.059 |
| **Saat + tarih** | **0.931 ± 0.032** | 0.881 ± 0.055 | 0.912 ± 0.051 |
| Yaş | 0.681 ± 0.076 | 0.671 ± 0.080 | 0.665 ± 0.091 |
| Yaş + saat + tarih | 0.933 ± 0.034 | 0.908 ± 0.044 | 0.912 ± 0.052 |

Tasarım dengesi (tabakalı AUC; 0.5 = denge) ve kesinlik: rapor Bölüm 6.9. En dengeli tasarım E2h (65/50; saat 0.378, tarih 0.375, yaş 0.704; ±0.125); E2 (saat 0.125) ve E3 (saat 0.103) saat açısından dengesiz.

**Yorum.** Kayıt bağlamı tek başına etiketi AUC 0.93 ile ayırıyor; bu, herhangi bir ses modelinin "yüksek" sonucunu tek başına anlamsız kılıyor. Birincil test buna göre revize edildi (D-023): ses skoru, yaş + saat + tarih ötesinde bilgi ekliyor mu?

**Bilinen kısıtlar.** Saat yalnız Apple zincirindeki zaman damgalarından (2 katılımcıda yok); saat kısmen gerçek fizyoloji olabilir (sabah sesi, astım ritmi, bronkodilatör sonrası kayıt — veri ekibine soruldu); doğrusal model.

**EXP-003 eki (6. tur) — her tasarımın "yalnız bağlam" referansı.** Bağlam-only modelin OOF skoru (5-fold × 20) her tasarımın tabakaları içinde değerlendirildi (rapor Bölüm F, `recording_context.md` F). E2h'de: yalnız yaş **0.703**, saat + tarih **0.351**, yaş + saat + tarih 0.497. → Ham E2h AUC'sinin şans çizgisi 0.5 değil ve kestirmenin türüne göre değişiyor; E2h testi koşullu lojistik regresyona çevrildi (D-025).

---

## EXP-004 — Günün saatinin kaba akustik izi (etiket içinde) + EXP-004b klinik profil
| Alan | Değer |
|---|---|
| Tarih | 2026-10-08 |
| DURUM | Tamamlandı |
| Araştırma sorusu | RQ7 — Etiket sabitken, sabah ve öğleden sonra kayıtları kaba ses ölçümleriyle ayrılabiliyor mu? (b) Sabah ve öğleden sonra hastaları klinik olarak farklı mı? |
| Beklenti (önceden) | Oda koşulları gün içinde değişiyorsa (klinik yoğunluğu, klima) gürültü tabanı veya seviye ayırır [HYPOTHESIS] |
| Veri | `audio_inventory.csv` denetim ölçümleri (katılımcı ortalaması) + `participant_context.csv`; öğleden sonra = seans başlangıcı ≥ 12:00 |
| Model | (a) Tek ölçüm AUC + 500 permütasyon (eşik %0.3–99.7, ~11 ölçüm için kaba Bonferroni); (b) tüm ölçümler, lojistik regresyon, 5-fold × 20 |
| Script / sonuç | `scripts/analyze_recording_context.py` Bölüm E, E-b → `reports/recording_context/recording_context.{md,json}` |

| Analiz | n (öğleden sonra) | CV AUC ort ± SD | Şanstan ayrılan tekil değişken |
|---|---|---|---|
| Hastalar: ses ölçümleri → öğleden sonra | 282 (86) | 0.549 ± 0.067 | yok |
| Sağlıklılar: ses ölçümleri → öğleden sonra | 58 (46) | 0.525 ± 0.179 | yok |
| **EXP-004b** hastalar: klinik profil → öğleden sonra | 282 (86) | 0.528 ± 0.071 (fold %2.5–97.5: 0.37–0.64) | yalnız dönem (p = 0.023, düzeltmesiz) |

**Yorum.** **Olumsuz sonuç.** Kaba, ölçülebilir bir "sabah/öğleden sonra odası" farkı yok. Bu yüzden harmonizasyon saat confounding'ine karşı kozmetik kalır (D-026). Sabah ve öğleden sonra hastaları klinik olarak benzer. Bu sayede etiket içi saat kontrastı (Spisak kısmi testi, N5) temiz bir kontrol olarak kullanılabilir; tarih yine kovaryat olarak eklenir.

**Bilinen kısıtlar.**
- Kaba ölçümler ince spektral veya prozodik izleri görmez → derin gömmeler EXP-020'de test edilecek.
- Sağlıklılarda sabah kaydı yalnız 12 kişi → güç çok düşük.
- Öğleden sonra sınırı 12:00 (ikili).

---

## SIM-001 — T1'in gücü ve yanlış pozitif riski (ses kullanmadan, gerçek bağlamla)
| Alan | Değer |
|---|---|
| Tarih | 2026-10-08 |
| DURUM | Tamamlandı (tasarım simülasyonu; model deneyi değil) |
| Soru | Bağlamdan bağımsız sinyal varsa T1 onu yakalar mı (güç)? Ses yalnız bağlamı kodluyorsa T1 yanılır mı (yanlış pozitif)? |
| Veri | Gerçek etiket, saat, tarih, yaş, kayıt günü (339 kişi: 282/57; 65 gün). Ses skoru yapay: `γ·astım + 1·kestirme(bağlam) + N(0,1)` |
| Senaryolar | S1 düzgün kestirme (doğru belirtilmiş), S2 basamak kestirme ("sabah mı", "geç dönem mi"), S3 ölçülmemiş güne özgü etki |
| Testler | T1 kuadratik bağlam; T1 spline (4 df) bağlam; T1 spline + gün-kümeli SE; E2h koşullu lojistik regresyon |
| Simülasyon | Hücre başına 500, tohum 0 |
| Script / sonuç | `scripts/simulate_t1_power.py` → `reports/recording_context/SIM-001_t1_power.{md,json}` |

Sonuç tablosu ve yorum: `docs/CONFOUND_CONTROL_DESIGN.md` Bölüm 2.4.

| Bulgu | Sayı |
|---|---|
| Kuadratik bağlam modelinde yanlış pozitif (S2, basamak kestirme) | **0.216** (spline 0.060, E2h-KLR 0.060) |
| Güne özgü ölçülmemiş etkide yanlış pozitif (S3) | spline 0.122, gün-kümeli 0.110, E2h-KLR 0.142; p < 0.01'de 0.034 |
| %80 güç için minimum aynı-bağlam AUC'si | ≈ 0.69 (α = 0.05), ≈ 0.74 (α ≈ 0.01) |
| Saptanabilir sinyalde ortalama ΔAUC | 0.002–0.007 |

**Yorum.**
- T1'in bağlam modeli spline olmalı.
- ΔAUC karar istatistiği olamaz.
- Bilgi örtüşme bölgesinde: tam kohort T1 gücü ≈ E2h-KLR gücü.
- Güne özgü bağlam kalıntı bir yanlış pozitif riski.

Bunların hepsi D-025'e girdi.

**Bilinen kısıtlar.**
- Ses skoru yapay: tek boyutlu, normal gürültülü. Gerçek güç daha düşük olacak.
- Kestirme gücü λ = 1 sabit.
- Spline df (4) önceden sabit.

---

## EXP-010 — MFCC: yayınlanmış çalışmanın sadık yeniden üretimi
| Alan | Değer |
|---|---|
| Tarih | 2026-10-09 (planlandı) |
| DURUM | **Tamamlandı** (2026-10-09, Colab CPU; scikit-learn 1.6.1, librosa 0.11.0) |
| Araştırma sorusu | RQ1 tabanı — Makalenin tarifi ve protokolü bizim veride aynı AUC aralığını (≈0.65–0.77) veriyor mu? |
| Beklenti (önceden) | Makale kuralıyla seçilen "en iyi" modeller 0.65–0.77 aralığında. 14 modelin medyanı bundan ~0.03–0.06 düşük (seçim iyimserliği). Fold SD ~0.05–0.10 [HYPOTHESIS] |
| Veri | Orijinal `.m4a` → 22.05 kHz → `librosa.effects.trim(top_db=60)` → 44 özellik (D-031); 342 katılımcı, görev başına 342 (görev 4: 341) |
| Split | Görev başına `StratifiedKFold(5, shuffle=True, random_state=42)`, katılımcı düzeyinde (makale) |
| Model | 14 model (makale), varsayılan ayarlar; fold içinde StandardScaler → SMOTE |
| Metrikler | Fold AUC ort ± SD, havuzlanmış OOF AUC + bootstrap CI, dengeli doğruluk, duyarlılık/özgüllük, doğruluk, F1 (makaleyle karşılaştırma için) |
| Script / sonuç | `scripts/run_mfcc_baselines.py --exp EXP-010` → `reports/mfcc/EXP-010_mfcc.{md,json}`; OOF Drive `experiments/EXP-010_mfcc-paper/` |

**Yorum kuralı (önceden):**
- Aralığa ulaşılırsa: veri ve özellik tarifi makaleyle tutarlı.
- Ulaşılamazsa: farkın olası nedenleri yazılır (bilinmeyen kırpma eşiği, 2 eksik katılımcı, kütüphane sürümleri). Bu da bir sonuçtur.
- Her durumda sonuç bir **üst sınırdır** (D-028).

**Sonuçlar** (342 katılımcı; görev başına 342, görev 4: 341; tam tablo `reports/mfcc/EXP-010_mfcc.md`):

| görev | makale kuralıyla "en iyi" (AUC ort ± SD) | 14 modelin medyanı | makale Tablo 4 / Tablo 2 |
|---|---|---|---|
| aaa | MLP 0.725 ± 0.051 | 0.614 | 0.700 / 0.649 |
| araba | LR 0.686 ± 0.036 | 0.663 | 0.656 / 0.704 |
| ana | XGBoost 0.687 ± 0.088 | 0.664 | 0.690 / 0.650 |
| ordu | SVM 0.677 ± 0.078 | 0.622 | 0.674 / 0.686 |
| gelecek | Stacking 0.719 ± 0.086 | 0.684 | 0.769 / 0.717 |
| titiz | MLP 0.695 ± 0.074 | 0.642 | 0.723 / 0.736 |
| ünlem | MLP 0.723 ± 0.049 | 0.682 | 0.749 / 0.726 |
| **ortalama** | **0.702** | 0.653 | 0.709 / 0.695 |

**Yorum.**
- **Yeniden üretim başarılı.** "En iyi" modellerin ortalaması 0.702; makalede 0.709 (Tablo 4) ve 0.695 (Tablo 2). Görev başına değerler makalenin aralığında. Veri ve özellik tarifi makaleyle tutarlı. [FACT]
- **Kazanan modeller makaledekilerden farklı:** makalede çoğunlukla Voting, bizde MLP (3 görev), Stacking, XGBoost, LR ve SVM. Bu, "en iyi model" etiketinin kararsız olduğunu gösteriyor. [FACT]
- **Duyarlılık / özgüllük örüntüsü makaleyle aynı:** eşik 0.5'te yüksek duyarlılık, düşük özgüllük (ör. SVM 0.96 / 0.08). Dengesiz veride 0.5 eşiğinin sonucu.
- EXP-012'ye göre seçim iyimserliği ortalama ~0.02. Tek bir 5-fold bölünmesinin şansı ise görev başına ±0.05 (2 SD).
- **Üst sınır (D-028):** Aynı katılımcılarda yalnız bağlam AUC 0.93.

---

## EXP-011 — MFCC: aynı özellikler, bizim protokolümüz (RQ1 tabanı)
| Alan | Değer |
|---|---|
| Tarih | 2026-10-09 (planlandı) |
| DURUM | **Tamamlandı** (2026-10-09, Colab CPU) |
| Araştırma sorusu | RQ1, RQ4, RQ5, RQ6 — Seçim iyimserliği ve tek-split şansı olmadan MFCC özellikleri ne kadar ayırıyor? Hangi görev, ve görev füzyonu yardımcı oluyor mu? |
| Beklenti (önceden) | [HYPOTHESIS] Bu deneyde makalenin iki iyimserlik kaynağı yok (tek split, test sonucuna göre seçim). Beklenenler: LR görev başına EXP-010'un "en iyi"sinden düşük (~0.60–0.70); füzyon tek görevlerden yüksek; hepsi bağlam referansının (~0.93) belirgin altında; yaş referansı ~0.68 |
| Veri | Harmonize önbellek (D-030) → 22.05 kHz → 44 özellik (D-031) |
| Split | `outer_r0–r4` (D-029): 5 tekrar × 5 fold = 25 dış fold; iç 5-fold yalnız düzenlileştirme seçimi |
| Model | **LR (birincil)**, SVM-RBF, GradientBoosting; sabit ızgaralar, `neg_log_loss`; sınıf/örnek ağırlığı |
| Referans (aynı fold'lar) | Yaş, bağlam (saat + saat² + tarih), yaş + bağlam — sesi kullanmayan LR |
| Metrikler | Fold AUC ort ± SD (25 fold) ve aralık; tekrar-ortalamalı OOF'tan havuzlanmış AUC + %95 bootstrap CI; dengeli doğruluk |
| Script / sonuç | `scripts/run_mfcc_baselines.py --exp EXP-011` → `reports/mfcc/EXP-011_mfcc.{md,json}`; OOF Drive `experiments/EXP-011_mfcc-protocol/` |

**Yorum kuralı (önceden):**
- Derin modellerle karşılaştırma bu deneyin **LR füzyon** ve **LR görev başına** sonuçlarıyla yapılır. Yöntem: aynı fold'larda eşleştirilmiş ΔAUC (D-011).
- Bütün sayılar üst sınırdır. Bağlamın etkisi değerlendirme aşamasında aynı OOF tahminleri üzerinde ölçülür (D-028).

**Sonuçlar** (fold AUC ort ± SD, 25 fold; havuzlanmış OOF AUC [%95 CI]; tam tablo `reports/mfcc/EXP-011_mfcc.md`):

| | LR (birincil) | SVM-RBF | GradientBoosting |
|---|---|---|---|
| aaa | 0.647 ± 0.087 · 0.648 [0.575–0.718] | 0.541 ± 0.133 | 0.580 ± 0.083 |
| araba | 0.691 ± 0.059 · 0.698 [0.619–0.770] | 0.674 ± 0.065 | 0.638 ± 0.064 |
| ana | 0.681 ± 0.075 · 0.690 [0.615–0.762] | 0.666 ± 0.074 | 0.635 ± 0.044 |
| ordu | 0.649 ± 0.092 · 0.649 [0.576–0.723] | 0.664 ± 0.073 | 0.614 ± 0.081 |
| gelecek | 0.647 ± 0.066 · 0.645 [0.566–0.720] | 0.672 ± 0.061 | 0.624 ± 0.089 |
| titiz | 0.665 ± 0.075 · 0.671 [0.588–0.748] | 0.702 ± 0.077 | 0.670 ± 0.077 |
| ünlem | 0.639 ± 0.065 · 0.644 [0.575–0.712] | 0.679 ± 0.083 | 0.611 ± 0.079 |
| **füzyon (7 görev)** | **0.771 ± 0.078 · 0.778 [0.713–0.836]** | 0.773 ± 0.061 | 0.728 ± 0.068 |
| *referans: yaş* | 0.682 ± 0.046 · 0.679 [0.601–0.750] | | |
| *referans: bağlam (saat + tarih)* | 0.934 ± 0.022 · 0.931 [0.899–0.958] | | |

**Yorum.**
- **Tek kayıt ≈ yaş.** Tek bir kaydın MFCC özellikleri (LR 0.64–0.69) etiketi, yalnız yaşı bilmekten (0.68) daha iyi ayırmıyor. [FACT]
- **Füzyon belirgin biçimde daha iyi.** Yedi kaydın ortalaması her tek görevden +0.08 ile +0.13 daha iyi; bütün bootstrap CI'ları 0'ı dışlıyor (EXP-012 E).
  - Ama bir kişinin 7 kaydı aynı seansta alınıyor. Ortalama almak hem hastalık bilgisini hem seans / bağlam bilgisini ve konuşmacı özelliklerini güçlendirir. Füzyonun kazancı tek başına "daha fazla astım bilgisi" anlamına gelmez. [INFERENCE]
- **Füzyon yaştan biraz iyi, ama belirsiz:** +0.095 [+0.008, +0.185]; Nadeau–Bengio p = 0.06.
- **Bağlam açık ara önde:** füzyondan 0.15 daha iyi [0.09, 0.22]. Bütün sayılar üst sınırdır (D-028).
- **Görevler arasında anlamlı sıralama yok.** CI'lar büyük ölçüde örtüşüyor. EXP-012'ye göre görev sıralaması girdi ve protokolle yer değiştiriyor (ör. aaa EXP-010'da en iyi, burada en düşüklerden). RQ4 / RQ5 bu örneklemle tek-görev sayılarından cevaplanamaz. [FACT + INFERENCE]
- **İkincil modeller:** SVM-RBF ve GB tek görevlerde LR'yi tutarlı biçimde geçmedi. SVM-RBF aaa'da kararsız çıktı (0.541 ± 0.133; fold'larda 0.28–0.71).
- **Uygulama hatası (yalnız eşik metrikleri):** SVM'de dengeli doğruluk 0.50 çıktı (duyarlılık 1.00 / özgüllük 0.00). Platt olasılıkları sınıf ağırlığını yok sayıp eğitimdeki %83 astım oranına göre kalibre ediliyor; 0.5 eşiği bu yüzden anlamsız. GB'de de kısmen aynı sorun var. **AUC etkilenmez.** D-031'deki "eşik 0.5 doğal karar sınırı" varsayımı yalnız LR için doğru çıktı. Düzeltme önerisi: D-032.
- **Beklentiyle karşılaştırma:** Önceden yazılan beklenti (LR görev başına ~0.60–0.70; füzyon tek görevlerden yüksek; hepsi bağlamın belirgin altında; yaş ~0.68) tuttu.

---

## EXP-012 — MFCC sonuçlarının keşifsel ayrıştırması (post hoc)
| Alan | Değer |
|---|---|
| Tarih | 2026-10-09 |
| DURUM | Tamamlandı — **önceden kaydedilmedi, sonuçlar görüldükten sonra tasarlandı** (hipotez üretir) |
| Soru | EXP-010 → EXP-011 farkı girdiden mi protokolden mi? Tek split ne kadar şanslı olabilir? Seçilen "en iyi" model yeni split'lerde ne alıyor? Füzyon tek görevlerden ve referanslardan ne kadar farklı? |
| Veri / kod | Colab'daki özellik dosyalarıyla aynı (sha256 `b94da5c9…`, `f2ac9f55…`); `scripts/analyze_mfcc_results.py` → `reports/mfcc/EXP-012_mfcc_analysis.{md,json}` (yerelde, scikit-learn 1.9.1) |

**Sonuçlar ve yorum.**
- **A. Yeniden üretilebilirlik:** EXP-011 LR başka bir makinede ve scikit-learn sürümünde çalıştırıldı; görev başına en büyük fark 4×10⁻⁵. [FACT]
- **B. Girdi mi, protokol mü?** (LR, 7 görev ortalaması)
  - Makale özellik + makale protokol: 0.665.
  - Makale özellik + bizim protokol: 0.664.
  - Önbellek özellik + bizim protokol: 0.660.
  - Füzyon: makale özellikleriyle 0.781, önbellekle 0.771.
  
  Yorum: **Harmonizasyon ortalamada performansı değiştirmedi**; görev başına ±0.07'ye varan kaymalar iki yönde de var ve gürültü düzeyinde. [FACT + INFERENCE]
- **C. Split şansı:** Aynı veri ve aynı model, 50 farklı 5-fold bölünmesiyle değerlendirildi.
  - Görev başına ortalama AUC'nin SD'si 0.019–0.027; en düşük ile en yüksek arası 0.08–0.14 (ordu: 0.545–0.685).
  - Makaledeki tohum 42, dört görevde dağılımın üst çeyreğine (%76–96. yüzdelik), üç görevde alt kısmına (%10–22) düştü.
  - Sonuç: tek bir 5-fold'dan gelen görev başına sayı yaklaşık ±0.05 belirsizlik taşır. [FACT]
- **D. Kazananın laneti:** Tohum 42'de seçilen kazanan aynı pipeline'la 25 fold'da yeniden değerlendirildi.
  - Ortalama düşüş 0.018; en büyüğü araba / LR, 0.061.
  - 7 kazananın 5'i 25 fold'da da (değerlendirilen 10–11 model arasında) en iyi kaldı.
  - Makale pipeline'ında MLP ve Stacking 25 fold'da 0.69–0.72 aldı.
  - Sonuç: **seçim iyimserliği mevcut ama küçük. Doğrusal olmayan modeller (SMOTE + MLP / Stacking) tek görevde LR'den biraz daha iyi olabilir.** [FACT + HYPOTHESIS]
- **E. Eşleştirilmiş karşılaştırmalar** (EXP-011 LR):
  - Füzyon − tek görev: +0.08 ile +0.13, hepsinde CI 0'ı dışlıyor.
  - Füzyon − yaş: +0.095 [+0.008, +0.185], NB p = 0.06.
  - Füzyon − bağlam: −0.154 [−0.224, −0.089].
- **F. Özellik uyumu (aynı kayıt, iki girdi):**
  - Statik MFCC 1–11 ve spektral özetlerde r ≥ 0.8.
  - **c0 (genel ses düzeyi) r = 0.23:** tepe normalizasyonu düzey bilgisini siliyor (beklenen; D-015 riski).
  - Δ ve ΔΔ ortalamaları r = 0.48–0.78: zaman ortalaması neredeyse sıfır ve kırpma sınırlarına duyarlı, yani pratikte kenar gürültüsü (A4 doğrulandı).
  - aaa'daki düşüş (0.692 → 0.647) ses düzeyi bilgisinin kaybıyla ilişkili olabilir [HYPOTHESIS]. Düzey hem hastalık hem kayıt koşulu bilgisi taşıyabilir. Kazanç `index.csv`'de saklı; değerlendirme aşamasında bakılabilir.

---

## EXP-013 — Etiket dengesizliği yöntemleri: hiçbiri / sınıf ağırlığı / SMOTE / rastgele aşırı örnekleme (keşifsel)
| Alan | Değer |
|---|---|
| Tarih | 2026-10-09 |
| DURUM | Tamamlandı — **keşifsel** (Alper'in 11. tur sorusu üzerine, sonuçlar görüldükten sonra) |
| Soru | SMOTE'u kullanmak ya da kullanmamak ayırmayı (AUC), kalibrasyonu ve eşik metriklerini nasıl değiştirir? |
| Veri / split | Önbellek özellikleri (EXP-011 ile aynı), split dosyaları 5×5 |
| Model | LR (C iç CV'de) ve MLP (varsayılan) × 4 yöntem; eşik iç 5-fold OOF'tan |
| Sonuç | `reports/mfcc/EXP-013_imbalance.md`, `docs/IMBALANCE_AND_SELECTION.md` Bölüm 3 |

**Sonuçlar** (füzyon, havuzlanmış AUC [%95 CI]):
- LR hiçbiri 0.776 [0.713–0.835], sınıf ağırlığı 0.778, SMOTE 0.768, rastgele aşırı örnekleme 0.778.
- MLP: 0.786–0.790.
- Eşleştirilmiş farklar ≤ 0.01; hepsinin CI'ı 0'ı içeriyor.
- Kalibrasyon (LR): düzeltme yokken ortalama tahmin − gerçek oran 0.000, Brier 0.134. SMOTE / ağırlık / aşırı örneklemede −0.24 ile −0.25, Brier 0.185–0.189.
- Dengeli doğruluk: 0.5 eşiğinde "hiçbiri" 0.500. İç-CV eşiğinde 0.698, yani düzeltmelerle aynı.

**Yorum.**
- SMOTE ve diğer düzeltmeler ayırmayı değiştirmiyor, kalibrasyonu bozuyor. Eşik metriklerindeki "kazanç" yalnızca eşik kaymasından geliyor (van den Goorbergh ve ark. 2022 ile tutarlı).
- Makaleyle bizim aramızdaki duyarlılık / özgüllük farkları çalışma noktası farkıdır.
- MLP füzyonu LR füzyonundan anlamlı biçimde iyi değil (D-032'nin "güçlü ikinci taban" gerekçesi zayıfladı).
- Karar önerisi: D-033.

---

## EXP-014 — Kayıt bağlamı dengesizliği SMOTE / ağırlıkla düzeltilebilir mi? (keşifsel)
| Alan | Değer |
|---|---|
| Tarih | 2026-10-09 |
| DURUM | Tamamlandı — **keşifsel** (Alper'in 11. tur sorusu üzerine) |
| Soru | Etiket × bağlam hücrelerinde dengeleme (SMOTE ya da ağırlık) daha iyi ya da bağlamdan daha az etkilenen bir taban üretir mi? |
| Veri / split | Önbellek özellikleri, split dosyaları 5×5; hücre = dönem (D-024) × sabah/öğleden sonra |
| Model | LR, füzyon. V0 sınıf ağırlığı (= EXP-011 LR); V2/V3 tüm veride hücre-içi SMOTE / hücre ağırlığı; V4/V5 yalnız erken dönem + hücre ağırlığı / hücre-içi SMOTE |
| Sonuç | `reports/mfcc/EXP-014_context_balancing.md`, `docs/IMBALANCE_AND_SELECTION.md` Bölüm 4 |

**Sonuçlar** (füzyon):

| | tüm test | yalnız erken dönem | hücre-içi | hastalarda skor: geç vs erken |
|---|---|---|---|---|
| V0 | 0.777 | 0.739 | 0.706 | 0.556 |
| V2 / V3 | 0.769 / 0.760 | 0.723 / 0.721 | 0.709 / 0.718 | 0.587 / 0.562 |
| V4 / V5 | 0.707 / 0.713 | 0.673 / 0.679 | 0.649 / 0.650 | 0.548 / 0.555 |
| ref: yaş / bağlam | 0.678 / 0.930 | 0.666 / 0.813 | 0.649 / 0.578 | — |

**Yorum.**
- **Olumsuz sonuç.** Bağlam dengeleme daha iyi bir taban üretmedi:
  - tüm veride fark yok (V2: −0.008 [−0.035, +0.018]);
  - yalnız örtüşmede eğitim AUC'yi düşürdü (V4: −0.070 [−0.118, −0.025]), çünkü verinin ~%54'ü atılıyor;
  - bağlam bağımlılığı göstergeleri değişmedi.
- Geç dönemde sağlıklı yok; SMOTE orada dengeleme yapamaz (pozitiflik).
- MFCC + LR bağlama zaten zayıf bağımlı (hastalar içinde geç / erken 0.556, sabah / öğleden sonra 0.524).
- Kaba bağlam sabitken füzyon 0.706 (yaş 0.649, bağlam 0.578). Bu, değerlendirme aşaması için bir **hipotez**: yaş ve ince bağlam ayarlanmadı; erken–sabah hücresinde yalnız 12 sağlıklı var.

---

## EXP-015 — Makalenin model seçim prosedürü için iç içe CV (keşifsel)
| Alan | Değer |
|---|---|
| Tarih | 2026-10-09 |
| DURUM | Tamamlandı — **keşifsel** (Alper'in 11. tur sorusu üzerine) |
| Soru | "14 model arasından en iyisini seç" prosedürü, seçim dış testten tamamen ayrıldığında ne verir? Seçim kararlı mı? |
| Veri / split | Makale özellikleri; makale pipeline'ı (StandardScaler → SMOTE → model); dış = split dosyaları 5×5, iç = 5-fold |
| Model | 10 model (CatBoost ve 3 ensemble hesap maliyeti nedeniyle dışarıda) |
| Sonuç | `reports/mfcc/EXP-015_nested_cv.md`, `docs/IMBALANCE_AND_SELECTION.md` Bölüm 2 |

**Sonuçlar** (görev başına AUC, 25 fold; 7 görev ortalaması):
- **İç içe CV, yani prosedürün dürüst performansı: 0.654.** Görev başına: aaa 0.693, araba 0.598, ana 0.630, ordu 0.644, gelecek 0.681, titiz 0.667, ünlem 0.664.
- Karşılaştırma: makale 0.709; EXP-010 en iyi 0.702; sonradan bakarak en iyi sabit model 0.683; 10 modelin medyanı 0.642; EXP-011 LR 0.660.
- Seçim kararsız: 7 görevin 6'sında 25 fold boyunca 5–8 farklı model seçildi.
- Füzyon (iç CV'nin seçtiği modellerle): 0.734 ± 0.087, havuzlanmış 0.759 [0.690–0.822].

**Yorum.**
- Makalenin "en iyi" sayıları ortalama ~0.05 (görev başına 0.03–0.09) iyimser.
- 14 model arasından seçim, önceden seçilmiş LR'ye göre bir şey kazandırmıyor.
- Seçim iyimserliği çıkarılınca bulgularımız makaleyle uyumlu.
- Faz 2'deki backbone / görev seçimi için aynı ders geçerli (D-033).

---

## Planlanan deneyler (ID'ler başlarken verilecek)

**Ana proje (D-012 merdiveni; D-028 ile değişmeden sürüyor):**

| Sıra | Deney | RQ | Ön koşul |
|---|---|---|---|
| 1 | ~~Ses denetimi~~ → AUD-001 tamamlandı | RQ7 | — |
| 1b | ~~EXP-003, EXP-004/004b, SIM-001~~ → tamamlandı (kayıt bağlamı belgelendi) | RQ7 | — |
| 1c | ~~Split dosyaları~~ → üretildi ve Colab'da aynı sha256 ile teyit edildi (`make_splits.py`, D-029; manifest `reports/splits/`) | altyapı | — |
| 1d | ~~Harmonize ses önbelleği~~ → Colab'da üretildi ve doğrulandı (D-030; rapor `reports/audio_cache/`) | altyapı | 1c |
| 2 | **MFCC sadık yeniden üretim (EXP-010)** — kod ve test hazır (D-031) | RQ1 | 1d |
| 3 | **MFCC, ortak protokol (EXP-011)** — kod ve test hazır (D-031) | RQ1, RQ4, RQ5, RQ6 | 1c, 1d |
| 4 | Dondurulmuş gömme + lineer prob (6 backbone × 7 görev); gömmeler saklanır (D-028) | RQ3, RQ4, RQ5 | Ses önbelleği, smoke test |
| 5 | Sıfırdan CNN10 | RQ2 | Eğitim döngüsü + checkpoint testi |
| 6 | Ham dalga formundan uçtan uca fine-tune: 4'te öne çıkan 1–2 aile + PANNs referans; Boll hiperparametreleri, iç doğrulamayla early stopping (D-009); en iyi ve son epoch tahminleri saklanır (D-028); seçim ölçütü Faz 2 sonunda yazılır | RQ2, RQ3 | 4 ve 5 tamam |
| 7 | Çok görevli füzyon, alt gruplar, ses + yaş/cinsiyet/sigara | RQ6, RQ8, RQ9 | OOF tahminleri |

Model geliştirme sırasında her sonuç **üst sınır** olarak raporlanır. Her tabloda iki referans satırı bulunur: yalnız bağlam (tarih + saat) AUC 0.931, yalnız yaş 0.681 (D-028).

**Kayıt bağlamı değerlendirmesi (D-023, D-028; model geliştirme bitince):**

| Sıra | Analiz | RQ | Ön koşul |
|---|---|---|---|
| E-1 | Kaydedilmiş tahminler üzerinde artımlı test T1, E1/E1h/E2/E3 + denge tabloları ve referans çizgileri, ayarlı E2h, D-005 baseline'ları, negatif kontroller (N1–N4), codec zinciri duyarlılığı; test ayrıntıları D-025 önerisi üzerinden kesinleşir | RQ7 | 2–7'nin saklanmış tahminleri |
| E-2 | Gerekirse: bağlam probları (EXP-020), azaltma denemeleri (EXP-022), gün-gruplu CV, fine-tune protokol varyantları (D-026, D-027) | RQ7 | E-1 sonuçları |

---

## Şablon (yeni deney için kopyala)

```markdown
## EXP-NNN — <kısa başlık>
| Alan | Değer |
|---|---|
| Tarih | |
| DURUM | Planlandı / Çalışıyor / Tamamlandı / Başarısız / İptal |
| Araştırma sorusu | RQ? — tek cümle |
| Beklenti (önceden) | Ne görmeyi bekliyoruz ve neden — [HYPOTHESIS] |
| Veri sürümü | CSV sha256 + recording_map sha256 + audio zip sha256 |
| Katılımcı / hasta / kontrol | |
| N kayıt / N segment | |
| Kayıt türleri | |
| Split stratejisi | split dosyası + sha256 |
| Girdi temsili / SR / ön işleme | |
| Model / pretrained ağırlık (dosya + sha256) | |
| Dondurulan katmanlar / başlık | |
| Kayıp / dengesizlik stratejisi / augmentation | |
| Hiperparametreler | |
| Tohum(lar) / fold sayısı | |
| Metrikler | |
| Ort. ± SD / CI / fold aralığı | |
| Eğitim süresi / donanım | |
| Git commit / notebook-script | |
| Sonuç konumu (Drive) | |

**Sonuçlar.** (tablo; N katılımcı ve N segment ayrı)
**Yorum.** (beklentiyle karşılaştır; olumsuz sonuçları da yorumla)
**Bilinen kısıtlar.**
**Sonraki adım / açılan sorular.**
```

---

## Araştırma soruları
- **Ana soru:** Ses tabanlı derin öğrenme, MFCC + klasik ML'ye göre astım/sağlıklı ayrımında daha iyi ve daha genellenebilir mi?
- RQ1 MFCC vs log-Mel/pretrained · RQ2 sıfırdan vs pretrained · RQ3 PANNs vs BEATs vs WavLM · RQ4 kayıt türüne göre performans · RQ5 sürdürülmüş ünlü vs kelimeler · RQ6 çok kaydı birleştirmek · RQ7 confounder mı astım mı · RQ8 alt grup kararlılığı · RQ9 ses, yaş/cinsiyet/sigaranın ötesinde bilgi taşıyor mu (D-006 ile yeniden tanımlandı)
