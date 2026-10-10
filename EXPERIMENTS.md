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

## SMK-001 — Faz 2 smoke test'i (6 backbone)
| Alan | Değer |
|---|---|
| Tarih | 2026-10-09 (tasarım) |
| DURUM | **Tamamlandı — 2. deneme PASS (2026-10-10).** 1. deneme FAIL (aşağıda; korunuyor). Yerel birim testleri (`tests/test_phase2.py`, rastgele küçük modeller) SMK-001 DEĞİLDİR |
| Soru | Her backbone resmi ağırlığı eksiksiz yüklüyor, resmi girdi biçimini alıyor, eval'de deterministik ve batch'ten bağımsız mı? Faz 3'te bir eğitim adımı ne kadar bellek / süre ister? |
| Beklenti (önceden) | Hepsi PASS. PANNs'te kelime kayıtlarında "Speech" ilk 3'te. Aynı-kişi benzerlik AUC'si > 0.6 (konuşmacı bilgisi güçlü kodlanır). WavLM Large batch 16 × 4 s eğitimde T4'e gradient checkpointing'siz sığmayabilir [HYPOTHESIS] |
| Kontroller | S1–S14 (`docs/PHASE2_DESIGN.md` Bölüm 9.5); S14 = dolgu yolu (tam uzunlukta `lengths` sonucu değiştirmez) |
| Veri | Harmonize önbellek (D-030), 8 katılımcı × 7 görev (deterministik örnek) |
| Ağırlıklar | PANNs Zenodo 3987831 (md5 doğrulamalı), WavLM HF (revizyon hash'i), BEATs iter3+ AS2M (OneDrive, elle) |
| Kod / notebook | `scripts/smoke_test_backbones.py`, `notebooks/20_smoke_tests.ipynb` |
| Sonuç konumu | `reports/phase2/SMK-001_smoke.{md,json}`; checkpoint sha256'ları Drive `models/MANIFEST.json` |
| 1. deneme (2026-10-09, T4, torch 2.11, transformers 5.18) | **FAIL.** (1) WavLM Base+ S2: sözleşme `return_attention_mask: false` bekliyordu, gerçek değer true → beklenti yanlıştı (bellekten yazılmış, "doğrulanmalı" işaretliydi); sözleşme düzeltildi. (2) CNN10, CNN14, CNN14_16k, BEATs S12: fp16 eğitim adımında kayıp ilk adımdan itibaren sonlu değil → neden: ön işleme fp16'da taşıyor / alt taşıyor (PANNs log-Mel amin=1e-10 → 0 → log(0); BEATs fbank ×2¹⁵ → taşma); CPU'da yeniden üretildi; düzeltme: ön işleme autocast altında fp32. WavLM Large S12 PASS (13.4 GB, 1.07 s/adım, gradient checkpointing'siz). Diğer kontroller beş modelde PASS; S9 PANNs "Speech" 1. sırada. Eşik gevşetilmedi; SMK-001 baştan çalıştırılacak. Rapor GitHub'da `phase2-smoke-results` dalında (commit `eae26bd`; main'e birleştirilmedi; olumsuz sonuç olarak korunur — dalı silme) |
| Notebook düzeltmesi (1. denemeden sonra) | Aynı notebook'u ikinci kez çalıştırınca repo hücresi `git pull` hatası verdi (repo rapor dalında, upstream yok) ve aynı adlı dala ikinci gönderim reddedilecekti. Düzeltme: repo hücresi `fetch` + güvenli `main`'e geçiş (gönderilmemiş commit ya da commit edilmemiş dosya varsa durur); rapor zaman damgalı yeni dala gider; `!` yerine hata durumunda duran `sh()`; FAIL raporu da gönderilir, karar son hücrede. Rapor artık `code_sha256` + `git_commit` içerir; notebook 21 kapısı kodun SMK-001'den sonra değişmediğini ve checkpoint sha256'larını denetler. Tasarım ve eşikler değişmedi |
| 2. deneme (2026-10-10, T4, torch 2.11, transformers 5.18, commit `568b8e3`) | **6/6 PASS** (rapor main'de, PR #15). S3 strict yükleme hepsinde PASS, beklenmeyen anahtar yok. S6 determinizm 0. S7 batch-değişmezlik göreli fark ≤ 4e-6. S8 kanca = resmi çıktı (SSL modelleri). S9: üç PANNs'te "Speech" 1. sırada. S10 aynı-kişi AUC: cnn10 0.67, cnn14 0.69, cnn14_16k 0.71, BEATs 0.83, Base+ 0.67, Large 0.70. S12 (batch 16 × 4 s, fp16, gradient checkpointing yok; tepe GB / s·adım): cnn10 0.63 / 0.063, cnn14 1.82 / 0.125, cnn14_16k 1.82 / 0.125, BEATs 3.86 / 0.316, Base+ 5.40 / 0.378, Large 13.05 / 1.004 → beklentinin aksine Large T4'e sığdı ama 15 GB'ye yakın. Base+'ta 3. adım GradScaler tarafından atlandı (sonlu olmayan gradyan; AMP'nin olağan davranışı) → Faz 3'te atlanan adım oranı izlenecek. Base+ `feat_extract_norm=group` (dolgu notu geçerli), Large `layer`. S14 kısa parça zeropad–nopad kosinüsü: PANNs 0.64–0.86 (maskesiz), BEATs 0.98, Base+ 0.89, Large 1.0 |
| Başarısızlık protokolü | FAIL varsa gömme çıkarımı ve EXP-016 başlatılmaz; neden incelenir ve buraya yazılır; eşik gevşetilerek geçirilmez; düzeltmeden sonra SMK-001 baştan çalıştırılır; başarısız rapor git geçmişinde kalır (tasarım belgesi 1.1) |

## EXP-016 — Dondurulmuş gömme + lineer prob (6 backbone vs 2 MFCC tabanı)
| Alan | Değer |
|---|---|
| Tarih | 2026-10-09 (tasarım; **sonuçlardan önce**) |
| DURUM | **Tamamlandı (2026-10-10)** — rapor `reports/frozen/EXP-016_frozen.md` (PR #16). Bütün sayılar D-028 gereği **üst sınır**: kayıt bağlamı değerlendirilmedi |
| Araştırma sorusu | RQ1 (onaylayıcı, füzyon): backbone temsili MFCC'den fazla doğrusal çözülebilir bilgi taşıyor mu? RQ3 / RQ4 / RQ5 keşifsel |
| Beklenti (önceden) [HYPOTHESIS] | B1: füzyon AUC'leri 0.70–0.85; en az biri MFCC-LR'yi (0.77) sayısal olarak geçebilir ama onaylayıcı testi geçen olmaması daha olası (kaba güç hesabı: saptanabilir fark ≈ 0.06–0.09; eşik değil, varsayımlar tasarım belgesi Bölüm 5.5). B2: bağlam izleme göstergeleri MFCC'den yüksek. B3: |cnn14 − cnn14_16k| < 0.03. B5: görevler arasında güvenilir sıralama yok. B7: Large, Base+'tan anlamlı biçimde iyi değil |
| Veri sürümü | CSV sha256 + önbellek index sha256 (D-030) + gömme sha256'ları (`embeddings_v1/*/MANIFEST.sha256`) |
| Katılımcı / hasta / kontrol | 342 / 283 / 59 (görev 4: 341) |
| N kayıt / N segment | 2 393 kayıt; ≈ 12 000 pencere (4 s / 2 s) [INFERENCE] |
| Kayıt türleri | 7 görev; birincil birim füzyon (D-032) |
| Split stratejisi | outer_r0–4 (D-029), 5 × 5, iç 5-fold |
| Girdi / SR / ön işleme | Harmonize önbellek; modelin resmi SR'si; 4 s / 2 s, son pencere sona hizalı; 17 kısa kayıt: birincil `zeropad` (D-015; BEATs/WavLM'de resmi maske), `nopad` EXP-016S'de (D-034 madde 1, onay bekliyor) |
| Model / ağırlık | cnn10, cnn14, cnn14_16k, beats (iter3+ AS2M), wavlm_base_plus, wavlm_large — hepsi dondurulmuş |
| Temsil | PANNs fc1; SSL katman 1..L z-skorlu ortalaması (D-034 madde 3) |
| Başlık / kayıp / dengesizlik | LR (class_weight=balanced); MFCC: LR + MLP (SMOTE, D-032) |
| Hiperparametreler | C ∈ 1e-5…1e2, iç 5-fold neg_log_loss; eşik iç 5-fold OOF (D-033) |
| Birim / fold'lar | Katılımcı (füzyon); bütün kollar aynı 25 dış fold'da, aynı katılımcılarda (assert) |
| İstatistik | Fold düzeyinde eşleştirilmiş ΔAUC; NB düzeltilmiş t, **tek yönlü** (H1: Δ > 0); kesişim-birleşim (LR ve MLP); Holm (6), eşik 0.025; ayrıca iki bootstrap CI'ının alt sınırı > 0; DeLong destekleyici. Keşifsel aileler içinde Holm |
| D-035 | Fold başına seçim, yalnız o fold'un eğitim verisindeki iç OOF füzyon AUC'si → `d035_selection.csv`; prosedürün dürüst tahmini ayrıca |
| Meta veri | Bağlam / süre karşılaştırmaları bu deneyde YOK → META-016 (ayrı rapor) |
| Yorum sınırı | Farklı **hazır ses temsillerinin** karşılaştırması; ön-eğitim yöntemlerinin (gözetimli / öz-gözetimli) saf karşılaştırması değil. BEATs iter3+ AudioSet etiketlerini dolaylı kullanır; modeller hedef, veri alanı, mimari, boyut ve bant genişliğinde aynı anda farklı (tasarım belgesi 9.6) |
| Kısa kayıt politikası | D-034 KABUL: birincil `zeropad`; bunun süre ipuçlarını ortadan kaldırdığı varsayılmaz (META-016, EXP-016S) |
| Regresyon testi | mfcc_lr füzyonu EXP-011 LR füzyonunu (0.771) ±0.01 içinde yeniden üretmeli |
| Kod / notebook | `scripts/extract_embeddings.py`, `scripts/run_probes.py`, `notebooks/21_frozen_embeddings.ipynb` |
| Sonuç konumu | `reports/frozen/EXP-016_frozen.{md,json}`; Drive `experiments/frozen_probe/`, `data_derived/embeddings_v1/` |

**Sonuçlar (füzyon, katılımcı düzeyi; 342 katılımcı, 2 393 kayıt, 11 170 pencere; 25 dış fold).** AUC fold ort. ± SD · havuzlanmış [%95 CI]:

| kol | AUC | havuzlanmış | onaylayıcı karar (vs mfcc_lr ve mfcc_mlp; Holm p) |
|---|---|---|---|
| ref_context (yalnız tarih + saat) | 0.934 ± 0.022 | 0.931 [0.899–0.958] | referans, test edilmez |
| beats | 0.921 ± 0.046 | 0.923 [0.886–0.953] | **destekleniyor** (ΔAUC +0.150 / +0.151; p 0.001) |
| wavlm_large | 0.909 ± 0.041 | 0.909 [0.868–0.942] | **destekleniyor** (+0.138 / +0.139; p 0.003) |
| wavlm_base_plus | 0.884 ± 0.039 | 0.886 [0.836–0.926] | **destekleniyor** (+0.113 / +0.114; p 0.006) |
| cnn10 | 0.855 ± 0.047 | 0.858 [0.807–0.903] | desteklenmiyor (+0.084 / +0.085; p 0.067; bootstrap CI'lar 0'ı dışlıyor ama Holm geçmedi) |
| cnn14 | 0.791 ± 0.063 | 0.811 [0.747–0.869] | desteklenmiyor (+0.020 / +0.022; p 0.649) |
| mfcc_lr | 0.771 ± 0.078 | 0.778 [0.713–0.836] | taban (EXP-011'i birebir yeniden üretti: regresyon testi GEÇTİ) |
| mfcc_mlp | 0.770 ± 0.058 | 0.788 [0.725–0.844] | taban |
| cnn14_16k | 0.729 ± 0.067 | 0.733 [0.666–0.794] | desteklenmiyor; iki tabandan da düşük (nokta tahmini; −0.042 / −0.040) |
| ref_age (yalnız yaş) | 0.682 ± 0.046 | 0.679 [0.601–0.750] | referans |

- **Keşifsel çiftler** (Holm 15 içinde): beats − wavlm_large +0.012 (havuz CI [−0.010, +0.038], Holm p 0.40) → ikisi arasında fark kanıtı yok. wavlm_large − base+ +0.025 (Holm 0.19). cnn10 − cnn14 +0.063 (Holm 0.087). cnn14_16k üç SSL modelinden ve cnn10'dan düşük (Holm ≤ 0.004).
- **Görev başına** (keşifsel): SSL modellerinin kazancı kelimelerde büyük (BEATs / Large: MFCC-LR'ye göre +0.13…+0.23), ünlüde (/aaa/) küçük. Ünlü − kelime ortalaması: Base+ −0.168 (Holm 0.001), Large −0.124 (Holm 0.008), BEATs −0.095 (Holm 0.14); PANNs ve MFCC'de ≈ 0.
- **D-035 fold başına seçim:** BEATs 24/25 fold, WavLM Large 1/25 (tekrar 3, fold 1). Prosedürün iç içe tahmini 0.920 ± 0.047; sonradan en iyi sabit kol (BEATs 0.921) ile fark +0.001 → kazananın laneti ihmal edilebilir.
- **Eşik metrikleri** (iç-CV eşiği): BEATs dengeli doğruluk 0.81 ± 0.08 (duyarlılık 0.82 / özgüllük 0.81). Test fold'unda ~12 sağlıklı → özgüllük çok gürültülü.
- **Kalibrasyon:** LR kollarında eğim 1.9–3.2 ve ortalama tahmin oranın altında → `class_weight=balanced`'ın beklenen etkisi (olasılıklar 50/50 önsele kayar) [INFERENCE]; AUC'yi etkilemez.
- **C ızgara ucu:** cnn14 %12, cnn14_16k %16 fold'da ızgaranın ucu seçildi → bu iki kolun AUC'si biraz eksik tahmin edilmiş olabilir [INFERENCE]; ızgara sonradan değiştirilmedi.
- **mfcc_mlp ConvergenceWarning:** D-032'nin önceden sabitlediği "varsayılan MLPClassifier" (makale pipeline'ı) 200 iterasyonda durur; uyarı beklenen davranış. Sonuç EXP-013 ile tutarlı (havuz 0.788).

**Önceden yazılmış beklentilerle karşılaştırma** (`docs/PHASE2_DESIGN.md` Bölüm 10): B1 **yanlış** (AUC'ler 0.73–0.92; üç backbone onaylayıcı testi geçti). B2 **kısmen** (yalnız "sağlıklılarda sabah vs öğleden sonra" göstergesinde backbone'lar MFCC'den yüksek). B3 **yanlış** (cnn14 − cnn14_16k +0.062). B4 **kısmen** (WavLM tepesi ortada değil, alt-orta katmanlarda). B5 **kısmen yanlış** (WavLM'de ünlü kelimelerden güvenilir biçimde kötü). B6 çoğunlukla doğru. B7 doğru. B8 doğru.

**Yorum.**
- [FACT] En iyi ses temsili (BEATs 0.921), yalnız kayıt bağlamını (tarih + saat) kullanan modeli (0.934) geçmiyor. Bu tabloda sesin bağlamın ötesinde astım bilgisi taşıdığına dair kanıt **yok**; bu soru D-023/D-025'in T1 testinin sorusu ve henüz yapılmadı (D-028).
- [INFERENCE] "Backbone > MFCC" kararı istatistiksel olarak sağlam (üç SSL modeli, iki taban, Holm). Ama bu üstünlüğün astım bilgisinden mi yoksa bağlamı (oda / gün / saat izi) daha iyi kodlamaktan mı geldiği ayrılamaz.
- [HYPOTHESIS] BEATs'in (AudioSet; genel ses ve ortam olayları) en yüksek AUC'yi alması iki biçimde açıklanabilir: ses kalitesini daha iyi kodlaması ya da akustik ortam izini daha iyi kodlaması. META-016'daki zayıf işaret ikinciyle uyumlu ama belirsizlik çok büyük.
- Olumsuz sonuçlar: CNN14 CNN10'dan kötü (dondurulmuş, keşifsel; Holm 0.087); CNN14_16k iki MFCC tabanının da altında; CNN10 onaylayıcı testi geçemedi.

**Bilinen kısıtlar.**
- D-028 (bağlam değerlendirmesi yapılmadı).
- Fold'lar bağımsız değil (NB düzeltmesi sezgisel).
- Sağlıklı grup küçük (59).
- EXP-017 yalnız tekrar 0.
- Kilitli test seti yok (D-009).

## EXP-016S — Kısa kayıt dolgu politikası duyarlılığı (önceden belirlenmiş)
| Alan | Değer |
|---|---|
| DURUM | **Tamamlandı (2026-10-10).** Bütün kollarda \|Δ füzyon AUC\| ≤ 0.007 (Holm p = 1); WavLM Large'da tam 0 → **B8 doğru**. Etkilenen 17 katılımcıda ort. \|Δp\|: PANNs 0.030–0.036, Base+ 0.025, BEATs 0.003, Large 0. Birincil `zeropad` sonuçları politikaya duyarlı değil |
| Soru | 17 kısa kayıtta diğer politika (birincil zeropad ise nopad) kullanılsaydı füzyon sonuçları değişir miydi? |
| Beklenti (önceden) [HYPOTHESIS] | B8: \|Δ\| < 0.01 (etkilenen kayıt %0.7) |
| Tasarım | 6 backbone, birincil temsil, tekrar 0; eşleştirilmiş Δ (alternatif − birincil). Seçimde kullanılmaz |

## META-016 — Meta veri / confounder izleme (AYRI rapor; ana akışa girmez)
| Alan | Değer |
|---|---|
| DURUM | **Tamamlandı (2026-10-10)**; betimsel, karar için kullanılmaz |
| Sonuç: bağlam göstergeleri | Etiket sabitken skorun bağlamı ayırma AUC'si (0.5 = bağımlılık yok). **Sağlıklılarda sabah (12) vs öğleden sonra (46):** bütün backbone'lar 0.59–0.70 (BEATs 0.697 en yüksek, Large 0.589 en düşük), MFCC 0.46–0.47 → sabah kaydedilen sağlıklılar daha "astımlı" skor alıyor. Hastalarda sabah/öğleden sonra (0.47–0.50) ve geç/erken dönem (0.48–0.55) göstergelerinde MFCC'den farklı değil. Belirsizlik büyük: 12 × 46 karşılaştırmada AUC'nin SE'si ≈ 0.09 (Hanley–McNeil, yaklaşık) → %95 aralık ≈ ±0.18 [INFERENCE] |
| Sonuç: süre | Görev × etiket medyan süreler 10.1–10.4 s, gruplar arasında benzer; yalnız-süre referansı AUC 0.497 ± 0.080 (havuz 0.449 [0.354–0.543]) → süre etiketi taşımıyor. Kısa kaydı olan 17 katılımcı çıkarıldığında füzyon AUC değişimi ≤ 0.015 |
| İçerik | (1) EXP-014'ün üç bağlam bağımlılığı AUC'si, her kol ve referanslar; (2) görev × etiket süre dağılımı ve kısa kayıt sayıları; (3) yalnız-süre referans çizgisi (D-005; aynı fold'lar); (4) önceden belirlenmiş duyarlılık: kısa kaydı olan katılımcılar değerlendirmeden çıkarıldığında füzyon AUC'leri |
| Kural | Hiçbir sayı onaylayıcı karara, D-035 seçimine ya da katman seçimine girmez. Asıl bağlam değerlendirmesi model geliştirme bitince (D-028) |
| Kod | `scripts/report_metadata_monitor.py` → `reports/frozen/META-016_metadata_monitor.{md,json}` |

## EXP-017 — Katman katman prob (BEATs, WavLM; keşifsel)
| Alan | Değer |
|---|---|
| DURUM | **Tamamlandı (2026-10-10)**; yalnız tekrar 0, 5 fold, keşifsel. BEATs: L0 0.64, L1 0.75, L2 0.85 → L5'ten itibaren ≈ 0.92 düzlüğü (L10 0.927); katman ortalaması 0.923. Base+: L0 zaten 0.86, tepe L3–L4 ≈ 0.90, L8'de 0.84'e iner, sonda 0.87; ortalama 0.888. Large: L0 0.84, tepe L4 0.917, sonra 0.89–0.905; ortalama 0.913. Fold SD 0.03–0.07 → komşu katman farkları yorumlanmaz. **B4 kısmen:** WavLM tepesi alt-orta katmanlarda; eşit katman ortalaması en iyi katmanlarla aynı düzeyde (bilgiyi seyreltmiyor). [INFERENCE] BEATs L0'ın düşüklüğü kısmen havuzlamadan: ilk katmanda yamalar karışmadığı için zaman + frekans yamalarının ortalaması spektral düzeni siliyor |
| Soru | SSL modellerinde astım / sağlıklı bilgisi hangi katmanlarda? Eşit katman ortalaması (birincil temsil) bilgiyi seyreltiyor mu? |
| Beklenti (önceden) [HYPOTHESIS] | B4: WavLM'de eğri ortada tepe yapar; son katman ortalamadan kötü |
| Tasarım | Yalnız tekrar 0; her katman (0..L) ayrı kol; aynı prob protokolü. **Hiçbir seçimde kullanılmaz** |

## EXP-018 — Tüm kayıt tek girdi vs 4 s pencereler (keşifsel)
| Alan | Değer |
|---|---|
| DURUM | **Tamamlandı (2026-10-10).** Tüm kayıt − 4 s pencere (füzyon, 25 fold): BEATs −0.011, Base+ −0.004, Large −0.003, cnn10 −0.027, cnn14 +0.014, cnn14_16k +0.035; hepsinde Holm p ≥ 0.46. **B6 çoğunlukla doğru** (cnn10 ve cnn14_16k nokta tahminleri 0.02'yi aşıyor ama CI'lar 0'ı içeriyor). Faz 3 için 4 s pencere girdisi bilgi kaybettirmiyor |
| Soru | Kayıtlar ~10 s ve PANNs / BEATs 10 s'lik AudioSet klipleriyle eğitildi. Tüm kaydı tek girdi olarak vermek, Boll'un 4 s pencerelemesinden farklı sonuç veriyor mu? |
| Beklenti (önceden) [HYPOTHESIS] | B6: |Δ| < 0.02 |
| Tasarım | 6 backbone, birincil temsil, 5 tekrar; eşleştirilmiş Δ (NB, bootstrap). Faz 3 girdi tasarımına bilgi verir; Faz 2'de seçim için kullanılmaz |

## EVAL-016 — Dondurulmuş aile için bağlam değerlendirmesi (T1 ve lens seti; onaylayıcı)
| Alan | Değer |
|---|---|
| Tarih | 2026-10-10 (plan; **E1 sonuçlarından sonra, bağlam testlerinden önce**) |
| DURUM | Planlandı — D-025 (içerik değişmeden) ve D-037 onayı bekliyor; kod yazılmadı |
| Araştırma sorusu | D-023 / RQ7 (onaylayıcı): ses, ölçülen bağlam (saat, tarih) ve yaşın ötesinde astım bilgisi taşıyor mu? Ayrıca C2: model bağlamı kullanıyor mu? |
| Beklenti (önceden) [HYPOTHESIS] | Bağlamın etiketi neredeyse belirlemesi ve sağlıklı n'in küçük olması nedeniyle T1'in gücü sınırlı (SIM-001: %80 güç için aynı-bağlam AUC'si ≈ 0.69 / Holm düzeyinde ≈ 0.74). SSL backbone'larının en az birinde T1 pozitif çıkması da, hiçbirinde çıkmaması da makul. META-016'nın zayıf işareti nedeniyle C2'nin (bağlam kullanımı) en az bir backbone'da pozitif çıkması daha olası |
| Veri | EXP-016 `partial/` dış-test tahminleri (8 kol × 5 tekrar), `participant_context.csv`, `embeddings_v1` (G1 / N5) |
| Kohort | EXP-016 kohortu ∩ saat, tarih ve yaşı tam (beklenen 339; çalıştırmada sayılır) |
| Birincil test | T1: `y ~ cr(saat, 4) + cr(tarih, 4) + yaş + ŝ`, ŝ için LR testi; Holm (6 backbone), α 0.05, OR > 1 |
| Destekleyici | Bootstrap OR CI (B 2000); gün-kümeli SE; cr df 6; E2h-KLR; Spisak tam / kısmi (`mlconfound`, linear); L5 karşıt hücreler; N2; EXP-020 G1 (hastalarda AM/PM probu); N5 (G1+ ise) |
| Aşama 2 (koşullu) | Yalnız T1-pozitif backbone'larda: N3 (arka plan; GPU dakikalar), L7 (codec zinciri), gün-gruplu CV |
| Karar kuralları | C1 / C2 (CONFOUND 6; `docs/EVAL016_PLAN.md` Bölüm 4) |
| Faz 3'e etkisi | Backbone seçimi (D-035) ve C3 ölçütü değişmez; D-027 kol B yalnız G1+ ya da L3 kısmi pozitifse; Faz 3 tasarımı "T1 sonrası" etiketli |
| Kod / notebook | `scripts/evaluate_context.py`, `notebooks/22_context_eval_frozen.ipynb` (yazılacak) |
| Sonuç konumu | `reports/context/EVAL-016_context.{md,json}`, `reports/context/EXP-020_g1.{md,json}` |

---

## Planlanan deneyler (ID'ler başlarken verilecek)

**Ana proje (D-012 merdiveni; D-028 ile değişmeden sürüyor):**

| Sıra | Deney | RQ | Ön koşul |
|---|---|---|---|
| 1 | ~~Ses denetimi~~ → AUD-001 tamamlandı | RQ7 | — |
| 1b | ~~EXP-003, EXP-004/004b, SIM-001~~ → tamamlandı (kayıt bağlamı belgelendi) | RQ7 | — |
| 1c | ~~Split dosyaları~~ → üretildi ve Colab'da aynı sha256 ile teyit edildi (`make_splits.py`, D-029; manifest `reports/splits/`) | altyapı | — |
| 1d | ~~Harmonize ses önbelleği~~ → Colab'da üretildi ve doğrulandı (D-030; rapor `reports/audio_cache/`) | altyapı | 1c |
| 2 | ~~MFCC sadık yeniden üretim (EXP-010)~~ → tamamlandı | RQ1 | 1d |
| 3 | ~~MFCC, ortak protokol (EXP-011)~~ → tamamlandı; EXP-012–015 keşifsel analizler | RQ1, RQ4, RQ5, RQ6 | 1c, 1d |
| 4 | ~~Dondurulmuş gömme + lineer prob~~ → **tamamlandı (2026-10-10)**: SMK-001 PASS, EXP-016 / 016S / 017 / 018, META-016. Üç SSL backbone MFCC'yi onaylayıcı testte geçti; hiçbiri yalnız-bağlam referansını (0.934) geçmedi (üst sınır, D-028). D-035: BEATs 24/25 fold | RQ1, RQ3, RQ4, RQ5 | — |
| 5 | Sıfırdan CNN10 — **zamanlaması D-036'da önerildi** (Faz 3'ün aynı pipeline'ında kontrol kolu olarak, ana kollardan sonra) | RQ2 | Faz 3 eğitim döngüsü + SMK-002 |
| 6 | Ham dalga formundan uçtan uca fine-tune: 4'te öne çıkan 1–2 aile + PANNs referans; Boll hiperparametreleri, iç doğrulamayla early stopping (D-009); en iyi ve son epoch tahminleri saklanır (D-028); seçim ölçütü **Faz 2 sonuçlarından önce yazıldı: D-035 (KABUL)** — CNN10 + CNN14 sabit; PANNs dışı backbone **her dış fold'da**, yalnız o fold'un eğitim verisindeki iç doğrulamayla | RQ2, RQ3 | 4 ve 5 tamam |
| 7 | Çok görevli füzyon, alt gruplar, ses + yaş/cinsiyet/sigara | RQ6, RQ8, RQ9 | OOF tahminleri |

Model geliştirme sırasında her sonuç **üst sınır** olarak raporlanır. Her tabloda iki referans satırı bulunur: yalnız bağlam (tarih + saat) AUC 0.931, yalnız yaş 0.681 (D-028).

**Kayıt bağlamı değerlendirmesi (D-023, D-028; model geliştirme bitince):**

| Sıra | Analiz | RQ | Ön koşul |
|---|---|---|---|
| E-0 | **EVAL-016 (dondurulmuş aile, onaylayıcı) — D-037 önerisiyle Faz 3'ten önce** (`docs/EVAL016_PLAN.md`) | RQ7 | 4'ün saklanmış tahminleri |
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
