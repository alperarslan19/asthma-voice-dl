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

---

## Planlanan deneyler (ID'ler başlarken verilecek)
| Sıra | Deney | RQ | Ön koşul |
|---|---|---|---|
| 1 | ~~Ses denetimi~~ → AUD-001 tamamlandı | RQ7 | — |
| 1b | ~~EXP-003~~ → tamamlandı; kesim 11.0 kHz | RQ7 | — |
| 1c | Split dosyaları (`make_splits.py`) + harmonize ses önbelleği (`build_audio_cache.py`, D-021) | altyapı | D-024 ile yeniden hesaplanan dönem |
| 2 | MFCC sadık yeniden üretim | RQ1 | Eşleme doğrulandı |
| 3 | MFCC, ortak protokol + kontroller | RQ1, RQ4, RQ5, RQ7 | Split dosyaları |
| 4 | Dondurulmuş gömme + lineer prob (6 backbone × 7 görev), E1/E2/E3 ile | RQ3, RQ4, RQ5 | Ses önbelleği, smoke test |
| 4b | Zamansal negatif kontroller N1–N3 + **zincir probu** (harmonizasyon öncesi/sonrası, D-021) | RQ7 | 4 ile aynı gömmeler |
| 5 | Sıfırdan CNN10 | RQ2 | Eğitim döngüsü + checkpoint testi |
| 6 | Fine-tune (seçilmiş backbone/görev) | RQ2, RQ3 | 4 ve 5 tamam |
| 7 | Çok görevli füzyon, alt gruplar, ses + yaş/cinsiyet/sigara | RQ6, RQ7, RQ8, RQ9 | OOF tahminleri |

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
