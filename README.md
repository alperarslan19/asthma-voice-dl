# asthma-voice-dl

Ses kayıtlarından astım / sağlıklı ayrımı: MFCC + klasik ML'den, pretrained ses modellerine (PANNs, BEATs, WavLM) uzanan, sızıntı ve confounder kontrollü bir araştırma.

**Veri:** 348 katılımcı (284 astım, 64 sağlıklı), kişi başı 7 kayıt (~10 s): sürdürülmüş /a/ + 6 Türkçe kelimenin tekrarı. Birincil kohort: yayınlanmış çalışmayla aynı 344 kişi (D-002). Veri bu repoda **yoktur** (D-014).

## Nereden başlamalı
1. `docs/PHASE0_REPORT.md` — ne bildiğimiz, ne bilmediğimiz, neden böyle tasarladığımız
2. `docs/CONFOUND_CONTROL_DESIGN.md` — kayıt bağlamı confounder'ı: yöntemlerin değerlendirmesi; model geliştirme bittikten sonraki değerlendirme aşamasının yol haritası (D-028)
3. `docs/COLAB_WORKFLOW.md` — GitHub / Drive / Colab kurulumu ve her oturumun başı
4. `notebooks/01_data_audit.ipynb` — ses denetimi (tamamlandı)

## Proje haritası

| Faz | İş | Durum |
|---|---|---|
| 0 | Literatür denetimi | ✅ |
| 0 | Klinik veri denetimi + EXP-001 (sesi kullanmayan baseline'lar) | ✅ |
| 0 | Model girdi sözleşmeleri (resmi kaynaklardan) | ✅ |
| 0 | Değerlendirme protokolü, altyapı, karar/deney günlükleri | ✅ (D-002, D-009 kesinleşti) |
| 0 | Zamansal confounder stratejisi (E1 tam / E2 zaman penceresi / E3 aynı gün + negatif kontroller) | ✅ D-017 |
| 0 | Ses denetimi + eşleme + dinleme teyidi | ✅ 342 katılımcı (283/59), 2 393 kayıt |
| 0 | Harmonizasyon (D-021, kesim 11.0 kHz), kayıt bağlamı analizi EXP-003, revize birincil test (D-023) | ✅ |
| 0 | Confounder kontrol yöntemlerinin değerlendirmesi, EXP-004/004b, SIM-001, test spesifikasyonu (D-025/026/027 önerildi) | ✅ |
| 0 | Confounder ayrı araştırma başlığı; değerlendirme geliştirme sonuna (D-028); ön işleme (D-015) ve augmentation (D-016) kabul | ✅ |
| 1 | Split dosyaları (`make_splits.py`, D-029): 5 tekrar × 5 fold, manifest `reports/splits/` | ✅ Colab'da aynı sha256 ile teyit edildi |
| 1 | Harmonize ses önbelleği (`build_audio_cache.py`, D-030): 2 393 kayıt, rapor `reports/audio_cache/` | ✅ Colab'da üretildi, doğrulandı, dinlendi |
| 1 | MFCC baseline'ları (EXP-010 sadık yeniden üretim, EXP-011 bizim protokol, EXP-012 keşifsel ayrıştırma) | ✅ makale aralığı yeniden üretildi (en iyi ort. 0.702 vs 0.709); dürüst taban: LR füzyon 0.771 [0.713–0.836], tek görev ≈ yaş (0.68), bağlam 0.93 |
| 1 | Metodoloji soruları (`docs/IMBALANCE_AND_SELECTION.md`): EXP-013 dengesizlik (SMOTE ≈ ağırlık ≈ hiçbiri; düzeltmeler kalibrasyonu bozuyor), EXP-014 bağlam dengeleme (işe yaramıyor; pozitiflik), EXP-015 iç içe CV (makale prosedürü dürüstçe 0.654) | ✅ keşifsel; D-032, D-033 önerildi |
| 2 | **Smoke test'ler, dondurulmuş gömme + lineer prob (6 backbone)** | ⏳ sıradaki faz (tasarım) |
| 3 | Sıfırdan CNN10, ham dalga formundan uçtan uca fine-tune, füzyon, alt gruplar | ⏸ |
| 4 | **Kayıt bağlamı değerlendirmesi** (D-023, D-028): kaydedilmiş tahminler üzerinde T1, E-tasarımları, negatif kontroller | ⏸ model geliştirme bitince |

## Klasörler
```
DECISIONS.md          karar günlüğü (D-001 …)
EXPERIMENTS.md        bilimsel deney günlüğü (EXP-001 …)
results/registry.csv  makinece okunan deney kaydı (Google Sheets'e içe aktarılır)
docs/                 Faz raporları ve iş akışı
scripts/              tüm mantık (notebook'lar bunları çağırır)
notebooks/            ince orkestrasyon notebook'ları (Colab)
configs/              model girdi sözleşmeleri, slot→görev eşlemesi
reports/              yalnız agrega denetim çıktıları
tests/                script testleri (gerçek veri gerektirmez): python tests/test_audit_audio.py, tests/test_make_splits.py, tests/test_build_audio_cache.py, tests/test_mfcc_pipeline.py
```

## Değişmez kurallar
- Split katılımcı düzeyinde; segment sayısı hasta sayısı değildir.
- İstatistik öğrenen her işlem (ölçekleme, SMOTE, augmentation havuzu) yalnız train fold'unda.
- Kayıt bağlamı (tarih + saat) sesi kullanmadan etiketi AUC 0.93 ile ele veriyor. Değerlendirme aşamasına kadar her ses modeli sonucu **üst sınır** olarak raporlanır; tabloda iki referans satırı bulunur: yalnız bağlam 0.93, yalnız yaş 0.68 (D-028). "Ses astım bilgisi taşıyor" iddiası yalnız değerlendirme aşamasındaki artımlı testten çıkar (D-023).
- Her deney dış-test tahminlerini katılımcı, kayıt ve segment düzeyinde saklar. Fine-tune'da hem en iyi doğrulama checkpoint'inin hem son epoch'un tahminleri saklanır (D-028).
- Olumsuz sonuçlar silinmez.
- Smoke test geçmeden tam eğitim yok; her epoch checkpoint.
