# asthma-voice-dl

Ses kayıtlarından astım / sağlıklı ayrımı: MFCC + klasik ML'den, pretrained ses modellerine (PANNs, BEATs, WavLM) uzanan, sızıntı ve confounder kontrollü bir araştırma.

**Veri:** 348 katılımcı (284 astım, 64 sağlıklı), kişi başı 7 kayıt (~10 s): sürdürülmüş /a/ + 6 Türkçe kelimenin tekrarı. Birincil kohort: yayınlanmış çalışmayla aynı 344 kişi (D-002). Veri bu repoda **yoktur** (D-014).

## Nereden başlamalı
1. `docs/PHASE0_REPORT.md` — ne bildiğimiz, ne bilmediğimiz, neden böyle tasarladığımız
2. `docs/COLAB_WORKFLOW.md` — GitHub / Drive / Colab kurulumu ve her oturumun başı
3. `notebooks/01_data_audit.ipynb` — şu anki adım

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
| 1 | **Split dosyaları + harmonize ses önbelleği** | ⏳ **sıradaki adım** |
| 1 | MFCC baseline yeniden üretimi (EXP-010/011) | ⏸ split dosyalarından sonra |
| 2 | Smoke test'ler, dondurulmuş gömme + lineer prob (6 backbone) | ⏸ |
| 3 | Sıfırdan CNN, fine-tune, füzyon, confounder ve alt grup analizleri | ⏸ |

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
tests/                script testleri (gerçek veri gerektirmez): python tests/test_audit_audio.py
```

## Değişmez kurallar
- Split katılımcı düzeyinde; segment sayısı hasta sayısı değildir.
- İstatistik öğrenen her işlem (ölçekleme, SMOTE, augmentation havuzu) yalnız train fold'unda.
- Birincil test: ses skoru, kayıt bağlamı (tarih + saat) ve yaşın ötesinde bilgi ekliyor mu (D-023); her sonuç E-tasarımları ve denge tablosuyla raporlanır.
- Olumsuz sonuçlar silinmez.
- Smoke test geçmeden tam eğitim yok; her epoch checkpoint.
