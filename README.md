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
| 0 | **Ses denetimi + eşleme + kısa dinleme teyidi** | ⏳ **sıradaki adım** — `01_data_audit.ipynb` |
| 0 | MFCC baseline yeniden üretimi | ⏸ eşleme doğrulanınca |
| 1 | Split dosyaları, ses önbelleği (16/32 kHz), MFCC ortak protokol | ⏸ |
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
- Her başlık sonucu üç düzeyde raporlanır (tam kohort / zaman penceresi / aynı gün) ve yaş-only baseline'la karşılaştırılır; başlık sayısı zaman penceresidir (D-017).
- Olumsuz sonuçlar silinmez.
- Smoke test geçmeden tam eğitim yok; her epoch checkpoint.
