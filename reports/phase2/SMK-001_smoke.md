# SMK-001 — Faz 2 smoke test, GERÇEK checkpoint'ler (otomatik üretildi; katılımcı ID'si içermez)

- Ortam: {'python': '3.13.15', 'torch': '2.11.0+cu130', 'torchaudio': '2.11.0+cu130', 'transformers': '5.18.0', 'numpy': '2.1.3'}
- Cihaz: cuda · random_init: False

| backbone | durum | S3 strict | S5 şekil | S6 det. | S7 batch | S8 kanca | S9 Speech sırası | S10 aynı kişi AUC | S11 | S12 eğitim (GB, s/adım) | S13 pencere/s | S14 dolgu (tam=; kısa cos min) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cnn10 | **PASS** | PASS | 5×1×512 | 0.0e+00 | 2.0e-06 | — | 1 | 0.67 | PASS | 0.63, 0.063 | 424.6 | PASS; 0.8593 |
| cnn14 | **PASS** | PASS | 5×1×2048 | 0.0e+00 | 5.4e-06 | — | 1 | 0.69 | PASS | 1.82, 0.125 | 233.1 | PASS; 0.8043 |
| cnn14_16k | **PASS** | PASS | 5×1×2048 | 0.0e+00 | 7.6e-06 | — | 1 | 0.71 | PASS | 1.82, 0.125 | 272.5 | PASS; 0.639 |
| beats | **PASS** | PASS | 5×13×768 | 0.0e+00 | 4.8e-07 | 0.0e+00 | — | 0.83 | PASS | 3.86, 0.316 | 73.8 | PASS; 0.9829 |
| wavlm_base_plus | **PASS** | PASS | 5×13×768 | 0.0e+00 | 1.4e-05 | 0.0e+00 | — | 0.67 | PASS | 5.4, 0.378 | 54.6 | PASS; 0.8889 |
| wavlm_large | **PASS** | PASS | 5×25×1024 | 0.0e+00 | 1.5e-03 | 0.0e+00 | — | 0.70 | PASS | 13.05, 1.004 | 20.9 | PASS; 1.0 |

Ayrıntılar (parametre sayısı, sha256, yükleme yöntemi, config) JSON'da.

**Kural (docs/PHASE2_DESIGN.md 1.1):** FAIL olan bir backbone varsa gömme çıkarımı ve EXP-016 başlatılmaz. Önce neden incelenir (JSON'daki değerler), düzeltilir, EXPERIMENTS.md'ye yazılır ve SMK-001 baştan çalıştırılır. Eşik gevşetilerek geçirme yapılmaz. Bu rapor birim testlerinden (UNITTEST_random_init_smoke) ayrıdır.
