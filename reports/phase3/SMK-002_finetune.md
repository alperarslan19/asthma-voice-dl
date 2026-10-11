# SMK-002 — Faz 3 fine-tuning smoke testi (otomatik üretildi; katılımcı ID'si içermez)

- Ortam: {'torch': '2.11.0+cu130', 'device': 'cuda', 'random_init_test': False, 'gpu': 'Tesla T4'}
- Protokol: Faz 3 protokolü v1 (D-042) — ÖNERİLDİ

| kol | durum | T2 ileri | T3 eğitim (kayıp e1→e2; değişen gövde) | T4 değ. | T5 kurtarma (fark; ckpt MB) | GPU tepe GB | s/adım | tahmini epoch / fold (dk, en çok epoch) |
|---|---|---|---|---|---|---|---|---|
| cnn10 | **PASS** | PASS | PASS (0.226→0.207; 28/30) | PASS | PASS (0.0e+00; 62.1) | 0.65 | 0.1 | 0.2 / 8.2 (40) |
| cnn10_scratch | **PASS** | PASS | PASS (0.215→0.215; 28/30) | PASS | PASS (0.0e+00; 62.1) | 0.66 | 0.1 | 0.2 / 8.1 (40) |
| beats | **PASS** | PASS | PASS (0.249→0.206; 240/240) | PASS | PASS (0.0e+00; 1086.4) | 4.25 | 0.35 | 0.69 / 14.6 (20) |
| wavlm_base_plus | **PASS** | PASS | PASS (0.250→0.220; 239/240) | PASS | PASS (0.0e+00; 1084.8) | 5.55 | 0.425 | 0.84 / 17.8 (20) |
| wavlm_large | **PASS** | PASS | PASS (0.247→0.212; 465/468) | PASS | PASS (0.0e+00; 3738.7) | 11.71 | 1.175 | 2.33 / 49.0 (20) |
| cnn14 | **PASS** | PASS | PASS (0.253→0.203; 40/42) | PASS | PASS (0.0e+00; 966.9) | 6.91 | 0.15 | 0.3 / 12.3 (40) |

Süre tahmini: 2. küçük epoch'un adım süresinden (8 kayıtlık izleme değerlendirmesi dahil) × tam epoch adım sayısı; fold = en çok epoch × epoch + test değerlendirmesi. Kalibrasyondan sonra gerçek epoch sayısıyla güncellenir. [INFERENCE]

**Kural:** FAIL olan kolun kalibrasyonu ve eğitimi başlatılmaz; neden incelenir, düzeltilir, SMK-002 yeniden çalıştırılır.
