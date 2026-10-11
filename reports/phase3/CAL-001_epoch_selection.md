# CAL-001 — Faz 3 kalibrasyon eğrileri ve epoch seçimi (otomatik üretildi; katılımcı ID'si içermez)

Kural: kalibrasyon: iç doğrulama katılımcı AUC'si, 3'lü hareketli ort., 1-SE kuralı (Hanley–McNeil SE); ısınma sabit epoch sayısı (D-043) (Faz 3 protokolü v2 (D-042 + D-043), KABUL). v1 (kayıp) kaydı için yanında.

| kol | v2 epoch (AUC 1-SE) | v1 epoch (kayıp) | AUC MA en yüksek (epoch) | SE | seçilen epoch'ta AUC MA | eğri | val kaybı min / son | eğitim kaybı son | üst sınırda |
|---|---|---|---|---|---|---|---|---|---|
| beats | 8 | 1 | 0.916 (e11) | 0.038 | 0.891 | 20 | 0.235 / 0.552 | 0.013 | False |
| cnn10 | 7 | 5 | 0.806 (e31) | 0.065 | 0.748 | 40 | 0.241 / 1.429 | 0.058 | False |
| cnn10_scratch | 9 | 1 | 0.775 (e13) | 0.071 | 0.726 | 40 | 0.236 / 0.834 | 0.118 | False |
| cnn14 | 9 | 1 | 0.767 (e31) | 0.072 | 0.696 | 40 | 0.263 / 1.545 | 0.026 | False |
| wavlm_base_plus | 1 | 1 | 0.730 (e19) | 0.078 | 0.671 | 20 | 0.232 / 0.904 | 0.011 | False |
| wavlm_large | 9 | 1 | 0.857 (e14) | 0.054 | 0.807 | 20 | 0.227 / 0.789 | 0.025 | False |
