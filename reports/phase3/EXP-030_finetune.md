# EXP-030 — Faz 3 uçtan uca fine-tune: değerlendirme (otomatik üretildi; katılımcı ID'si içermez)

Kohort 339 (282 / 57); 3 tekrar × 5 fold, gün-gruplu (splits_day). Tanımlar PHASE3_DESIGN 5.2 (sonuçlardan önce). Bootstrap B 2000; permütasyon 1000. C3 keşifsel; Holm, ΔE1 ve O1 betimsel.

## E1 ve T1

| kol | skor | E1 fold AUC ort ± SD [min–maks] | ŝ havuzlanmış AUC [%95] | T1 PLR p | OR / SD [bootstrap %95] | CV3 p | dengeli doğr. (eşik 0) |
|---|---|---|---|---|---|---|---|
| cnn10 | fine-tune | 0.867 ± 0.063 [0.75–0.97] | 0.866 [0.811, 0.913] | <0.001 | 3.068 [1.802, 11.663] | 0.083 | 0.717 |
| | dondurulmuş eş | 0.823 ± 0.093 | 0.821 | 0.004 | 2.293 | 0.082 | |
| cnn10_scratch | fine-tune | 0.686 ± 0.096 [0.43–0.79] | 0.691 [0.614, 0.767] | 0.244 | 1.415 [0.891, 2.829] | 0.306 | 0.583 |
| ssl | fine-tune | 0.893 ± 0.077 [0.71–0.97] | 0.917 [0.868, 0.956] | <0.001 | 3.516 [2.132, 14.049] | 0.050 | 0.741 |
| | dondurulmuş eş | 0.882 ± 0.062 | 0.895 | <0.001 | 3.166 | 0.051 | |
| cnn14 | fine-tune | 0.865 ± 0.078 [0.67–0.98] | 0.871 [0.806, 0.926] | <0.001 | 2.529 [1.476, 10.598] | 0.072 | 0.677 |
| | dondurulmuş eş | 0.767 ± 0.084 | 0.772 | 0.008 | 2.008 | 0.026 | |

## C3 — fine-tune, dondurulmuş eşe göre bağlamın ötesinde daha fazla bilgi taşıyor mu? (keşifsel)

| kol | LR fine-tune | LR dondurulmuş | fark [%95 bootstrap] | okuma | ΔE1 FT − dondurulmuş (fold-eşli; NB p) | FT daha iyi fold | ŝ korelasyonu |
|---|---|---|---|---|---|---|---|
| cnn10 | 12.89 | 8.35 | +4.54 [-5.76, +17.64] | **belirsiz** | +0.044 ± 0.072 (p 0.297) | 11 / 15 | 0.80 |
| cnn10_scratch | — | — | — | dondurulmuş eş yok | — | — | — |
| ssl | 15.71 | 13.01 | +2.70 [-6.27, +12.92] | **belirsiz** | +0.010 ± 0.053 (p 0.734) | 7 / 15 | 0.78 |
| cnn14 | 11.87 | 6.99 | +4.88 [-6.24, +24.15] | **belirsiz** | +0.098 ± 0.134 (p 0.222) | 12 / 15 | 0.57 |

## C2 — bağlam kullanımı (etiket-içi permütasyon kısmi testi; L5)

| kol | skor | kısmi R² (p) | hastalarda PM katsayısı [%95] | sağlıklılarda AM − PM [%95] |
|---|---|---|---|---|
| cnn10 | fine-tune | 0.020 (0.524) | -0.00 [-0.23, +0.23] | +0.18 [-0.24, +0.60] (n 12 / 45) |
| cnn10 | dondurulmuş eş | 0.024 (0.439) | +0.05 [-0.21, +0.31] | +0.12 [-0.30, +0.60] (n 12 / 45) |
| cnn10_scratch | fine-tune | 0.075 (0.003) | -0.01 [-0.24, +0.23] | -0.09 [-0.63, +0.47] (n 12 / 45) |
| ssl | fine-tune | 0.055 (0.024) | +0.00 [-0.17, +0.18] | +0.78 [+0.28, +1.29] (n 12 / 45) |
| ssl | dondurulmuş eş | 0.037 (0.132) | -0.02 [-0.24, +0.21] | +0.24 [-0.16, +0.67] (n 12 / 45) |
| cnn14 | fine-tune | 0.026 (0.398) | +0.02 [-0.18, +0.21] | +0.65 [+0.10, +1.22] (n 12 / 45) |
| cnn14 | dondurulmuş eş | 0.038 (0.121) | +0.06 [-0.19, +0.30] | +0.23 [-0.37, +0.84] (n 12 / 45) |

## RQ2 — ön-eğitim (cnn10 − cnn10_scratch)

Fold-eşli ΔAUC +0.181 ± 0.128; Nadeau–Bengio t 2.50, p 0.025; önceden eğitilmiş daha iyi: 14 / 15 fold. T1 iki kol için yukarıdaki tabloda.

## O1 — aşırı uyum ve kalibrasyon (betimsel; post hoc kullanıcı sorusu)

Test kalibrasyon eğimi: y ~ ŝ lojistik regresyonunda ŝ'nin katsayısı. 1'den küçükse skorlar fazla uç (aşırı güven).
Sabit tahminci eğitim kaybı ≈ 0.23.

| kol | epoch | son eğitim kaybı (ort ± SD) | atlanan adım | FT eğim [%95] | FT ort. abs(ŝ) | dondurulmuş eğim | dondurulmuş ort. abs(ŝ) |
|---|---|---|---|---|---|---|---|
| cnn10 | 7.0 | 0.172 ± 0.010 | 11 | 1.03 [0.76, 1.29] | 2.48 | 2.17 | 0.77 |
| cnn10_scratch | 9.0 | 0.221 ± 0.009 | 11 | 0.77 [0.43, 1.10] | 1.06 | — | — |
| ssl | 7.7 | 0.055 ± 0.050 | 14 | 0.61 [0.46, 0.76] | 5.50 | 1.95 | 1.44 |
| cnn14 | 9.0 | 0.102 ± 0.010 | 38 | 0.56 [0.42, 0.70] | 6.21 | 2.83 | 0.45 |

Holm (betimsel, 4 kol): ssl <0.001, cnn10 <0.001, cnn14 0.001, cnn10_scratch 0.244

- E1 artışı tek başına başarı sayılmaz (C3). Bütün kollar ve olumsuz sonuçlar raporlanır.
- ssl kolu fold başına D-035 seçimidir: {'beats': 12, 'wavlm_large': 2, 'wavlm_base_plus': 1}.
