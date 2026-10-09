# EXP-012 — MFCC sonuçlarının keşifsel ayrıştırması (post hoc; katılımcı ID'si içermez)

**Bu analiz sonuçlar görüldükten sonra tasarlandı:** bulgular hipotez üretir. Bütün AUC'ler üst sınırdır (D-028).

## A. Yeniden üretilebilirlik

EXP-011 LR bu makinede (farklı scikit-learn sürümü) yeniden çalıştırıldı; görev başına en büyük fark: 3.9e-05.

## B. Girdi mi, protokol mü? (yalnız LR, fold AUC ortalaması)

| görev | P1 makale özellik + makale protokol (EXP-010 LR) | P4 önbellek özellik + makale protokol | P2 makale özellik + bizim protokol | P3 önbellek özellik + bizim protokol (EXP-011 LR) |
|---|---|---|---|---|
| aaa | 0.713 | 0.694 | 0.692 | 0.647 |
| araba | 0.686 | 0.694 | 0.623 | 0.691 |
| ana | 0.686 | 0.683 | 0.649 | 0.681 |
| ordu | 0.602 | 0.574 | 0.662 | 0.649 |
| gelecek | 0.643 | 0.590 | 0.690 | 0.647 |
| titiz | 0.627 | 0.659 | 0.634 | 0.665 |
| ünlem | 0.698 | 0.654 | 0.699 | 0.639 |
| fusion (7 görev) | — | — | 0.781 | 0.771 |

## C. Split şansı (makale protokolü + LR; aynı veri, 50 farklı 5-fold bölünmesi)

| görev | tohum 42 | 50 tohum ort ± SD | min–max | tohum 42'nin yüzdeliği |
|---|---|---|---|---|
| aaa | 0.713 | 0.696 ± 0.024 | 0.631–0.736 | 78 |
| araba | 0.686 | 0.657 ± 0.019 | 0.611–0.692 | 92 |
| ana | 0.686 | 0.643 ± 0.023 | 0.577–0.691 | 96 |
| ordu | 0.602 | 0.624 ± 0.027 | 0.545–0.685 | 18 |
| gelecek | 0.643 | 0.671 ± 0.019 | 0.630–0.714 | 10 |
| titiz | 0.627 | 0.647 ± 0.021 | 0.608–0.706 | 22 |
| ünlem | 0.698 | 0.678 ± 0.026 | 0.621–0.729 | 76 |

## D. Kazananın laneti (makale pipeline'ı; makale özellikleri)

| görev | tohum 42'de kazanan | AUC (tohum 42, 5 fold) | aynı model, bizim 25 fold | düşüş | 25 fold'da sırası | 25 fold'da en iyi |
|---|---|---|---|---|---|---|
| aaa | MLP | 0.725 | 0.699 | +0.026 | 1/10 | MLP (0.699) |
| ünlem | MLP | 0.723 | 0.704 | +0.019 | 1/10 | MLP (0.704) |
| gelecek | StackingEnsemble | 0.719 | 0.720 | -0.001 | 1/11 | StackingEnsemble (0.720) |
| titiz | MLP | 0.695 | 0.691 | +0.004 | 2/10 | SVM (0.694) |
| ana | XGBoost | 0.687 | 0.662 | +0.025 | 1/10 | XGBoost (0.662) |
| araba | LogisticRegression | 0.686 | 0.625 | +0.061 | 4/10 | XGBoost (0.639) |
| ordu | SVM | 0.677 | 0.683 | -0.007 | 1/10 | SVM (0.683) |

## E. Eşleştirilmiş karşılaştırmalar (EXP-011 LR; D-011)

| karşılaştırma | ΔAUC fold ort. | Nadeau–Bengio t, p | ΔAUC havuzlanmış [%95 bootstrap CI] | n |
|---|---|---|---|---|
| LR füzyon − aaa | +0.124 | 3.09, 0.005 | +0.130 [+0.065, +0.200] | 342 |
| LR füzyon − araba | +0.080 | 2.10, 0.046 | +0.081 [+0.025, +0.137] | 342 |
| LR füzyon − ana | +0.090 | 2.07, 0.050 | +0.088 [+0.023, +0.155] | 342 |
| LR füzyon − ordu | +0.122 | 2.87, 0.008 | +0.129 [+0.070, +0.186] | 341 |
| LR füzyon − gelecek | +0.124 | 2.86, 0.009 | +0.134 [+0.066, +0.202] | 342 |
| LR füzyon − titiz | +0.106 | 2.62, 0.015 | +0.107 [+0.019, +0.197] | 342 |
| LR füzyon − ünlem | +0.132 | 3.78, 0.001 | +0.134 [+0.068, +0.198] | 342 |
| LR füzyon − age | +0.089 | 1.98, 0.060 | +0.095 [+0.008, +0.185] | 340 |
| LR füzyon − context (hour+hour²+date) | -0.163 | -3.47, 0.002 | -0.154 [-0.224, -0.089] | 340 |

## F. Aynı kaydın makale ve önbellek özellikleri ne kadar benzer?

- 2393 kayıt; özellik başına Pearson r medyanı 0.75, en düşük 0.226
- r < 0.8 olan özellikler: {'mfcc00_mean': 0.226, 'd1_mfcc00_mean': 0.699, 'd1_mfcc01_mean': 0.622, 'd1_mfcc02_mean': 0.707, 'd1_mfcc03_mean': 0.758, 'd1_mfcc04_mean': 0.742, 'd1_mfcc05_mean': 0.782, 'd1_mfcc06_mean': 0.743, 'd1_mfcc07_mean': 0.771, 'd1_mfcc08_mean': 0.742, 'd1_mfcc09_mean': 0.74, 'd1_mfcc10_mean': 0.721, 'd1_mfcc11_mean': 0.724, 'd2_mfcc00_mean': 0.632, 'd2_mfcc01_mean': 0.535, 'd2_mfcc02_mean': 0.517, 'd2_mfcc03_mean': 0.593, 'd2_mfcc04_mean': 0.479, 'd2_mfcc05_mean': 0.585, 'd2_mfcc06_mean': 0.594, 'd2_mfcc07_mean': 0.52, 'd2_mfcc08_mean': 0.547, 'd2_mfcc09_mean': 0.584, 'd2_mfcc10_mean': 0.56, 'd2_mfcc11_mean': 0.501}
- Kırpma sonrası süre medyanı: makale 10.43 s, önbellek 10.26 s

Süre: 1217.4 s
