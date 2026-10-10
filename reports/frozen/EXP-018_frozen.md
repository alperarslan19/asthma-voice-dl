# EXP-018 — dondurulmuş gömme + lineer prob (otomatik üretildi; katılımcı ID'si içermez)

**Rol:** keşifsel (tüm kayıt vs 4 s pencere; seçimde kullanılmaz)  
**Üst sınır uyarısı (D-028):** bütün AUC'ler kayıt bağlamı değerlendirilmeden hesaplandı. Aynı fold'larda yalnız bağlam (saat + saat² + tarih) ve yalnız yaş referansları aşağıda.

Split: ['outer_r0.csv', 'outer_r1.csv', 'outer_r2.csv', 'outer_r3.csv', 'outer_r4.csv'] · tekrar: 5 · test/train oranı (Nadeau–Bengio): 0.250

**İstatistik notu:** Bütün Nadeau–Bengio (NB) testleri 25 fold skoruna (5 tekrar × 5 dış fold) uygulanır. Bu skorlar **bağımsız değildir**: aynı tekrardaki fold'ların eğitim kümeleri büyük ölçüde örtüşür ve tekrarlar aynı katılımcıları yeniden böler. NB düzeltmesi (varyansa n_test/n_train terimi eklenir) bu bağımlılığı kabaca telafi eden sezgisel bir düzeltmedir; df = 24 yaklaşıktır. Düz t-testi burada fazla iyimser olurdu.

Değerlendirme birimi: **katılımcı**. Bütün kollar aynı split dosyalarında, aynı dış test katılımcılarında değerlendirildi (assert). Kayıt bağlamı / süre gibi meta veri karşılaştırmaları bu raporda yok → META-016.

## Füzyon (katılımcı düzeyi; 7 görev olasılığının ortalaması) — betimsel

| kol | AUC fold ort ± SD [p2.5–p97.5] | havuzlanmış AUC [%95 CI] | dengeli doğr. (iç-CV eşiği) | duyarlılık / özgüllük | Brier | ort. tahmin − oran | kal. eğimi |
|---|---|---|---|---|---|---|---|
| beats | 0.921 ± 0.046 [0.825–0.984] | 0.923 [0.886–0.953] | 0.811 ± 0.077 | 0.82 / 0.81 | 0.095 | -0.091 | 2.08 |
| beats@full | 0.910 ± 0.051 [0.796–0.978] | 0.914 [0.872–0.946] | 0.818 ± 0.080 | 0.81 / 0.83 | 0.097 | -0.090 | 1.93 |
| cnn10 | 0.855 ± 0.047 [0.768–0.934] | 0.858 [0.807–0.903] | 0.757 ± 0.052 | 0.77 / 0.75 | 0.146 | -0.188 | 2.42 |
| cnn10@full | 0.828 ± 0.047 [0.726–0.906] | 0.835 [0.779–0.885] | 0.752 ± 0.052 | 0.69 / 0.82 | 0.154 | -0.194 | 2.26 |
| cnn14 | 0.791 ± 0.063 [0.665–0.886] | 0.811 [0.747–0.869] | 0.721 ± 0.058 | 0.70 / 0.74 | 0.183 | -0.239 | 3.22 |
| cnn14@full | 0.805 ± 0.055 [0.689–0.893] | 0.816 [0.756–0.867] | 0.726 ± 0.055 | 0.72 / 0.73 | 0.156 | -0.188 | 2.38 |
| cnn14_16k | 0.729 ± 0.067 [0.605–0.859] | 0.733 [0.666–0.794] | 0.657 ± 0.077 | 0.59 / 0.72 | 0.196 | -0.253 | 2.24 |
| cnn14_16k@full | 0.764 ± 0.061 [0.656–0.857] | 0.775 [0.711–0.831] | 0.710 ± 0.053 | 0.61 / 0.81 | 0.180 | -0.231 | 2.31 |
| ref_age — referans çizgisi (D-028; test edilmez) | 0.682 ± 0.046 [0.615–0.757] | 0.679 [0.601–0.750] | 0.620 ± 0.041 (eşik 0.5) | 0.66 / 0.58 | 0.225 | -0.299 | 0.92 |
| ref_context — referans çizgisi (D-028; test edilmez) | 0.934 ± 0.022 [0.894–0.972] | 0.931 [0.899–0.958] | 0.839 ± 0.042 (eşik 0.5) | 0.86 / 0.82 | 0.106 | -0.141 | 0.84 |
| wavlm_base_plus | 0.884 ± 0.039 [0.815–0.944] | 0.886 [0.836–0.926] | 0.774 ± 0.047 | 0.77 / 0.78 | 0.109 | -0.118 | 1.87 |
| wavlm_base_plus@full | 0.880 ± 0.044 [0.785–0.940] | 0.887 [0.837–0.927] | 0.778 ± 0.060 | 0.80 / 0.76 | 0.108 | -0.115 | 1.93 |
| wavlm_large | 0.909 ± 0.041 [0.813–0.970] | 0.909 [0.868–0.942] | 0.791 ± 0.049 | 0.76 / 0.82 | 0.100 | -0.102 | 1.94 |
| wavlm_large@full | 0.905 ± 0.040 [0.808–0.960] | 0.907 [0.868–0.941] | 0.793 ± 0.064 | 0.78 / 0.81 | 0.099 | -0.098 | 1.94 |

## Önceden listelenmiş keşifsel karşılaştırma: tüm kayıt − 4 s pencere (füzyon)

| a − b | fold ort. Δ ± SD | havuzlanmış Δ [%95 CI] | NB p | Holm p |
|---|---|---|---|---|
| cnn10@full − cnn10 | -0.027 ± 0.027 | -0.023 [-0.049, +0.001] | 0.077 | 0.462 |
| cnn14@full − cnn14 | +0.014 ± 0.055 | +0.005 [-0.036, +0.047] | 0.640 | 1.000 |
| cnn14_16k@full − cnn14_16k | +0.035 ± 0.045 | +0.042 [-0.003, +0.086] | 0.154 | 0.770 |
| beats@full − beats | -0.011 ± 0.015 | -0.009 [-0.020, +0.001] | 0.199 | 0.795 |
| wavlm_base_plus@full − wavlm_base_plus | -0.004 ± 0.016 | +0.001 [-0.011, +0.014] | 0.623 | 1.000 |
| wavlm_large@full − wavlm_large | -0.003 ± 0.013 | -0.002 [-0.012, +0.008] | 0.640 | 1.000 |

## Görev başına AUC (fold ortalaması ± SD; keşifsel, RQ4/RQ5)

| kol | aaa | araba | ana | ordu | gelecek | titiz | ünlem |
|---|---|---|---|---|---|---|---|
| beats | 0.743 ± 0.072 | 0.816 ± 0.058 | 0.815 ± 0.073 | 0.856 ± 0.065 | 0.797 ± 0.070 | 0.896 ± 0.045 | 0.844 ± 0.063 |
| beats@full | 0.737 ± 0.076 | 0.827 ± 0.073 | 0.805 ± 0.072 | 0.858 ± 0.066 | 0.789 ± 0.064 | 0.876 ± 0.045 | 0.825 ± 0.071 |
| cnn10 | 0.731 ± 0.074 | 0.757 ± 0.052 | 0.762 ± 0.059 | 0.723 ± 0.073 | 0.725 ± 0.076 | 0.715 ± 0.070 | 0.755 ± 0.055 |
| cnn10@full | 0.729 ± 0.073 | 0.740 ± 0.059 | 0.696 ± 0.064 | 0.705 ± 0.078 | 0.707 ± 0.061 | 0.721 ± 0.077 | 0.730 ± 0.051 |
| cnn14 | 0.720 ± 0.063 | 0.693 ± 0.067 | 0.700 ± 0.067 | 0.706 ± 0.082 | 0.639 ± 0.076 | 0.676 ± 0.087 | 0.657 ± 0.062 |
| cnn14@full | 0.706 ± 0.071 | 0.669 ± 0.069 | 0.698 ± 0.069 | 0.687 ± 0.068 | 0.643 ± 0.088 | 0.675 ± 0.072 | 0.715 ± 0.071 |
| cnn14_16k | 0.671 ± 0.068 | 0.675 ± 0.069 | 0.690 ± 0.086 | 0.666 ± 0.081 | 0.620 ± 0.082 | 0.581 ± 0.090 | 0.685 ± 0.070 |
| cnn14_16k@full | 0.704 ± 0.063 | 0.656 ± 0.060 | 0.679 ± 0.079 | 0.697 ± 0.082 | 0.645 ± 0.070 | 0.646 ± 0.077 | 0.698 ± 0.074 |
| wavlm_base_plus | 0.639 ± 0.058 | 0.820 ± 0.064 | 0.815 ± 0.060 | 0.817 ± 0.053 | 0.778 ± 0.065 | 0.846 ± 0.050 | 0.761 ± 0.088 |
| wavlm_base_plus@full | 0.697 ± 0.059 | 0.807 ± 0.063 | 0.824 ± 0.057 | 0.791 ± 0.063 | 0.784 ± 0.067 | 0.831 ± 0.050 | 0.758 ± 0.076 |
| wavlm_large | 0.715 ± 0.065 | 0.835 ± 0.059 | 0.857 ± 0.055 | 0.857 ± 0.055 | 0.834 ± 0.045 | 0.864 ± 0.046 | 0.786 ± 0.071 |
| wavlm_large@full | 0.728 ± 0.068 | 0.827 ± 0.062 | 0.860 ± 0.051 | 0.852 ± 0.053 | 0.829 ± 0.046 | 0.852 ± 0.050 | 0.790 ± 0.066 |

## C ızgara ucu (seçilen C'nin 1e-5 ya da 1e2 olduğu fold oranı; yüksekse ızgara dar)

beats: 0.00, beats@full: 0.00, cnn10: 0.00, cnn10@full: 0.00, cnn14: 0.12, cnn14@full: 0.01, cnn14_16k: 0.16, cnn14_16k@full: 0.06, wavlm_base_plus: 0.00, wavlm_base_plus@full: 0.00, wavlm_large: 0.00, wavlm_large@full: 0.00
