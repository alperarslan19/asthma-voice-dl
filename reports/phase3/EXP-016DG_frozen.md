# EXP-016DG — dondurulmuş gömme + lineer prob (otomatik üretildi; katılımcı ID'si içermez)

**Rol:** Faz 3 baz çizgisi (D-041 / D-042): EXP-016'nın AYNI probu, gün-gruplu split'lerde (splits_day); D-035 fold başına seçim bu split'lerin iç OOF tahminlerinden. Onaylayıcı değil; karşılaştırma testi yok  
**Üst sınır uyarısı (D-028):** bütün AUC'ler kayıt bağlamı değerlendirilmeden hesaplandı. Aynı fold'larda yalnız bağlam (saat + saat² + tarih) ve yalnız yaş referansları aşağıda.

Split: ['outer_r0.csv', 'outer_r1.csv', 'outer_r2.csv'] · tekrar: 3 · test/train oranı (Nadeau–Bengio): 0.250

**İstatistik notu:** Bütün Nadeau–Bengio (NB) testleri 15 fold skoruna (3 tekrar × 5 dış fold) uygulanır. Bu skorlar **bağımsız değildir**: aynı tekrardaki fold'ların eğitim kümeleri büyük ölçüde örtüşür ve tekrarlar aynı katılımcıları yeniden böler. NB düzeltmesi (varyansa n_test/n_train terimi eklenir) bu bağımlılığı kabaca telafi eden sezgisel bir düzeltmedir; df = 14 yaklaşıktır. Düz t-testi burada fazla iyimser olurdu.

Değerlendirme birimi: **katılımcı**. Bütün kollar aynı split dosyalarında, aynı dış test katılımcılarında değerlendirildi (assert). Kayıt bağlamı / süre gibi meta veri karşılaştırmaları bu raporda yok → META-016.

## Füzyon (katılımcı düzeyi; 7 görev olasılığının ortalaması) — betimsel

| kol | AUC fold ort ± SD [p2.5–p97.5] | havuzlanmış AUC [%95 CI] | dengeli doğr. (iç-CV eşiği) | duyarlılık / özgüllük | Brier | ort. tahmin − oran | kal. eğimi |
|---|---|---|---|---|---|---|---|
| beats | 0.884 ± 0.057 [0.803–0.987] | 0.890 [0.845–0.928] | 0.792 ± 0.080 | 0.82 / 0.77 | 0.104 | -0.094 | 1.87 |
| cnn10 | 0.823 ± 0.093 [0.681–0.972] | 0.821 [0.759–0.878] | 0.741 ± 0.080 | 0.78 / 0.70 | 0.152 | -0.187 | 2.07 |
| cnn14 | 0.767 ± 0.084 [0.630–0.886] | 0.779 [0.710–0.839] | 0.703 ± 0.083 | 0.69 / 0.72 | 0.182 | -0.233 | 2.60 |
| cnn14_16k | 0.698 ± 0.092 [0.546–0.820] | 0.697 [0.626–0.763] | 0.645 ± 0.072 | 0.49 / 0.80 | 0.198 | -0.251 | 1.90 |
| mfcc_lr | 0.718 ± 0.067 [0.620–0.841] | 0.726 [0.653–0.792] | 0.643 ± 0.059 | 0.71 / 0.58 | 0.189 | -0.240 | 1.64 |
| mfcc_mlp | 0.729 ± 0.081 [0.563–0.816] | 0.742 [0.677–0.804] | 0.659 ± 0.066 | 0.65 / 0.67 | 0.129 | +0.007 | 0.68 |
| ref_age — referans çizgisi (D-028; test edilmez) | 0.680 ± 0.137 [0.430–0.849] | 0.661 [0.575–0.733] | 0.620 ± 0.117 (eşik 0.5) | 0.66 / 0.58 | 0.228 | -0.296 | 0.77 |
| ref_context — referans çizgisi (D-028; test edilmez) | 0.942 ± 0.038 [0.883–0.991] | 0.921 [0.886–0.951] | 0.831 ± 0.053 (eşik 0.5) | 0.84 / 0.82 | 0.115 | -0.146 | 0.76 |
| wavlm_base_plus | 0.864 ± 0.066 [0.759–0.971] | 0.858 [0.801–0.907] | 0.750 ± 0.085 | 0.76 / 0.74 | 0.113 | -0.118 | 1.71 |
| wavlm_large | 0.884 ± 0.061 [0.780–0.978] | 0.887 [0.842–0.927] | 0.770 ± 0.063 | 0.77 / 0.77 | 0.103 | -0.095 | 1.70 |

## Görev başına AUC (fold ortalaması ± SD; keşifsel, RQ4/RQ5)

| kol | aaa | araba | ana | ordu | gelecek | titiz | ünlem |
|---|---|---|---|---|---|---|---|
| beats | 0.742 ± 0.092 | 0.792 ± 0.061 | 0.776 ± 0.054 | 0.841 ± 0.052 | 0.743 ± 0.090 | 0.863 ± 0.044 | 0.817 ± 0.073 |
| cnn10 | 0.722 ± 0.061 | 0.758 ± 0.101 | 0.753 ± 0.098 | 0.664 ± 0.074 | 0.696 ± 0.080 | 0.659 ± 0.092 | 0.732 ± 0.100 |
| cnn14 | 0.696 ± 0.040 | 0.682 ± 0.103 | 0.698 ± 0.072 | 0.671 ± 0.075 | 0.623 ± 0.105 | 0.656 ± 0.088 | 0.661 ± 0.055 |
| cnn14_16k | 0.625 ± 0.030 | 0.648 ± 0.105 | 0.666 ± 0.130 | 0.646 ± 0.075 | 0.612 ± 0.095 | 0.555 ± 0.101 | 0.663 ± 0.094 |
| mfcc_lr | 0.598 ± 0.095 | 0.700 ± 0.078 | 0.670 ± 0.105 | 0.659 ± 0.062 | 0.619 ± 0.076 | 0.601 ± 0.079 | 0.575 ± 0.096 |
| mfcc_mlp | 0.573 ± 0.105 | 0.670 ± 0.114 | 0.642 ± 0.109 | 0.622 ± 0.062 | 0.653 ± 0.089 | 0.663 ± 0.112 | 0.609 ± 0.088 |
| wavlm_base_plus | 0.639 ± 0.110 | 0.802 ± 0.032 | 0.813 ± 0.063 | 0.790 ± 0.065 | 0.758 ± 0.112 | 0.839 ± 0.055 | 0.748 ± 0.098 |
| wavlm_large | 0.707 ± 0.076 | 0.812 ± 0.068 | 0.847 ± 0.051 | 0.828 ± 0.068 | 0.812 ± 0.107 | 0.838 ± 0.062 | 0.757 ± 0.091 |

## C. D-035 — Faz 3 için fold başına seçim (yalnız o fold'un eğitim verisindeki iç doğrulama)

Seçim her dış fold'da, o fold'un eğitim katılımcılarının iç 5-fold OOF füzyon AUC'siyle yapıldı; dış test tahminleri seçime girmedi. Fold'lar farklı backbone seçebilir; Faz 3 her fold'da o fold'un seçtiğini fine-tune eder. Tablo `d035_selection.csv` (Drive) ile aynıdır.

| tekrar | fold | iç AUC beats | iç AUC wavlm_base_plus | iç AUC wavlm_large | eşit (<0.01) | seçilen |
|---|---|---|---|---|---|---|
| 0 | 0 | 0.879 | 0.857 | 0.878 | beats,wavlm_large | **beats** |
| 0 | 1 | 0.904 | 0.898 | 0.910 | beats,wavlm_large | **beats** |
| 0 | 2 | 0.882 | 0.856 | 0.869 | beats | **beats** |
| 0 | 3 | 0.891 | 0.872 | 0.892 | beats,wavlm_large | **beats** |
| 0 | 4 | 0.858 | 0.823 | 0.857 | beats,wavlm_large | **beats** |
| 1 | 0 | 0.847 | 0.841 | 0.862 | wavlm_large | **wavlm_large** |
| 1 | 1 | 0.857 | 0.862 | 0.868 | wavlm_base_plus,wavlm_large | **wavlm_base_plus** |
| 1 | 2 | 0.901 | 0.900 | 0.927 | wavlm_large | **wavlm_large** |
| 1 | 3 | 0.893 | 0.864 | 0.878 | beats | **beats** |
| 1 | 4 | 0.873 | 0.824 | 0.860 | beats | **beats** |
| 2 | 0 | 0.910 | 0.877 | 0.896 | beats | **beats** |
| 2 | 1 | 0.897 | 0.884 | 0.898 | beats,wavlm_large | **beats** |
| 2 | 2 | 0.862 | 0.851 | 0.863 | beats,wavlm_large | **beats** |
| 2 | 3 | 0.888 | 0.866 | 0.881 | beats,wavlm_large | **beats** |
| 2 | 4 | 0.852 | 0.822 | 0.855 | beats,wavlm_large | **beats** |

## Seçim prosedürünün dürüst tahmini (iç içe; dış test AUC'si yalnız değerlendirmede okunur)

| kapsam | prosedür AUC ort ± SD | sonradan en iyi sabit kol (ort.) | kazananın laneti | seçim sıklıkları |
|---|---|---|---|---|
| D-035 (BEATs / WavLM adayları) | 0.882 ± 0.062 | beats (0.884) | +0.002 | {'beats': 12, 'wavlm_large': 2, 'wavlm_base_plus': 1} |

## C ızgara ucu (seçilen C'nin 1e-5 ya da 1e2 olduğu fold oranı; yüksekse ızgara dar)

beats: 0.00, cnn10: 0.03, cnn14: 0.07, cnn14_16k: 0.17, mfcc_lr: 0.00, wavlm_base_plus: 0.00, wavlm_large: 0.00
