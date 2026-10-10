# EXP-016S — dondurulmuş gömme + lineer prob (otomatik üretildi; katılımcı ID'si içermez)

**Rol:** önceden belirlenmiş duyarlılık analizi (kısa kayıt dolgu politikası; yalnız tekrar 0; seçimde kullanılmaz)  
**Üst sınır uyarısı (D-028):** bütün AUC'ler kayıt bağlamı değerlendirilmeden hesaplandı. Aynı fold'larda yalnız bağlam (saat + saat² + tarih) ve yalnız yaş referansları aşağıda.

Split: ['outer_r0.csv'] · tekrar: 1 · test/train oranı (Nadeau–Bengio): 0.250

**İstatistik notu:** Bütün Nadeau–Bengio (NB) testleri 5 fold skoruna (1 tekrar × 5 dış fold) uygulanır. Bu skorlar **bağımsız değildir**: aynı tekrardaki fold'ların eğitim kümeleri büyük ölçüde örtüşür ve tekrarlar aynı katılımcıları yeniden böler. NB düzeltmesi (varyansa n_test/n_train terimi eklenir) bu bağımlılığı kabaca telafi eden sezgisel bir düzeltmedir; df = 4 yaklaşıktır. Düz t-testi burada fazla iyimser olurdu.

Değerlendirme birimi: **katılımcı**. Bütün kollar aynı split dosyalarında, aynı dış test katılımcılarında değerlendirildi (assert). Kayıt bağlamı / süre gibi meta veri karşılaştırmaları bu raporda yok → META-016.

## Füzyon (katılımcı düzeyi; 7 görev olasılığının ortalaması) — betimsel

| kol | AUC fold ort ± SD [p2.5–p97.5] | havuzlanmış AUC [%95 CI] | dengeli doğr. (iç-CV eşiği) | duyarlılık / özgüllük | Brier | ort. tahmin − oran | kal. eğimi |
|---|---|---|---|---|---|---|---|
| beats | 0.923 ± 0.042 [0.876–0.978] | 0.918 [0.881–0.949] | 0.823 ± 0.073 | 0.80 / 0.84 | 0.094 | -0.090 | 2.06 |
| beats@alt | 0.923 ± 0.041 [0.876–0.976] | 0.918 [0.880–0.949] | 0.821 ± 0.072 | 0.80 / 0.84 | 0.094 | -0.090 | 2.05 |
| cnn10 | 0.855 ± 0.071 [0.763–0.934] | 0.852 [0.796–0.899] | 0.744 ± 0.073 | 0.78 / 0.71 | 0.147 | -0.191 | 2.52 |
| cnn10@alt | 0.853 ± 0.069 [0.769–0.935] | 0.852 [0.796–0.899] | 0.771 ± 0.069 | 0.76 / 0.78 | 0.149 | -0.193 | 2.60 |
| cnn14 | 0.799 ± 0.071 [0.725–0.877] | 0.792 [0.723–0.853] | 0.696 ± 0.062 | 0.75 / 0.65 | 0.170 | -0.219 | 3.03 |
| cnn14@alt | 0.792 ± 0.074 [0.726–0.874] | 0.783 [0.716–0.844] | 0.685 ± 0.062 | 0.74 / 0.63 | 0.175 | -0.225 | 3.10 |
| cnn14_16k | 0.735 ± 0.041 [0.692–0.780] | 0.729 [0.662–0.789] | 0.642 ± 0.046 | 0.56 / 0.73 | 0.193 | -0.250 | 2.37 |
| cnn14_16k@alt | 0.728 ± 0.035 [0.691–0.769] | 0.722 [0.656–0.781] | 0.650 ± 0.032 | 0.52 / 0.78 | 0.192 | -0.245 | 2.19 |
| ref_age — referans çizgisi (D-028; test edilmez) | 0.679 ± 0.027 [0.648–0.714] | 0.680 [0.602–0.750] | 0.622 ± 0.050 (eşik 0.5) | 0.66 / 0.58 | 0.225 | -0.299 | 0.95 |
| ref_context — referans çizgisi (D-028; test edilmez) | 0.937 ± 0.024 [0.914–0.969] | 0.931 [0.900–0.957] | 0.841 ± 0.048 (eşik 0.5) | 0.86 / 0.82 | 0.106 | -0.141 | 0.83 |
| wavlm_base_plus | 0.888 ± 0.056 [0.808–0.945] | 0.890 [0.843–0.929] | 0.766 ± 0.055 | 0.75 / 0.78 | 0.104 | -0.110 | 2.01 |
| wavlm_base_plus@alt | 0.893 ± 0.057 [0.810–0.949] | 0.895 [0.849–0.933] | 0.797 ± 0.068 | 0.75 / 0.85 | 0.102 | -0.110 | 2.03 |
| wavlm_large | 0.913 ± 0.043 [0.860–0.969] | 0.912 [0.872–0.944] | 0.763 ± 0.026 | 0.73 / 0.80 | 0.097 | -0.102 | 2.08 |
| wavlm_large@alt | 0.913 ± 0.043 [0.860–0.969] | 0.912 [0.872–0.944] | 0.763 ± 0.026 | 0.73 / 0.80 | 0.097 | -0.102 | 2.08 |

## Önceden belirlenmiş duyarlılık: kısa kayıtlarda diğer dolgu politikası − birincil (füzyon, tekrar 0)

| a − b | fold ort. Δ ± SD | havuzlanmış Δ [%95 CI] | NB p | Holm p |
|---|---|---|---|---|
| cnn10@alt − cnn10 | -0.002 ± 0.009 | -0.000 [-0.008, +0.007] | 0.767 | 1.000 |
| cnn14@alt − cnn14 | -0.007 ± 0.020 | -0.009 [-0.024, +0.005] | 0.631 | 1.000 |
| cnn14_16k@alt − cnn14_16k | -0.007 ± 0.011 | -0.007 [-0.022, +0.008] | 0.378 | 1.000 |
| beats@alt − beats | -0.000 ± 0.001 | -0.000 [-0.001, +0.001] | 0.736 | 1.000 |
| wavlm_base_plus@alt − wavlm_base_plus | +0.005 ± 0.005 | +0.005 [+0.001, +0.010] | 0.188 | 1.000 |
| wavlm_large@alt − wavlm_large | +0.000 ± 0.000 | +0.000 [+0.000, +0.000] | 1.000 | 1.000 |

## Görev başına AUC (fold ortalaması ± SD; keşifsel, RQ4/RQ5)

| kol | aaa | araba | ana | ordu | gelecek | titiz | ünlem |
|---|---|---|---|---|---|---|---|
| beats | 0.760 ± 0.097 | 0.819 ± 0.074 | 0.811 ± 0.120 | 0.869 ± 0.034 | 0.811 ± 0.055 | 0.898 ± 0.035 | 0.847 ± 0.058 |
| beats@alt | 0.761 ± 0.098 | 0.819 ± 0.074 | 0.811 ± 0.120 | 0.869 ± 0.034 | 0.811 ± 0.055 | 0.898 ± 0.035 | 0.847 ± 0.058 |
| cnn10 | 0.719 ± 0.080 | 0.753 ± 0.045 | 0.757 ± 0.093 | 0.753 ± 0.115 | 0.724 ± 0.086 | 0.714 ± 0.081 | 0.751 ± 0.090 |
| cnn10@alt | 0.708 ± 0.079 | 0.753 ± 0.045 | 0.757 ± 0.093 | 0.753 ± 0.115 | 0.724 ± 0.086 | 0.714 ± 0.081 | 0.751 ± 0.090 |
| cnn14 | 0.737 ± 0.035 | 0.684 ± 0.085 | 0.693 ± 0.071 | 0.722 ± 0.108 | 0.644 ± 0.043 | 0.691 ± 0.038 | 0.674 ± 0.026 |
| cnn14@alt | 0.710 ± 0.028 | 0.684 ± 0.085 | 0.693 ± 0.071 | 0.722 ± 0.108 | 0.644 ± 0.043 | 0.691 ± 0.038 | 0.674 ± 0.026 |
| cnn14_16k | 0.679 ± 0.051 | 0.675 ± 0.107 | 0.680 ± 0.090 | 0.673 ± 0.069 | 0.634 ± 0.085 | 0.581 ± 0.047 | 0.687 ± 0.050 |
| cnn14_16k@alt | 0.654 ± 0.039 | 0.675 ± 0.107 | 0.680 ± 0.090 | 0.673 ± 0.069 | 0.634 ± 0.085 | 0.581 ± 0.047 | 0.687 ± 0.050 |
| wavlm_base_plus | 0.625 ± 0.071 | 0.831 ± 0.054 | 0.810 ± 0.088 | 0.810 ± 0.046 | 0.802 ± 0.063 | 0.863 ± 0.042 | 0.782 ± 0.095 |
| wavlm_base_plus@alt | 0.682 ± 0.058 | 0.831 ± 0.054 | 0.810 ± 0.088 | 0.810 ± 0.046 | 0.802 ± 0.063 | 0.863 ± 0.042 | 0.782 ± 0.095 |
| wavlm_large | 0.725 ± 0.071 | 0.844 ± 0.023 | 0.837 ± 0.065 | 0.862 ± 0.057 | 0.854 ± 0.041 | 0.871 ± 0.034 | 0.799 ± 0.099 |
| wavlm_large@alt | 0.725 ± 0.071 | 0.844 ± 0.023 | 0.837 ± 0.065 | 0.862 ± 0.057 | 0.854 ± 0.041 | 0.871 ± 0.034 | 0.799 ± 0.099 |

### Kısa kaydı olan katılımcılarda füzyon olasılığının değişimi (|alternatif − birincil|, tekrar 0)

Kısa kaydı olmayanlarda da küçük değişim beklenir: eğitim kümesindeki kısa kayıtlar değiştiği için model biraz değişir.

| backbone | etkilenen katılımcı | ort. \|Δp\| | en büyük \|Δp\| | etkilenmeyenlerde ort. \|Δp\| |
|---|---|---|---|---|
| cnn10 | 17 | 0.0331 | 0.0802 | 0.0073 |
| cnn14 | 17 | 0.0362 | 0.1353 | 0.0117 |
| cnn14_16k | 17 | 0.0303 | 0.0753 | 0.0105 |
| beats | 17 | 0.0028 | 0.0104 | 0.0008 |
| wavlm_base_plus | 17 | 0.0245 | 0.0954 | 0.0037 |
| wavlm_large | 17 | 0.0000 | 0.0000 | 0.0000 |

## C ızgara ucu (seçilen C'nin 1e-5 ya da 1e2 olduğu fold oranı; yüksekse ızgara dar)

beats: 0.00, beats@alt: 0.00, cnn10: 0.00, cnn10@alt: 0.00, cnn14: 0.11, cnn14@alt: 0.11, cnn14_16k: 0.09, cnn14_16k@alt: 0.09, wavlm_base_plus: 0.00, wavlm_base_plus@alt: 0.00, wavlm_large: 0.00, wavlm_large@alt: 0.00
