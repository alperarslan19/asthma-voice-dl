# EXP-016 — dondurulmuş gömme + lineer prob (otomatik üretildi; katılımcı ID'si içermez)

**Rol:** onaylayıcı (füzyon, 6 backbone vs 2 MFCC tabanı) + keşifsel (görev başına, çiftler)  
**Üst sınır uyarısı (D-028):** bütün AUC'ler kayıt bağlamı değerlendirilmeden hesaplandı. Aynı fold'larda yalnız bağlam (saat + saat² + tarih) ve yalnız yaş referansları aşağıda.

Split: ['outer_r0.csv', 'outer_r1.csv', 'outer_r2.csv', 'outer_r3.csv', 'outer_r4.csv'] · tekrar: 5 · test/train oranı (Nadeau–Bengio): 0.250

**İstatistik notu:** Bütün Nadeau–Bengio (NB) testleri 25 fold skoruna (5 tekrar × 5 dış fold) uygulanır. Bu skorlar **bağımsız değildir**: aynı tekrardaki fold'ların eğitim kümeleri büyük ölçüde örtüşür ve tekrarlar aynı katılımcıları yeniden böler. NB düzeltmesi (varyansa n_test/n_train terimi eklenir) bu bağımlılığı kabaca telafi eden sezgisel bir düzeltmedir; df = 24 yaklaşıktır. Düz t-testi burada fazla iyimser olurdu.

Değerlendirme birimi: **katılımcı**. Bütün kollar aynı split dosyalarında, aynı dış test katılımcılarında değerlendirildi (assert). Kayıt bağlamı / süre gibi meta veri karşılaştırmaları bu raporda yok → META-016.

## Füzyon (katılımcı düzeyi; 7 görev olasılığının ortalaması) — betimsel

| kol | AUC fold ort ± SD [p2.5–p97.5] | havuzlanmış AUC [%95 CI] | dengeli doğr. (iç-CV eşiği) | duyarlılık / özgüllük | Brier | ort. tahmin − oran | kal. eğimi |
|---|---|---|---|---|---|---|---|
| beats | 0.921 ± 0.046 [0.825–0.984] | 0.923 [0.886–0.953] | 0.811 ± 0.077 | 0.82 / 0.81 | 0.095 | -0.091 | 2.08 |
| cnn10 | 0.855 ± 0.047 [0.768–0.934] | 0.858 [0.807–0.903] | 0.757 ± 0.052 | 0.77 / 0.75 | 0.146 | -0.188 | 2.42 |
| cnn14 | 0.791 ± 0.063 [0.665–0.886] | 0.811 [0.747–0.869] | 0.721 ± 0.058 | 0.70 / 0.74 | 0.183 | -0.239 | 3.22 |
| cnn14_16k | 0.729 ± 0.067 [0.605–0.859] | 0.733 [0.666–0.794] | 0.657 ± 0.077 | 0.59 / 0.72 | 0.196 | -0.253 | 2.24 |
| mfcc_lr | 0.771 ± 0.078 [0.611–0.887] | 0.778 [0.713–0.836] | 0.696 ± 0.081 | 0.66 / 0.73 | 0.185 | -0.244 | 2.00 |
| mfcc_mlp | 0.770 ± 0.058 [0.670–0.858] | 0.788 [0.725–0.844] | 0.696 ± 0.059 | 0.70 / 0.69 | 0.123 | +0.005 | 0.86 |
| ref_age — referans çizgisi (D-028; test edilmez) | 0.682 ± 0.046 [0.615–0.757] | 0.679 [0.601–0.750] | 0.620 ± 0.041 (eşik 0.5) | 0.66 / 0.58 | 0.225 | -0.299 | 0.92 |
| ref_context — referans çizgisi (D-028; test edilmez) | 0.934 ± 0.022 [0.894–0.972] | 0.931 [0.899–0.958] | 0.839 ± 0.042 (eşik 0.5) | 0.86 / 0.82 | 0.106 | -0.141 | 0.84 |
| wavlm_base_plus | 0.884 ± 0.039 [0.815–0.944] | 0.886 [0.836–0.926] | 0.774 ± 0.047 | 0.77 / 0.78 | 0.109 | -0.118 | 1.87 |
| wavlm_large | 0.909 ± 0.041 [0.813–0.970] | 0.909 [0.868–0.942] | 0.791 ± 0.049 | 0.76 / 0.82 | 0.100 | -0.102 | 1.94 |

## A. Önceden belirlenmiş onaylayıcı analiz (RQ1; D-034 madde 6) — backbone füzyonu vs iki MFCC tabanı

NB düzeltilmiş t (25 bağımsız olmayan fold skoru; yukarıdaki not). Yönlü iddia (backbone > MFCC) için **tek yönlü** p; kesişim-birleşim p = max(p_LR, p_MLP); Holm (6); Holm-düzeltilmiş p **0.025** ile karşılaştırılır. Bu eşik sonuçlar görülmeden belirlenmiş, muhafazakâr bir karar eşiğidir: altı backbone'dan en az birini yanlışlıkla "MFCC'den iyi" ilan etme olasılığını (aile bazında) en çok 0.025'te tutar; tek bir test için iki yönlü 0.05'in pozitif kuyruğuna karşılık gelir ve "daha iyi" kararları için iki yönlü Holm 0.05'ten hiçbir zaman gevşek değildir (docs/PHASE2_DESIGN.md 5.3a). "Destekleniyor" ayrıca iki bootstrap CI'ının da 0'ı dışlamasını gerektirir. Parantez içinde iki yönlü p (tek yönlü test "daha kötü"yü kanıtlayamaz; negatif farklar betimsel olarak korunur). Destek: havuzlanmış ΔAUC katılımcı bootstrap CI ve tekrar başına DeLong (iki yönlü, medyan p).

| backbone | ΔAUC vs mfcc_lr: fold ort. · havuz [%95 CI] · tek yönlü p (iki yönlü) | ΔAUC vs mfcc_mlp: aynı | kesişim-birleşim p | Holm p | DeLong medyan p (LR / MLP) | sonuç |
|---|---|---|---|---|---|---|
| cnn10 | +0.084 · +0.080 [+0.018, +0.148] · 0.022 (0.045) | +0.085 · +0.070 [+0.006, +0.141] · 0.022 (0.044) | 0.022 | 0.067 | 0.014 / 0.019 | desteklenmiyor |
| cnn14 | +0.020 · +0.033 [-0.031, +0.103] · 0.325 (0.649) | +0.022 · +0.023 [-0.053, +0.101] · 0.316 (0.632) | 0.325 | 0.649 | 0.503 / 0.787 | desteklenmiyor |
| cnn14_16k | -0.042 · -0.045 [-0.113, +0.024] · 0.807 (0.385) | -0.040 · -0.055 [-0.130, +0.020] · 0.789 (0.422) | 0.807 | 0.807 | 0.128 / 0.137 | desteklenmiyor; iki tabandan da düşük (nokta tahmini) |
| beats | +0.150 · +0.145 [+0.090, +0.205] · 0.000 (0.000) | +0.151 · +0.135 [+0.078, +0.197] · 0.000 (0.000) | 0.000 | 0.001 | 0.000 / 0.000 | destekleniyor (üst sınır, geçici) |
| wavlm_base_plus | +0.113 · +0.107 [+0.054, +0.169] · 0.002 (0.003) | +0.114 · +0.098 [+0.040, +0.162] · 0.001 (0.003) | 0.002 | 0.006 | 0.000 / 0.001 | destekleniyor (üst sınır, geçici) |
| wavlm_large | +0.138 · +0.131 [+0.083, +0.186] · 0.001 (0.001) | +0.139 · +0.121 [+0.066, +0.179] · 0.000 (0.001) | 0.001 | 0.003 | 0.000 / 0.000 | destekleniyor (üst sınır, geçici) |

## B. Önceden listelenmiş keşifsel karşılaştırmalar (füzyon; Holm aile içinde; yorum keşifsel)

Yorum: farklı **hazır ses temsillerinin** karşılaştırması. Modeller ön-eğitim hedefi, etiket kullanımı (BEATs iter3+ AudioSet etiketlerini tokenizer öğretmeni üzerinden dolaylı kullanır), veri alanı, mimari, boyut ve bant genişliğinde aynı anda farklıdır; ön-eğitim yöntemi hakkında çıkarım yapılmaz (docs/PHASE2_DESIGN.md 9.6).

| a − b | fold ort. Δ ± SD | havuzlanmış Δ [%95 CI] | NB p | Holm p |
|---|---|---|---|---|
| cnn10 − cnn14 | +0.063 ± 0.044 | +0.046 [-0.005, +0.096] | 0.012 | 0.087 |
| cnn10 − cnn14_16k | +0.126 ± 0.056 | +0.125 [+0.064, +0.184] | 0.000 | 0.004 |
| cnn10 − beats | -0.066 ± 0.043 | -0.065 [-0.110, -0.022] | 0.009 | 0.073 |
| cnn10 − wavlm_base_plus | -0.029 ± 0.038 | -0.028 [-0.067, +0.011] | 0.162 | 0.325 |
| cnn10 − wavlm_large | -0.054 ± 0.042 | -0.051 [-0.088, -0.013] | 0.024 | 0.142 |
| cnn14 − cnn14_16k | +0.062 ± 0.049 | +0.078 [+0.031, +0.129] | 0.026 | 0.142 |
| cnn14 − beats | -0.129 ± 0.056 | -0.111 [-0.169, -0.055] | 0.000 | 0.003 |
| cnn14 − wavlm_base_plus | -0.093 ± 0.059 | -0.074 [-0.126, -0.022] | 0.008 | 0.072 |
| cnn14 − wavlm_large | -0.117 ± 0.055 | -0.098 [-0.151, -0.048] | 0.001 | 0.005 |
| cnn14_16k − beats | -0.191 ± 0.047 | -0.190 [-0.249, -0.130] | 0.000 | 0.000 |
| cnn14_16k − wavlm_base_plus | -0.155 ± 0.059 | -0.153 [-0.207, -0.095] | 0.000 | 0.001 |
| cnn14_16k − wavlm_large | -0.180 ± 0.048 | -0.176 [-0.233, -0.122] | 0.000 | 0.000 |
| beats − wavlm_base_plus | +0.037 ± 0.032 | +0.037 [+0.008, +0.070] | 0.047 | 0.190 |
| beats − wavlm_large | +0.012 ± 0.025 | +0.014 [-0.010, +0.038] | 0.398 | 0.398 |
| wavlm_large − wavlm_base_plus | +0.025 ± 0.023 | +0.024 [+0.003, +0.046] | 0.052 | 0.190 |

## Görev başına Δ (backbone − mfcc_lr; keşifsel, RQ4; Holm 42 karşılaştırma içinde)

| backbone | görev | fold ort. Δ ± SD | havuzlanmış Δ [%95 CI] | NB p (iki yönlü) | Holm p |
|---|---|---|---|---|---|
| cnn10 | aaa | +0.084 ± 0.104 | +0.092 [-0.001, +0.187] | 0.144 | 1.000 |
| cnn10 | araba | +0.065 ± 0.073 | +0.064 [-0.010, +0.138] | 0.110 | 1.000 |
| cnn10 | ana | +0.081 ± 0.075 | +0.073 [-0.016, +0.158] | 0.056 | 1.000 |
| cnn10 | ordu | +0.074 ± 0.091 | +0.075 [-0.017, +0.168] | 0.142 | 1.000 |
| cnn10 | gelecek | +0.079 ± 0.075 | +0.087 [+0.004, +0.169] | 0.064 | 1.000 |
| cnn10 | titiz | +0.050 ± 0.106 | +0.053 [-0.047, +0.152] | 0.386 | 1.000 |
| cnn10 | ünlem | +0.116 ± 0.075 | +0.118 [+0.034, +0.207] | 0.009 | 0.213 |
| cnn14 | aaa | +0.073 ± 0.105 | +0.079 [-0.011, +0.169] | 0.214 | 1.000 |
| cnn14 | araba | +0.001 ± 0.072 | -0.007 [-0.099, +0.086] | 0.970 | 1.000 |
| cnn14 | ana | +0.018 ± 0.075 | +0.013 [-0.074, +0.103] | 0.659 | 1.000 |
| cnn14 | ordu | +0.057 ± 0.098 | +0.066 [-0.027, +0.155] | 0.288 | 1.000 |
| cnn14 | gelecek | -0.008 ± 0.059 | +0.002 [-0.098, +0.103] | 0.813 | 1.000 |
| cnn14 | titiz | +0.011 ± 0.113 | -0.001 [-0.103, +0.096] | 0.858 | 1.000 |
| cnn14 | ünlem | +0.017 ± 0.089 | +0.020 [-0.075, +0.115] | 0.720 | 1.000 |
| cnn14_16k | aaa | +0.024 ± 0.117 | +0.028 [-0.073, +0.128] | 0.702 | 1.000 |
| cnn14_16k | araba | -0.017 ± 0.082 | -0.035 [-0.126, +0.061] | 0.712 | 1.000 |
| cnn14_16k | ana | +0.009 ± 0.086 | +0.027 [-0.070, +0.125] | 0.848 | 1.000 |
| cnn14_16k | ordu | +0.017 ± 0.123 | +0.017 [-0.080, +0.105] | 0.797 | 1.000 |
| cnn14_16k | gelecek | -0.027 ± 0.092 | -0.020 [-0.118, +0.078] | 0.595 | 1.000 |
| cnn14_16k | titiz | -0.084 ± 0.127 | -0.139 [-0.249, -0.028] | 0.231 | 1.000 |
| cnn14_16k | ünlem | +0.046 ± 0.100 | +0.032 [-0.053, +0.121] | 0.407 | 1.000 |
| beats | aaa | +0.096 ± 0.091 | +0.105 [+0.026, +0.189] | 0.062 | 1.000 |
| beats | araba | +0.125 ± 0.076 | +0.123 [+0.038, +0.208] | 0.005 | 0.146 |
| beats | ana | +0.134 ± 0.070 | +0.127 [+0.051, +0.208] | 0.002 | 0.049 |
| beats | ordu | +0.207 ± 0.094 | +0.206 [+0.124, +0.284] | 0.000 | 0.015 |
| beats | gelecek | +0.151 ± 0.071 | +0.157 [+0.080, +0.228] | 0.001 | 0.022 |
| beats | titiz | +0.230 ± 0.086 | +0.231 [+0.151, +0.312] | 0.000 | 0.002 |
| beats | ünlem | +0.205 ± 0.062 | +0.203 [+0.129, +0.275] | 0.000 | 0.000 |
| wavlm_base_plus | aaa | -0.008 ± 0.088 | +0.001 [-0.086, +0.086] | 0.860 | 1.000 |
| wavlm_base_plus | araba | +0.129 ± 0.062 | +0.127 [+0.057, +0.201] | 0.001 | 0.024 |
| wavlm_base_plus | ana | +0.134 ± 0.066 | +0.126 [+0.054, +0.200] | 0.001 | 0.029 |
| wavlm_base_plus | ordu | +0.168 ± 0.101 | +0.165 [+0.076, +0.249] | 0.005 | 0.137 |
| wavlm_base_plus | gelecek | +0.131 ± 0.090 | +0.145 [+0.063, +0.226] | 0.012 | 0.284 |
| wavlm_base_plus | titiz | +0.181 ± 0.087 | +0.168 [+0.067, +0.263] | 0.001 | 0.024 |
| wavlm_base_plus | ünlem | +0.121 ± 0.077 | +0.124 [+0.049, +0.197] | 0.007 | 0.190 |
| wavlm_large | aaa | +0.068 ± 0.081 | +0.077 [+0.004, +0.152] | 0.134 | 1.000 |
| wavlm_large | araba | +0.143 ± 0.065 | +0.143 [+0.067, +0.217] | 0.000 | 0.015 |
| wavlm_large | ana | +0.176 ± 0.053 | +0.170 [+0.102, +0.244] | 0.000 | 0.000 |
| wavlm_large | ordu | +0.208 ± 0.101 | +0.212 [+0.136, +0.283] | 0.001 | 0.027 |
| wavlm_large | gelecek | +0.187 ± 0.069 | +0.197 [+0.126, +0.268] | 0.000 | 0.001 |
| wavlm_large | titiz | +0.199 ± 0.086 | +0.198 [+0.105, +0.290] | 0.000 | 0.010 |
| wavlm_large | ünlem | +0.147 ± 0.060 | +0.146 [+0.075, +0.215] | 0.000 | 0.005 |

## Ünlü (aaa) − kelimelerin ortalaması (fold düzeyinde AUC farkı; keşifsel, RQ5; Holm kollar içinde)

| kol | ort. Δ ± SD | NB p | Holm p |
|---|---|---|---|
| cnn10 | -0.008 ± 0.083 | 0.854 | 1.000 |
| cnn14 | +0.041 ± 0.075 | 0.317 | 1.000 |
| cnn14_16k | +0.019 ± 0.096 | 0.722 | 1.000 |
| beats | -0.095 ± 0.073 | 0.024 | 0.144 |
| wavlm_base_plus | -0.168 ± 0.066 | 0.000 | 0.001 |
| wavlm_large | -0.124 ± 0.062 | 0.001 | 0.008 |
| mfcc_lr | -0.015 ± 0.076 | 0.719 | 1.000 |
| mfcc_mlp | -0.029 ± 0.098 | 0.587 | 1.000 |

## Görev başına AUC (fold ortalaması ± SD; keşifsel, RQ4/RQ5)

| kol | aaa | araba | ana | ordu | gelecek | titiz | ünlem |
|---|---|---|---|---|---|---|---|
| beats | 0.743 ± 0.072 | 0.816 ± 0.058 | 0.815 ± 0.073 | 0.856 ± 0.065 | 0.797 ± 0.070 | 0.896 ± 0.045 | 0.844 ± 0.063 |
| cnn10 | 0.731 ± 0.074 | 0.757 ± 0.052 | 0.762 ± 0.059 | 0.723 ± 0.073 | 0.725 ± 0.076 | 0.715 ± 0.070 | 0.755 ± 0.055 |
| cnn14 | 0.720 ± 0.063 | 0.693 ± 0.067 | 0.700 ± 0.067 | 0.706 ± 0.082 | 0.639 ± 0.076 | 0.676 ± 0.087 | 0.657 ± 0.062 |
| cnn14_16k | 0.671 ± 0.068 | 0.675 ± 0.069 | 0.690 ± 0.086 | 0.666 ± 0.081 | 0.620 ± 0.082 | 0.581 ± 0.090 | 0.685 ± 0.070 |
| mfcc_lr | 0.647 ± 0.087 | 0.691 ± 0.059 | 0.681 ± 0.075 | 0.649 ± 0.092 | 0.647 ± 0.066 | 0.665 ± 0.075 | 0.639 ± 0.065 |
| mfcc_mlp | 0.635 ± 0.091 | 0.668 ± 0.043 | 0.666 ± 0.064 | 0.626 ± 0.087 | 0.667 ± 0.068 | 0.712 ± 0.087 | 0.642 ± 0.078 |
| wavlm_base_plus | 0.639 ± 0.058 | 0.820 ± 0.064 | 0.815 ± 0.060 | 0.817 ± 0.053 | 0.778 ± 0.065 | 0.846 ± 0.050 | 0.761 ± 0.088 |
| wavlm_large | 0.715 ± 0.065 | 0.835 ± 0.059 | 0.857 ± 0.055 | 0.857 ± 0.055 | 0.834 ± 0.045 | 0.864 ± 0.046 | 0.786 ± 0.071 |

## C. D-035 — Faz 3 için fold başına seçim (yalnız o fold'un eğitim verisindeki iç doğrulama)

Seçim her dış fold'da, o fold'un eğitim katılımcılarının iç 5-fold OOF füzyon AUC'siyle yapıldı; dış test tahminleri seçime girmedi. Fold'lar farklı backbone seçebilir; Faz 3 her fold'da o fold'un seçtiğini fine-tune eder. Tablo `d035_selection.csv` (Drive) ile aynıdır.

| tekrar | fold | iç AUC beats | iç AUC wavlm_base_plus | iç AUC wavlm_large | eşit (<0.01) | seçilen |
|---|---|---|---|---|---|---|
| 0 | 0 | 0.875 | 0.837 | 0.878 | beats,wavlm_large | **beats** |
| 0 | 1 | 0.918 | 0.889 | 0.906 | beats | **beats** |
| 0 | 2 | 0.895 | 0.869 | 0.885 | beats,wavlm_large | **beats** |
| 0 | 3 | 0.931 | 0.886 | 0.915 | beats | **beats** |
| 0 | 4 | 0.935 | 0.899 | 0.918 | beats | **beats** |
| 1 | 0 | 0.933 | 0.904 | 0.921 | beats | **beats** |
| 1 | 1 | 0.883 | 0.871 | 0.892 | beats,wavlm_large | **beats** |
| 1 | 2 | 0.905 | 0.862 | 0.903 | beats,wavlm_large | **beats** |
| 1 | 3 | 0.926 | 0.865 | 0.890 | beats | **beats** |
| 1 | 4 | 0.935 | 0.889 | 0.918 | beats | **beats** |
| 2 | 0 | 0.913 | 0.874 | 0.909 | beats,wavlm_large | **beats** |
| 2 | 1 | 0.900 | 0.884 | 0.905 | beats,wavlm_large | **beats** |
| 2 | 2 | 0.913 | 0.886 | 0.900 | beats | **beats** |
| 2 | 3 | 0.949 | 0.905 | 0.923 | beats | **beats** |
| 2 | 4 | 0.895 | 0.874 | 0.899 | beats,wavlm_large | **beats** |
| 3 | 0 | 0.910 | 0.880 | 0.894 | beats | **beats** |
| 3 | 1 | 0.920 | 0.897 | 0.930 | wavlm_large | **wavlm_large** |
| 3 | 2 | 0.907 | 0.878 | 0.879 | beats | **beats** |
| 3 | 3 | 0.905 | 0.869 | 0.875 | beats | **beats** |
| 3 | 4 | 0.918 | 0.859 | 0.887 | beats | **beats** |
| 4 | 0 | 0.925 | 0.893 | 0.928 | beats,wavlm_large | **beats** |
| 4 | 1 | 0.909 | 0.858 | 0.875 | beats | **beats** |
| 4 | 2 | 0.903 | 0.863 | 0.875 | beats | **beats** |
| 4 | 3 | 0.929 | 0.912 | 0.898 | beats | **beats** |
| 4 | 4 | 0.921 | 0.878 | 0.907 | beats | **beats** |

## Seçim prosedürünün dürüst tahmini (iç içe; dış test AUC'si yalnız değerlendirmede okunur)

| kapsam | prosedür AUC ort ± SD | sonradan en iyi sabit kol (ort.) | kazananın laneti | seçim sıklıkları |
|---|---|---|---|---|
| D-035 (BEATs / WavLM adayları) | 0.920 ± 0.047 | beats (0.921) | +0.001 | {'beats': 24, 'wavlm_large': 1} |
| 6 backbone (keşifsel) | 0.920 ± 0.047 | beats (0.921) | +0.001 | {'beats': 24, 'wavlm_large': 1} |

## Regresyon testi (D-034 madde 5)

mfcc_lr füzyonu 0.771; EXP-011 LR füzyonu 0.771; fark +0.000 → **GEÇTİ**

## C ızgara ucu (seçilen C'nin 1e-5 ya da 1e2 olduğu fold oranı; yüksekse ızgara dar)

beats: 0.00, cnn10: 0.00, cnn14: 0.12, cnn14_16k: 0.16, mfcc_lr: 0.00, wavlm_base_plus: 0.00, wavlm_large: 0.00
