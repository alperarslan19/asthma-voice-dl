# META-016 — meta veri / confounder izleme (AYRI rapor; ana deney akışına girmez; katılımcı ID'si içermez)

Bu rapordaki hiçbir sayı EXP-016'nın onaylayıcı kararına, D-035 seçimine ya da katman seçimine girmez. Kayıt bağlamının etkisinin asıl değerlendirmesi model geliştirme bitince yapılacak (D-028).

## 1. Bağlam bağımlılığı (EXP-014 ölçüleri; etiket sabitken skorun bağlamı ayırma gücü, 0.5 = bağımlılık yok)

| kol | hastalarda geç vs erken | hastalarda sabah vs öğleden sonra | sağlıklılarda sabah vs öğleden sonra |
|---|---|---|---|
| cnn10 | 0.524 | 0.474 | 0.605 |
| cnn14 | 0.484 | 0.471 | 0.652 |
| cnn14_16k | 0.484 | 0.503 | 0.667 |
| beats | 0.549 | 0.498 | 0.697 |
| wavlm_base_plus | 0.546 | 0.505 | 0.614 |
| wavlm_large | 0.537 | 0.494 | 0.589 |
| mfcc_lr | 0.557 | 0.525 | 0.464 |
| mfcc_mlp | 0.554 | 0.498 | 0.467 |
| ref_context | 0.939 | 0.740 | 1.000 |
| ref_age | 0.510 | 0.511 | 0.448 |

## 2. Kayıt süresi (kırpma sonrası, önbellekten)

| görev | etiket | n | medyan [Q1–Q3] s | < 4 s kayıt |
|---|---|---|---|---|
| aaa | sağlıklı | 59 | 10.06 [9.24–10.50] | 7 |
| aaa | astım | 283 | 10.33 [9.78–10.56] | 10 |
| araba | sağlıklı | 59 | 10.29 [9.91–10.45] | 0 |
| araba | astım | 283 | 10.22 [9.91–10.44] | 0 |
| ana | sağlıklı | 59 | 10.27 [9.87–10.50] | 0 |
| ana | astım | 283 | 10.28 [9.96–10.45] | 0 |
| ordu | sağlıklı | 59 | 10.36 [9.94–10.51] | 0 |
| ordu | astım | 282 | 10.17 [9.88–10.41] | 0 |
| gelecek | sağlıklı | 59 | 10.28 [9.89–10.50] | 0 |
| gelecek | astım | 283 | 10.25 [10.00–10.45] | 0 |
| titiz | sağlıklı | 59 | 10.28 [9.98–10.50] | 0 |
| titiz | astım | 283 | 10.24 [9.93–10.43] | 0 |
| ünlem | sağlıklı | 59 | 10.35 [10.00–10.55] | 0 |
| ünlem | astım | 283 | 10.29 [9.97–10.47] | 0 |

Toplam < 4 s kayıt: 17.

## 3. Yalnız-süre referans çizgisi (D-005; katılımcının ortalama süresi + kısa kayıt sayısı; aynı dış fold'lar)

AUC fold ort 0.497 ± 0.080 · havuzlanmış 0.449 [0.354–0.543] (25 fold). 0.5'ten belirgin büyükse süre etiketle ilişkilidir ve modellerin süreyi dolaylı kullanması bir risk olur; yorum değerlendirme aşamasında.

## 4. Önceden belirlenmiş duyarlılık: kısa kaydı olan katılımcılar değerlendirmeden çıkarıldığında (modeller aynı)

Çıkarılan katılımcı: 17 {'astım': 10, 'sağlıklı': 7}

| kol | füzyon AUC fold ort (tümü → hariç) | havuzlanmış AUC (tümü → hariç) |
|---|---|---|
| cnn10 | 0.855 → 0.853 | 0.858 → 0.854 |
| cnn14 | 0.791 → 0.776 | 0.811 → 0.797 |
| cnn14_16k | 0.729 → 0.727 | 0.733 → 0.730 |
| beats | 0.921 → 0.922 | 0.923 → 0.924 |
| wavlm_base_plus | 0.884 → 0.886 | 0.886 → 0.889 |
| wavlm_large | 0.909 → 0.908 | 0.909 → 0.911 |
| mfcc_lr | 0.771 → 0.765 | 0.778 → 0.771 |
| mfcc_mlp | 0.770 → 0.769 | 0.788 → 0.787 |
