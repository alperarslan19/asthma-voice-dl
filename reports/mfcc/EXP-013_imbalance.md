# EXP-013 — Etiket dengesizliğiyle başa çıkma yöntemleri (keşifsel; katılımcı ID'si içermez)

Önbellek özellikleri (EXP-011 ile aynı), split dosyaları 5×5. Eşik: iç 5-fold OOF tahminlerinde dengeli doğruluğu en yükselten değer. Kalibrasyon: Brier (düşük iyi), ortalama tahmin − gerçek astım oranı (0 iyi), kalibrasyon eğimi (1 iyi). Bütün AUC'ler üst sınırdır (D-028).

## Füzyon (7 görev)

| model | yöntem | AUC ort ± SD | havuzlanmış AUC [%95 CI] | dengeli doğr. @0.5 | dengeli doğr. @iç-CV eşiği | Brier | ort. tahmin − oran | kalibrasyon eğimi |
|---|---|---|---|---|---|---|---|---|
| LR | SMOTE | 0.755 ± 0.073 | 0.768 [0.702–0.826] | 0.669 | 0.678 | 0.188 | -0.247 | 2.10 |
| LR | class_weight | 0.771 ± 0.078 | 0.778 [0.713–0.836] | 0.667 | 0.696 | 0.185 | -0.244 | 2.00 |
| LR | none | 0.762 ± 0.069 | 0.776 [0.713–0.835] | 0.500 | 0.698 | 0.134 | -0.000 | 3.03 |
| LR | random_oversampling | 0.763 ± 0.074 | 0.778 [0.714–0.834] | 0.662 | 0.687 | 0.189 | -0.250 | 2.22 |
| MLP | SMOTE | 0.770 ± 0.058 | 0.788 [0.725–0.844] | 0.567 | 0.696 | 0.123 | +0.005 | 0.86 |
| MLP | class_weight | 0.775 ± 0.053 | 0.790 [0.729–0.846] | 0.560 | 0.686 | 0.122 | -0.009 | 1.03 |
| MLP | none | 0.772 ± 0.056 | 0.788 [0.724–0.843] | 0.516 | 0.695 | 0.125 | +0.042 | 1.02 |
| MLP | random_oversampling | 0.769 ± 0.056 | 0.786 [0.723–0.844] | 0.550 | 0.681 | 0.123 | +0.012 | 0.89 |

## Eşleştirilmiş fark (füzyon, havuzlanmış AUC; katılımcı bootstrap)

| model | karşılaştırma | ΔAUC [%95 CI] |
|---|---|---|
| LR | class_weight − none (füzyon) | +0.002 [-0.019, +0.023] |
| LR | SMOTE − none (füzyon) | -0.008 [-0.032, +0.014] |
| LR | random_oversampling − none (füzyon) | +0.002 [-0.016, +0.019] |
| MLP | class_weight − none (füzyon) | +0.002 [-0.008, +0.012] |
| MLP | SMOTE − none (füzyon) | +0.000 [-0.016, +0.015] |
| MLP | random_oversampling − none (füzyon) | -0.001 [-0.012, +0.010] |

## Görev başına AUC (fold ortalaması)

| görev | LR none | LR class_weight | LR SMOTE | LR random_oversampling | MLP none | MLP class_weight | MLP SMOTE | MLP random_oversampling |
|---|---|---|---|---|---|---|---|---|
| aaa | 0.634 | 0.647 | 0.640 | 0.637 | 0.640 | 0.639 | 0.635 | 0.639 |
| araba | 0.686 | 0.691 | 0.676 | 0.684 | 0.672 | 0.667 | 0.668 | 0.666 |
| ana | 0.656 | 0.681 | 0.650 | 0.672 | 0.666 | 0.664 | 0.666 | 0.669 |
| ordu | 0.651 | 0.649 | 0.643 | 0.646 | 0.633 | 0.620 | 0.626 | 0.618 |
| gelecek | 0.647 | 0.647 | 0.662 | 0.646 | 0.663 | 0.664 | 0.667 | 0.665 |
| titiz | 0.670 | 0.665 | 0.677 | 0.674 | 0.718 | 0.714 | 0.712 | 0.713 |
| ünlem | 0.631 | 0.639 | 0.638 | 0.636 | 0.649 | 0.649 | 0.642 | 0.643 |
