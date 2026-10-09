# EXP-014 — Kayıt bağlamı dengesizliği SMOTE / ağırlıkla düzeltilebilir mi? (keşifsel; katılımcı ID'si içermez)

Önbellek özellikleri, LR, füzyon (7 görev), split dosyaları 5×5. Hücre = dönem (erken/geç; D-024) × seans başlangıcı (sabah/öğleden sonra).

## Veri yapısı: hücrelerde katılımcı sayısı (pozitiflik)

| hücre | astım | sağlıklı |
|---|---|---|
| early-AM | 76 | 12 |
| early-PM | 21 | 46 |
| early-unknown | 1 | 0 |
| late-AM | 120 | 0 |
| late-PM | 65 | 0 |
| unknown-unknown | 0 | 1 |

## Sonuçlar (füzyon; tekrar-ortalamalı OOF)

| varyant / referans | AUC tüm test | AUC yalnız erken dönem | hücre-içi AUC (erken-AM + erken-PM) | erken-AM | erken-PM | hastalarda skor: geç vs erken | hastalarda skor: sabah vs öğleden sonra | sağlıklılarda skor: sabah vs öğleden sonra |
|---|---|---|---|---|---|---|---|---|
| V0 sınıf ağırlığı (tüm veri) | 0.777 | 0.739 | 0.706 | 0.782 | 0.635 | 0.556 | 0.524 | 0.464 |
| V2 hücre-içi SMOTE (tüm veri) | 0.769 | 0.723 | 0.709 | 0.766 | 0.655 | 0.587 | 0.526 | 0.428 |
| V3 hücre-ters-eğilim ağırlığı (tüm veri) | 0.760 | 0.721 | 0.718 | 0.777 | 0.661 | 0.562 | 0.521 | 0.426 |
| V4 yalnız erken dönem + hücre ağırlığı | 0.707 | 0.673 | 0.649 | 0.700 | 0.600 | 0.548 | 0.516 | 0.505 |
| V5 yalnız erken dönem + hücre-içi SMOTE | 0.713 | 0.679 | 0.650 | 0.691 | 0.612 | 0.555 | 0.514 | 0.522 |
| ref: bağlam (saat+saat²+tarih) | 0.930 | 0.813 | 0.578 | 0.533 | 0.622 | 0.939 | 0.739 | 1.000 |
| ref: yaş | 0.678 | 0.666 | 0.649 | 0.685 | 0.614 | 0.514 | 0.511 | 0.446 |

Son üç sütun "bağlam bağımlılığı": etiket sabitken skorun bağlamı ayırma gücü (0.5 = bağımlılık yok). Hastalarda geç > 0.5 → geç dönem hastalarına daha yüksek 'astım' skoru.

## Eşleştirilmiş fark (V0'a göre; katılımcı bootstrap)

| karşılaştırma | küme | ΔAUC [%95 CI] |
|---|---|---|
| V2 hücre-içi SMOTE (tüm veri) − V0 | tüm test | -0.008 [-0.035, +0.018] |
| V2 hücre-içi SMOTE (tüm veri) − V0 | yalnız erken dönem | -0.016 [-0.053, +0.020] |
| V3 hücre-ters-eğilim ağırlığı (tüm veri) − V0 | tüm test | -0.017 [-0.055, +0.020] |
| V3 hücre-ters-eğilim ağırlığı (tüm veri) − V0 | yalnız erken dönem | -0.018 [-0.068, +0.030] |
| V4 yalnız erken dönem + hücre ağırlığı − V0 | tüm test | -0.070 [-0.118, -0.025] |
| V4 yalnız erken dönem + hücre ağırlığı − V0 | yalnız erken dönem | -0.066 [-0.125, -0.007] |
| V5 yalnız erken dönem + hücre-içi SMOTE − V0 | tüm test | -0.064 [-0.113, -0.017] |
| V5 yalnız erken dönem + hücre-içi SMOTE − V0 | yalnız erken dönem | -0.060 [-0.120, -0.001] |
