# G1 — dondurulmuş problar gün-gruplu split'lerde (EXP-016DG) ve T1 (otomatik üretildi; katılımcı ID'si içermez)

**Okuma kuralı (önceden yazıldı):** ŝ_DG ile T1-PLR p < 0.05 ve OR > 1 → gün parmak izi yoluyla açıklanmıyor (PHASE3_DESIGN 5.1). CV3 ve Holm betimsel. Kohort 339 (282 / 57); 3 tekrar × 5 fold, gün-gruplu.

| kol | E1 fold AUC (gün-gruplu) | T1 PLR p | OR / SD [bootstrap %95] | CV3 p | Holm (betimsel) | Aşama 1 (katılımcı-gruplu): p · OR | G1 okuması |
|---|---|---|---|---|---|---|---|
| cnn10 | 0.823 ± 0.093 | 0.004 | 2.293 [1.399, 7.168] | 0.082 | 0.012 | <0.001 · 3.064 | gün parmak izi yoluyla açıklanmıyor |
| cnn14 | 0.767 ± 0.084 | 0.008 | 2.008 [1.200, 4.357] | 0.026 | 0.016 | 0.006 · 2.145 | gün parmak izi yoluyla açıklanmıyor |
| cnn14_16k | 0.698 ± 0.092 | 0.266 | 1.427 [0.834, 2.947] | 0.349 | 0.266 | 0.145 · 1.551 | — |
| beats | 0.884 ± 0.057 | <0.001 | 3.304 [1.983, 12.106] | 0.035 | 0.001 | <0.001 · 4.691 | gün parmak izi yoluyla açıklanmıyor |
| wavlm_base_plus | 0.864 ± 0.066 | 0.002 | 2.540 [1.284, 9.677] | 0.120 | 0.008 | 0.002 · 2.747 | — |
| wavlm_large | 0.884 ± 0.061 | <0.001 | 3.437 [1.670, 16.040] | 0.054 | <0.001 | <0.001 · 3.834 | gün parmak izi yoluyla açıklanmıyor |
| mfcc_lr | 0.718 ± 0.067 | 0.040 | 1.753 [1.035, 4.196] | 0.271 | ref. | 0.009 · 2.105 | — |
| mfcc_mlp | 0.729 ± 0.081 | 0.049 | 1.783 [1.119, 6.878] | 0.445 | ref. | 0.022 · 2.030 | — |

- E1 değerleri katılımcı-gruplu Faz 2 sayılarıyla doğrudan karşılaştırılamaz: split'ler ve tekrar sayısı farklı; gün-gruplu split'te eğitim günleri daha az çeşitli.
- Bu tablo Faz 3'ün dondurulmuş baz çizgisidir (C3 eşleştirmesi aynı split'lerde yapılır).
