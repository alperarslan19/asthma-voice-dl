# BG-0b — Kalite kapısı: konuşma sızıntısı (cinsiyet) (EXP-032; otomatik üretildi, katılımcı ID'si içermez)

Gün-gruplu dış CV (3 tekrar × 5 fold, `splits_day`), iç CV ile C (neg log-loss), dengeli LR. AUC: fold ort ± SD; havuz AUC (3 tekrar ort.) ve tekrar-0 havuzunun katılımcı bootstrap %95 GA'sı.

## Görev: sex

| kol | n (pozitif) | fold AUC ort ± SD | havuz AUC | %95 GA (r0) | C sabit pay |
|---|---|---|---|---|---|
| A1|ltas_abs | 77 (21) | 0.650 ± 0.172 | 0.599 | [0.397, 0.698] | 0.07 |
| A1|ltas_shape | 77 (21) | 0.670 ± 0.146 | 0.647 | [0.521, 0.789] | 0.07 |
| A1|mfcc44 | 77 (21) | 0.618 ± 0.211 | 0.556 | [0.441, 0.740] | 0.07 |
| A2|ltas_abs | 105 (33) | 0.764 ± 0.092 | 0.762 | [0.635, 0.844] | 0.00 |
| A2|ltas_shape | 105 (33) | 0.781 ± 0.087 | 0.781 | [0.677, 0.867] | 0.00 |
| A2|mfcc44 | 105 (33) | 0.693 ± 0.124 | 0.714 | [0.590, 0.808] | 0.00 |
| B|ltas_abs | 341 (107) | 0.933 ± 0.036 | 0.929 | [0.902, 0.957] | 0.00 |
| B|ltas_shape | 341 (107) | 0.933 ± 0.036 | 0.928 | [0.902, 0.957] | 0.00 |
| B|mfcc44 | 341 (107) | 0.811 ± 0.061 | 0.811 | [0.757, 0.854] | 0.00 |
| ref_context | 340 (105) | 0.438 ± 0.043 | 0.440 | [0.391, 0.531] | 0.00 |
| speech_mfcc_pos_control | 342 (107) | 0.914 ± 0.034 | 0.903 | [0.841, 0.925] | 0.00 |

## Kapı G-b (konuşma sızıntısı)

Eşik: her özellik setinde cinsiyet AUC ≤ max(0.6, bağlam referansı + 0.05) = **0.600** (bağlam referansı 0.438).

| tür | en yüksek AUC | sonuç |
|---|---|---|
| A1 | 0.670 | **KALDI** |
| A2 | 0.781 | **KALDI** |
| B | 0.933 | **KALDI** |

Pozitif kontrol (konuşma MFCC'si, 7 birimin ortalaması) cinsiyeti yüksek AUC ile ayırmalı; ayırmıyorsa test güçsüzdür ve kapı yorumlanmaz.
