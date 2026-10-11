# Split manifesti (otomatik üretildi; katılımcı ID'si içermez)

- Kohort: 342 katılımcı (283 astım / 59 sağlıklı); son sağlıklı kayıt günü 2024-04-02
- Görev başına katılımcı: {'slot1': 342, 'slot2': 342, 'slot3': 342, 'slot4': 341, 'slot5': 342, 'slot6': 342, 'slot7': 342}
- Gruplama: kayıt günü (dış ve iç fold'lar; etikete göre dengeli, grouped_folds — sklearn'den bağımsız); kayıt günü grubu sayısı 66
- 3 tekrar × 5 dış fold; her dış train içinde 5 iç fold (iç fold 0 = doğrulama)
- Tabakalar (etiket|dönem|yaş grubu, birleştirmeden sonra): {'0|early|<=40': 32, '0|early|>40': 27, '1|early|<=40': 31, '1|early|>40': 67, '1|late|<=40': 53, '1|late|>40': 132}
- Birleştirilen seyrek tabakalar: [{'stratum': '0|early|unknown', 'n': 1, 'merged_into': '0|early|<=40'}, {'stratum': '0|unknown|unknown', 'n': 1, 'merged_into': '0|early|<=40'}]
- Sürümler: {'python': '3.13.16', 'scikit-learn': '1.6.1', 'numpy': '2.1.3', 'pandas': '2.2.3'}

## Dosya sha256'ları (Colab'da yeniden üretilen dosyalar bunlarla aynı olmalı)

- `outer_r0.csv`: `d46c8875fb87ebc0d56dfd9a65213cd09831ccca681ac37d03d38a0e6e791a39`
- `outer_r1.csv`: `70753773c40c5f4e0e06e0ed02e84a8de082c2c3cc844f50b87d331cb23a1bf6`
- `outer_r2.csv`: `07179c37b36fba067d398241969550aa4578e752cde8fd4cf3a27a5e8ae1a184`

## Tekrar r0

| fold | test n | astım | sağlıklı | erken / geç / bilinmiyor | ≤40 / >40 / bilinmiyor | 7 görevin hepsi | öğleden önce astım / sağlıklı (betimsel) | train (val hariç) | iç val astım / sağlıklı |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 70 | 57 | 13 | 24 / 46 / 0 | 25 / 45 / 0 | 69 | 40 / 5 | 216 | 46 / 10 |
| 1 | 68 | 56 | 12 | 37 / 31 / 0 | 18 / 49 / 1 | 68 | 42 / 0 | 219 | 45 / 10 |
| 2 | 69 | 57 | 12 | 36 / 33 / 0 | 26 / 43 / 0 | 69 | 42 / 2 | 217 | 46 / 10 |
| 3 | 67 | 56 | 11 | 36 / 30 / 1 | 19 / 47 / 1 | 67 | 35 / 2 | 219 | 46 / 10 |
| 4 | 68 | 57 | 11 | 23 / 45 / 0 | 26 / 42 / 0 | 68 | 37 / 3 | 218 | 46 / 10 |

## Tekrar r1

| fold | test n | astım | sağlıklı | erken / geç / bilinmiyor | ≤40 / >40 / bilinmiyor | 7 görevin hepsi | öğleden önce astım / sağlıklı (betimsel) | train (val hariç) | iç val astım / sağlıklı |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 70 | 57 | 13 | 30 / 40 / 0 | 26 / 44 / 0 | 70 | 40 / 5 | 217 | 46 / 9 |
| 1 | 68 | 56 | 12 | 40 / 28 / 0 | 25 / 43 / 0 | 68 | 38 / 3 | 219 | 46 / 9 |
| 2 | 69 | 57 | 12 | 43 / 25 / 1 | 22 / 45 / 2 | 68 | 45 / 0 | 218 | 46 / 9 |
| 3 | 67 | 56 | 11 | 20 / 47 / 0 | 18 / 49 / 0 | 67 | 31 / 1 | 218 | 46 / 11 |
| 4 | 68 | 57 | 11 | 23 / 45 / 0 | 23 / 45 / 0 | 68 | 42 / 3 | 218 | 46 / 10 |

## Tekrar r2

| fold | test n | astım | sağlıklı | erken / geç / bilinmiyor | ≤40 / >40 / bilinmiyor | 7 görevin hepsi | öğleden önce astım / sağlıklı (betimsel) | train (val hariç) | iç val astım / sağlıklı |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 69 | 57 | 12 | 34 / 35 / 0 | 33 / 36 / 0 | 68 | 36 / 3 | 217 | 46 / 10 |
| 1 | 69 | 57 | 12 | 44 / 25 / 0 | 20 / 48 / 1 | 69 | 37 / 0 | 217 | 46 / 10 |
| 2 | 68 | 56 | 12 | 28 / 40 / 0 | 22 / 46 / 0 | 68 | 42 / 2 | 219 | 45 / 10 |
| 3 | 68 | 56 | 12 | 20 / 47 / 1 | 18 / 49 / 1 | 68 | 37 / 2 | 219 | 45 / 10 |
| 4 | 68 | 57 | 11 | 30 / 38 / 0 | 21 / 47 / 0 | 68 | 44 / 5 | 218 | 46 / 10 |
