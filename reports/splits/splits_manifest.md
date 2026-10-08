# Split manifesti (otomatik üretildi; katılımcı ID'si içermez)

- Kohort: 342 katılımcı (283 astım / 59 sağlıklı); son sağlıklı kayıt günü 2024-04-02
- Görev başına katılımcı: {'slot1': 342, 'slot2': 342, 'slot3': 342, 'slot4': 341, 'slot5': 342, 'slot6': 342, 'slot7': 342}
- 5 tekrar × 5 dış fold; her dış train içinde 5 iç fold (iç fold 0 = doğrulama)
- Tabakalar (etiket|dönem|yaş grubu, birleştirmeden sonra): {'0|early|<=40': 32, '0|early|>40': 27, '1|early|<=40': 31, '1|early|>40': 67, '1|late|<=40': 53, '1|late|>40': 132}
- Birleştirilen seyrek tabakalar: [{'stratum': '0|early|unknown', 'n': 1, 'merged_into': '0|early|<=40'}, {'stratum': '0|unknown|unknown', 'n': 1, 'merged_into': '0|early|<=40'}]
- Sürümler: {'python': '3.13.16', 'scikit-learn': '1.9.1', 'numpy': '2.5.3', 'pandas': '3.0.5'}

## Dosya sha256'ları (Colab'da yeniden üretilen dosyalar bunlarla aynı olmalı)

- `outer_r0.csv`: `daa5f65b2ef805ba7535af1135a082bf00a58eab169660ae4d04b2ca602f71e5`
- `outer_r1.csv`: `42e394a3d4bf12cac036e00eb89e4e7a8dafd4879e5aff4733da9c1bc2161fe9`
- `outer_r2.csv`: `e3975a671a9d68afb6275af906fe6a699cc0fd6395a832d39f18a6d0c663d4cf`
- `outer_r3.csv`: `d7b2c8cdfad413756b1508202b6b347409102d85db7c9c4c1399b301ea4d0d53`
- `outer_r4.csv`: `312d3ea0864e9b3117f56f38ba61a0ddd3632ae3b382e5c6b7ed7cb9e55ee5fd`

## Tekrar r0

| fold | test n | astım | sağlıklı | erken / geç / bilinmiyor | ≤40 / >40 / bilinmiyor | 7 görevin hepsi | öğleden önce astım / sağlıklı (betimsel) | train (val hariç) | iç val astım / sağlıklı |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 69 | 57 | 12 | 32 / 37 / 0 | 23 / 46 / 0 | 69 | 38 / 3 | 218 | 46 / 9 |
| 1 | 69 | 57 | 12 | 31 / 37 / 1 | 22 / 46 / 1 | 68 | 41 / 4 | 218 | 46 / 9 |
| 2 | 68 | 56 | 12 | 31 / 37 / 0 | 23 / 45 / 0 | 68 | 39 / 1 | 219 | 45 / 10 |
| 3 | 68 | 56 | 12 | 31 / 37 / 0 | 22 / 45 / 1 | 68 | 36 / 2 | 219 | 45 / 10 |
| 4 | 68 | 57 | 11 | 31 / 37 / 0 | 24 / 44 / 0 | 68 | 42 / 2 | 219 | 45 / 10 |

## Tekrar r1

| fold | test n | astım | sağlıklı | erken / geç / bilinmiyor | ≤40 / >40 / bilinmiyor | 7 görevin hepsi | öğleden önce astım / sağlıklı (betimsel) | train (val hariç) | iç val astım / sağlıklı |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 69 | 57 | 12 | 32 / 37 / 0 | 23 / 46 / 0 | 69 | 38 / 3 | 218 | 46 / 9 |
| 1 | 69 | 57 | 12 | 32 / 37 / 0 | 22 / 46 / 1 | 69 | 39 / 4 | 218 | 46 / 9 |
| 2 | 68 | 56 | 12 | 30 / 37 / 1 | 22 / 45 / 1 | 67 | 36 / 2 | 219 | 45 / 10 |
| 3 | 68 | 56 | 12 | 31 / 37 / 0 | 23 / 45 / 0 | 68 | 39 / 3 | 219 | 45 / 10 |
| 4 | 68 | 57 | 11 | 31 / 37 / 0 | 24 / 44 / 0 | 68 | 44 / 0 | 219 | 45 / 10 |

## Tekrar r2

| fold | test n | astım | sağlıklı | erken / geç / bilinmiyor | ≤40 / >40 / bilinmiyor | 7 görevin hepsi | öğleden önce astım / sağlıklı (betimsel) | train (val hariç) | iç val astım / sağlıklı |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 69 | 57 | 12 | 32 / 37 / 0 | 22 / 46 / 1 | 69 | 38 / 1 | 218 | 46 / 9 |
| 1 | 69 | 57 | 12 | 32 / 37 / 0 | 23 / 46 / 0 | 68 | 40 / 2 | 218 | 46 / 9 |
| 2 | 68 | 56 | 12 | 30 / 37 / 1 | 22 / 45 / 1 | 68 | 42 / 2 | 219 | 45 / 10 |
| 3 | 68 | 56 | 12 | 31 / 37 / 0 | 23 / 45 / 0 | 68 | 37 / 4 | 219 | 45 / 10 |
| 4 | 68 | 57 | 11 | 31 / 37 / 0 | 24 / 44 / 0 | 68 | 39 / 3 | 219 | 45 / 10 |

## Tekrar r3

| fold | test n | astım | sağlıklı | erken / geç / bilinmiyor | ≤40 / >40 / bilinmiyor | 7 görevin hepsi | öğleden önce astım / sağlıklı (betimsel) | train (val hariç) | iç val astım / sağlıklı |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 69 | 57 | 12 | 32 / 37 / 0 | 23 / 46 / 0 | 69 | 39 / 4 | 218 | 46 / 9 |
| 1 | 69 | 57 | 12 | 32 / 37 / 0 | 23 / 46 / 0 | 69 | 35 / 3 | 218 | 46 / 9 |
| 2 | 68 | 56 | 12 | 31 / 37 / 0 | 23 / 45 / 0 | 68 | 46 / 2 | 219 | 45 / 10 |
| 3 | 68 | 56 | 12 | 30 / 37 / 1 | 21 / 45 / 2 | 68 | 34 / 1 | 219 | 45 / 10 |
| 4 | 68 | 57 | 11 | 31 / 37 / 0 | 24 / 44 / 0 | 67 | 42 / 2 | 219 | 45 / 10 |

## Tekrar r4

| fold | test n | astım | sağlıklı | erken / geç / bilinmiyor | ≤40 / >40 / bilinmiyor | 7 görevin hepsi | öğleden önce astım / sağlıklı (betimsel) | train (val hariç) | iç val astım / sağlıklı |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 69 | 57 | 12 | 32 / 37 / 0 | 23 / 46 / 0 | 69 | 37 / 1 | 218 | 46 / 9 |
| 1 | 69 | 57 | 12 | 32 / 37 / 0 | 22 / 46 / 1 | 69 | 40 / 3 | 218 | 46 / 9 |
| 2 | 68 | 56 | 12 | 31 / 37 / 0 | 23 / 45 / 0 | 67 | 42 / 4 | 219 | 45 / 10 |
| 3 | 68 | 56 | 12 | 31 / 37 / 0 | 23 / 45 / 0 | 68 | 43 / 3 | 219 | 45 / 10 |
| 4 | 68 | 57 | 11 | 30 / 37 / 1 | 23 / 44 / 1 | 68 | 34 / 1 | 219 | 45 / 10 |
