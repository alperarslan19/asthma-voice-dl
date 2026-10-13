# BG-0a — Arka plan envanteri (EXP-032, Adım 0a; otomatik üretildi, katılımcı ID'si içermez)

- Kayıt 2393, katılımcı 342 (astım 283 / sağlıklı 59); süre 771.9 s
- Önbellek kazancıyla uyum: medyan |fark| 0.000 dB, en büyük 0.000 dB (0'a yakın olmalı: aynı model görünümü)
- Kayıt başına: SNR medyan 33.35 [p10 16.56, p90 42.88] dB; taban medyan -47.17 [p10 -56.61, p90 -32.27] dBFS; konuşma oranı medyan 0.71 [p10 0.52, p90 0.99]
- Düşük SNR (≤ 20 dB) → A1 / A2 alınmayan kayıt: 375; sayısal sessizlik içeren kayıt: 85
- Kayıt başına arka plan (s): A1 medyan 0.00 [p10 0.00, p90 0.69]; A2 medyan 0.00 [p10 0.00, p90 1.20]
- B: katılımcı başına kullanılan (SNR > 20 dB) kayıt sayısı medyan 6.00 [p10 5.00, p90 7.00]

## Ses birimi başına medyan arka plan (s)

| slot | A1 | A2 |
|---|---|---|
| 1 | 0.0 | 0.0 |
| 2 | 0.0 | 0.0 |
| 3 | 0.0 | 0.0 |
| 4 | 0.0 | 0.0 |
| 5 | 0.0 | 0.0 |
| 6 | 0.0 | 0.24 |
| 7 | 0.0 | 0.0 |

## Dahil olma (A1 ≥ 2.0 s, A2 ≥ 3.0 s, B ≥ 1 geçerli kayıt)

| tür | sağlıklı | astım | astım-erken | astım-geç | fark (puan) |
|---|---|---|---|---|---|
| A1 | 4/59 | 73/283 | 23/98 | 50/185 | 19.0 ⚠ |
| A2 | 10/59 | 95/283 | 35/98 | 60/185 | 16.7 ⚠ |
| B | 58/59 | 283/283 | 98/98 | 185/185 | 1.7 |

Toplam süre (dahil olsun olmasın, s):

- A1: sağlıklı medyan 0.38 [p10 0.00, p90 1.71]; astım-erken medyan 1.03 [p10 0.00, p90 2.86]; astım-geç medyan 0.72 [p10 0.00, p90 4.68]
- A2: sağlıklı medyan 0.92 [p10 0.00, p90 4.16]; astım-erken medyan 1.97 [p10 0.00, p90 6.21]; astım-geç medyan 1.24 [p10 0.10, p90 8.45]

## A1 ↔ B spektrum tutarlılığı (kalite kapısı G-c)

- Katılımcı başına bantlar üzerinden medyan |A1 − B|: medyan 4.36 [p10 1.46, p90 9.95] dB (n = 77)
- Genel medyan 4.36 dB → **KALDI** (eşik ≤ 3.0 dB)

## Kör dinleme

- 30 katılımcı × 3 tür = 90 dosya Drive'da `listening/`; form `listening_sheet.csv`. Tür adları harfle gizlendi; anahtar ayrı dosyada.

Not: Bu adım astım etiketine dayalı bir başarım ölçmez; yalnız envanter ve kalite. Parametreler: `docs/BACKGROUND_DESIGN.md`, JSON'da `params`.
