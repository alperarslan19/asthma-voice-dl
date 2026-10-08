# Ses önbelleği raporu (otomatik üretildi; katılımcı ID'si içermez)

- Kayıt: 2393, katılımcı: 342; slot başına {1: 342, 2: 342, 3: 342, 4: 341, 5: 342, 6: 342, 7: 342}
- Kaynak SR: {48000: 2392, 44100: 1}; zincir: {'apple': 1759, 'ffmpeg': 634}
- Kontroller: kaynak sha256 = denetim ✓, NaN/Inf yok ✓, tepe = -1.0 dBFS ✓, ofsetler bitişik ✓, ilk 20 kayıt yeniden işlendiğinde birebir aynı ✓
- Boyut: {'32k': 3.08, '16k': 1.54} GB; süre 626.0 s; sürümler {'python': '3.13.16', 'numpy': '2.1.3', 'scipy': '1.16.3', 'pandas': '2.2.3', 'ffmpeg': 'ffmpeg version 6.1.1-3ubuntu5 Copyright (c) 2000-2023 the FFmpeg developers'}

## Kenar kırpma (D-015)

- Baştan kırpılan (s): min 0.0 · p1 0.0 · p5 0.0 · medyan 0.05 · p95 0.594 · p99 0.92 · max 2.6
- Sondan kırpılan (s): min 0.0 · p1 0.0 · p5 0.0 · medyan 0.0 · p95 0.532 · p99 4.459 · max 8.188
- Kalan süre (s): min 2.275 · p1 4.212 · p5 8.917 · medyan 10.259 · p95 10.716 · p99 11.095 · max 13.085
- Kalan süresi 4 s'den kısa: 17 kayıt; 2 s'den kısa: 0 kayıt (D-015: eğitimde sıfırla doldurulur)
- Düşük SNR nedeniyle yalnız tepeye-göre eşik kullanılan kayıt payı: 0.122

## Normalizasyon kazancı (dB)

- 32k: min -2.399 · p1 -0.348 · p5 0.764 · medyan 6.316 · p95 14.824 · p99 20.729 · max 28.015
- 16k: min -2.223 · p1 -0.308 · p5 0.825 · medyan 6.358 · p95 14.915 · p99 21.085 · max 28.524

## Bant genişliği (D-021) — denetimle aynı ölçüt

| zincir | önce (48 kHz) medyan / max | sonra (32 kHz yolu) medyan / max |
|---|---|---|
| apple | 13.148 / 24.0 | 11.07 / 11.133 |
| ffmpeg | 24.0 / 24.0 | 11.07 / 11.125 |

Bant genişliğinden zincir ayrımı (AUC; 0.5 = ayrım yok): önce 0.865, sonra 0.486.
Not: Filtre 11 kHz'in üstünü herkes için aynı biçimde siler; "sonra" sütununda bütün değerler ~11.0–11.3 kHz aralığında olmalı. Sonraki AUC 0.5'ten farklıysa bunun nedeni, kodlayıcıların 11 kHz'in ALTINDAKİ spektral şekillendirmesidir; filtre onu silemez (D-021 riski). Gömme düzeyinde zincir probu değerlendirme aşamasında (D-028).

## Dosya sha256'ları

- `audio_16k.f32`: `eca6e99914c7656f911c444341c4541b24b5f819a867158b496c56402b8f74d6`
- `audio_32k.f32`: `fa9362336e2c6850fc9ce8291c17c60c2bae407d43f5dd5ba584c8180c66e876`
- `index.csv`: `e41ed32ab435f3f79772491957c2869c64a940e2fea511740dc0ecf990cc7be9`
