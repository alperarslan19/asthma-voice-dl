# Kayıt bağlamı analizi (EXP-003) — otomatik üretildi, katılımcı ID'si içermez

## A. Kodlama zincirleri

- Dosya: {'apple': 1759, 'ffmpeg': 634} · katılımcı içi karışım: {'apple+ffmpeg': 285, 'apple': 55, 'ffmpeg': 2}
- FFmpeg payı slota göre: {1: 0.178, 2: 0.307, 3: 0.193, 4: 0.299, 5: 0.319, 6: 0.307, 7: 0.251} · zincir ↔ yeniden deneme Fisher p = 0.197
- En düşük bant genişliği: {'apple': 11.27, 'ffmpeg': 11.58} kHz → önerilen alçak geçiren kesim: **11.0 kHz** (D-021)
- apple: bitrate medyan 184749, bant genişliği medyan 13.15 kHz, süre medyan 10.52 s, SR {'48000': 1758, '44100': 1}
- ffmpeg: bitrate medyan 159180, bant genişliği medyan 24.00 kHz, süre medyan 10.07 s, SR {'48000': 634}

## B. Tarih ve saat

- Tarih kontrolleri: {'n_participants_file_vs_csv_date_differ': 4, 'their_labels': {'asthma': 4}, 'delta_days': [-30, 1, 5, 182], 'n_csv_date_missing_filled_from_file': 1, 'n_with_audio_but_no_date': 1}
- Saati bilinen katılımcı: 340
- healthy: seans başlangıcı medyan 15.16 (IQR 13.87–16.45), öğleden önce %21, aralık 10.2–20.9 (n=58)
- asthma: seans başlangıcı medyan 11.09 (IQR 10.22–12.55), öğleden önce %70, aralık 8.8–20.1 (n=282)

## C. EXP-003 — sesi kullanmayan bağlam baseline'ları (lojistik regresyon, 5-fold × 20)

| kohort | özellik | n (astım/sağlıklı) | AUC ort ± SD | fold %2.5–97.5 |
|---|---|---|---|---|
| audio_cohort | hour | 340 (282/58) | 0.841 ± 0.061 | 0.70–0.94 |
| audio_cohort | date | 341 (283/58) | 0.806 ± 0.055 | 0.70–0.91 |
| audio_cohort | hour+date | 340 (282/58) | 0.931 ± 0.032 | 0.86–0.98 |
| audio_cohort | age | 340 (283/57) | 0.681 ± 0.076 | 0.52–0.81 |
| audio_cohort | age+hour | 339 (282/57) | 0.863 ± 0.057 | 0.73–0.94 |
| audio_cohort | age+hour+date | 339 (282/57) | 0.933 ± 0.034 | 0.86–0.98 |
| time_window | hour | 154 (97/57) | 0.873 ± 0.053 | 0.77–0.96 |
| time_window | date | 155 (98/57) | 0.558 ± 0.118 | 0.30–0.79 |
| time_window | hour+date | 154 (97/57) | 0.881 ± 0.055 | 0.78–0.97 |
| time_window | age | 155 (98/57) | 0.671 ± 0.080 | 0.52–0.82 |
| time_window | age+hour | 154 (97/57) | 0.891 ± 0.051 | 0.79–0.98 |
| time_window | age+hour+date | 154 (97/57) | 0.908 ± 0.044 | 0.82–0.98 |
| afternoon_only | hour | 132 (86/46) | 0.767 ± 0.095 | 0.54–0.92 |
| afternoon_only | date | 132 (86/46) | 0.862 ± 0.059 | 0.75–0.96 |
| afternoon_only | hour+date | 132 (86/46) | 0.912 ± 0.051 | 0.80–1.00 |
| afternoon_only | age | 131 (86/45) | 0.665 ± 0.091 | 0.49–0.84 |
| afternoon_only | age+hour | 131 (86/45) | 0.796 ± 0.079 | 0.63–0.94 |
| afternoon_only | age+hour+date | 131 (86/45) | 0.912 ± 0.052 | 0.83–1.00 |

## D. Değerlendirme tasarımlarının dengesi

Her sütun: o tasarımın tabakaları İÇİNDE değişkenin etiketi ayırma gücü (tabakalı AUC). 0.5 = tam denge; 0.5'ten uzak = artık confounding.

| tasarım | astım/sağlıklı | çift | saat AUC | tarih AUC | yaş AUC | ±%95 (gerçek AUC 0.75) | P(tahmin > 0.5) |
|---|---|---|---|---|---|---|---|
| E1  tam kohort | 283/59 | 16697 | 0.158 | 0.808 | 0.681 | 0.069 | 1.0 |
| E1h tam kohort + aynı 1 saat dilimi | 208/51 | 1399 | 0.422 | 0.823 | 0.708 | 0.094 | 1.0 |
| E2  zaman penceresi | 98/57 | 5586 | 0.125 | 0.436 | 0.671 | 0.079 | 1.0 |
| E2h zaman penceresi + aynı 1 saat dilimi | 65/50 | 395 | 0.378 | 0.375 | 0.704 | 0.125 | 1.0 |
| E3  aynı gün | 39/52 | 177 | 0.103 | 0.5 | 0.711 | 0.121 | 1.0 |
| E3h aynı gün + aynı 1 saat dilimi | 9/14 | 14 | 0.214 | 0.5 | 0.929 | 0.25 | 0.936 |

## E. EXP-004 — Saatin kaba akustik izi (yalnız aynı etiket içinde; sabah vs öğleden sonra)

- asthma (n=282, öğleden sonra 86): tüm ölçümler birlikte CV AUC 0.549 ± 0.067; şanstan ayrılan tekil ölçüm: yok
- healthy (n=58, öğleden sonra 46): tüm ölçümler birlikte CV AUC 0.525 ± 0.179; şanstan ayrılan tekil ölçüm: yok

### E-b. EXP-004b — Hastalarda sabah ve öğleden sonra grupları klinik olarak farklı mı?

- n = 282 hasta (öğleden sonra 86); klinik profil → öğleden sonra CV AUC 0.528 ± 0.071 (fold %2.5–97.5: 0.37–0.64)
- Tek değişkenli p < 0.05 (düzeltmesiz): {'period(early/late)': 0.0227}
- Tüm tek değişkenli p'ler: {'asthma_step': 0.0847, 'sft_dx': 0.9462, 'smoking': 0.1227, 'sex': 1.0, 'gina_control': 0.1703, 'ics_use': 0.3698, 'allergic_rhinitis': 0.2529, 'age': 0.7636, 'act_total': 0.8875, 'fev1_pct': 0.4059, 'fvc_pct': 0.5483, 'fev1_fvc': 0.4001, 'period(early/late)': 0.0227}

## F. Her tasarımın 'yalnız bağlam' referansı (gerçek şans çizgisi)

Bağlam-only lojistik modelin OOF skoru (5-fold × 20) her tasarımın tabakaları içinde değerlendirildi. Bağlamı mükemmel kodlayan ama hiç astım bilgisi taşımayan bir ses modeli bu AUC'leri alırdı; bir ses modelinin o tasarımdaki sonucu 0.5 ile değil bu sayıyla karşılaştırılmalı.

| tasarım | age | hour+date | age+hour+date |
|---|---|---|---|
| E1  tam kohort | 0.674 ± 0.005 | 0.930 ± 0.002 | 0.930 ± 0.003 |
| E1h tam kohort + aynı 1 saat dilimi | 0.703 ± 0.005 | 0.817 ± 0.004 | 0.854 ± 0.008 |
| E2  zaman penceresi | 0.665 ± 0.008 | 0.810 ± 0.004 | 0.827 ± 0.004 |
| E2h zaman penceresi + aynı 1 saat dilimi | 0.703 ± 0.009 | 0.351 ± 0.016 | 0.497 ± 0.020 |
| E3  aynı gün | 0.691 ± 0.021 | 0.885 ± 0.009 | 0.918 ± 0.008 |
| E3h aynı gün + aynı 1 saat dilimi | 0.914 ± 0.029 | 0.639 ± 0.091 | 0.861 ± 0.075 |
