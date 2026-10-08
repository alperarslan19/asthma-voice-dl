# Audio audit (otomatik üretildi)

- Ses dosyası: 2393 / toplam dosya 2393; ses olmayan: []
- Formatlar: {"ext": {".m4a": 2393}, "container": {"mov,mp4,m4a,3gp,3g2,mj2": 2393}, "codec": {"aac": 2393}, "codec_profile": {"LC": 2393}, "sr": {"48000": 2392, "44100": 1}, "channels": {"1": 2393}, "n_video_streams": {"0": 2393}, "tag_major_brand": {"iso5": 1759, "M4A ": 634}, "tag_encoder": {"None": 1759, "Lavf59.16.100": 634}, "tag_apple_model": {"None": 2393}, "tag_apple_software": {"None": 2393}, "has_location_tag": {"False": 2393}, "n_probe_errors": 0, "n_decode_errors": 0, "n_fake_stereo": 0}
- Bitrate: {'count': 2393.0, 'mean': 178213.0, 'std': 12005.0, 'min': 141185.0, '25%': 174145.0, '50%': 184001.0, '75%': 185500.0, 'max': 196166.0}
- Etkin bant genişliği (kHz): {'count': 2393.0, 'mean': 17.18, 'std': 5.29, 'min': 11.27, '1%': 11.68, '5%': 12.08, '50%': 13.54, '95%': 24.0, '99%': 24.0, 'max': 24.0}
- Süre (s): {'count': 2393.0, 'mean': 10.37, 'std': 0.534, 'min': 4.352, '1%': 8.425, '5%': 9.472, '50%': 10.475, '95%': 10.88, '99%': 11.328, 'max': 13.611}
- Adlandırma: {'frac_matching_ID_slot_pattern': 1.0, 'n_pid_dir_conflict': 0, 'n_slot_out_of_range': 0, 'n_pid_not_in_clinical': 0}
- Eşleme: {"status_counts": {"OK": 2393, "MISSING": 43}, "participants_with_all_7_ok": 341, "participants_with_zero_files": [101043, 101149, 101345, 101346, 101347, 101348], "participants_partial": [101244], "paper_cohort_all_7_ok": 341, "paper_cohort_size": 344, "missing_by_slot": {"1": 6, "2": 6, "3": 6, "4": 7, "5": 6, "6": 6, "7": 6}, "multiple_examples": [], "n_unmatched_audio_files": 0, "unmatched_examples": [], "same_file_in_two_slots_within_participant": 0}
- Birebir kopyalar: {'file_sha256': {'n_groups': 0, 'n_groups_across_participants': 0}, 'pcm_md5': {'n_groups': 0, 'n_groups_across_participants': 0}}
- Yakın kopya: {'n_cross_participant_pairs_above_threshold': 0, 'cross_participant_corr_p99': 0.13766491413116455, 'max_corr': 0.5116568207740784}
- Dosya kayıt zamanı vs CSV tarihi: {'n_files_with_creation_time': 1759, 'n_compared': 1754, 'frac_same_day': 0.9874572405929305, 'frac_within_1_day': 0.9908779931584949, 'delta_days_quantiles': {0.0: -30.0, 0.05: 0.0, 0.5: 0.0, 0.95: 0.0, 1.0: 182.0}, 'n_participants_csv_date_missing_but_file_has_date': 1, 'n_distinct_creation_dates': 65}

## Meta veri: tekdüze mi, etikete / döneme göre değişiyor mu?

- ext: tekdüze (.m4a)
- container: tekdüze (mov,mp4,m4a,3gp,3g2,mj2)
- codec: tekdüze (aac)
- codec_profile: tekdüze (LC)
- **sr: DEĞİŞKEN** — etiket p=1, hastalarda dönem p=1 · etiket: {'44100.0': {'healthy': 0, 'asthma': 1}, '48000.0': {'healthy': 413, 'asthma': 1979}} · dönem: {'44100.0': {'early': 0, 'late': 1}, '48000.0': {'early': 693, 'late': 1286}}
- channels: tekdüze (1.0)
- n_video_streams: tekdüze (0.0)
- **tag_major_brand: DEĞİŞKEN** — etiket p=0.217, hastalarda dönem p=6e-10 · etiket: {'M4A ': {'healthy': 120, 'asthma': 514}, 'iso5': {'healthy': 293, 'asthma': 1466}} · dönem: {'M4A ': {'early': 238, 'late': 276}, 'iso5': {'early': 455, 'late': 1011}}
- **tag_encoder: DEĞİŞKEN** — etiket p=0.217, hastalarda dönem p=6e-10 · etiket: {'Lavf59.16.100': {'healthy': 120, 'asthma': 514}, 'None': {'healthy': 293, 'asthma': 1466}} · dönem: {'Lavf59.16.100': {'early': 238, 'late': 276}, 'None': {'early': 455, 'late': 1011}}
- tag_apple_make: tekdüze (None)
- tag_apple_model: tekdüze (None)
- tag_apple_software: tekdüze (None)
- **audio_handler: DEĞİŞKEN** — etiket p=0.217, hastalarda dönem p=6e-10 · etiket: {'Core Media Audio': {'healthy': 293, 'asthma': 1466}, 'SoundHandler': {'healthy': 120, 'asthma': 514}} · dönem: {'Core Media Audio': {'early': 455, 'late': 1011}, 'SoundHandler': {'early': 238, 'late': 276}}
- has_location_tag: tekdüze (False)

## Kayıt-koşulu ölçümleri ↔ etiket (en uç 15)

|   slot | feature             |   n |   auc_label |
|-------:|:--------------------|----:|------------:|
|      5 | creation_hour_local | 233 |       0.131 |
|      7 | creation_hour_local | 256 |       0.132 |
|      3 | creation_hour_local | 276 |       0.157 |
|      6 | creation_hour_local | 237 |       0.16  |
|      2 | creation_hour_local | 237 |       0.162 |
|      1 | creation_hour_local | 281 |       0.168 |
|      4 | creation_hour_local | 239 |       0.174 |
|      4 | duration_s          | 341 |       0.344 |
|      6 | duration_s          | 342 |       0.367 |
|      1 | speech_level_db     | 342 |       0.627 |
|      1 | rms_dbfs            | 342 |       0.624 |
|      6 | lead_silence_s      | 342 |       0.378 |
|      3 | duration_s          | 342 |       0.384 |
|      5 | lead_silence_s      | 342 |       0.394 |
|      7 | duration_s          | 342 |       0.395 |

## Kayıt-koşulu ölçümleri ↔ kayıt dönemi (yalnız hastalar) (en uç 15)

|   slot | feature             |   n |   auc_period_within_patients |
|-------:|:--------------------|----:|-----------------------------:|
|      7 | duration_s          | 342 |                        0.351 |
|      5 | lead_silence_s      | 342 |                        0.361 |
|      3 | creation_hour_local | 276 |                        0.637 |
|      6 | creation_hour_local | 237 |                        0.635 |
|      4 | trail_silence_s     | 341 |                        0.628 |
|      3 | noise_floor_db      | 342 |                        0.373 |
|      4 | duration_s          | 341 |                        0.377 |
|      3 | snr_proxy_db        | 342 |                        0.62  |
|      1 | rms_dbfs            | 342 |                        0.615 |
|      6 | noise_floor_db      | 342 |                        0.387 |
|      2 | creation_hour_local | 237 |                        0.609 |
|      1 | bandwidth_khz       | 342 |                        0.392 |
|      4 | lead_silence_s      | 341 |                        0.394 |
|      6 | audio_bit_rate      | 342 |                        0.603 |
|      3 | duration_s          | 342 |                        0.4   |

AUC 0.5 = ilişki yok. 'period' tablosu YALNIZ hastalarda: etiket sabitken ölçüm kayıt dönemini ayırıyorsa kayıt koşulları (ya da hasta profili) zamanla değişmiştir. Çoklu karşılaştırma: hipotez üretir, tek başına sonuç değildir.
