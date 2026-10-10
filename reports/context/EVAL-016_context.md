# EVAL-016 — dondurulmuş aile için bağlam değerlendirmesi, Aşama 1 (otomatik üretildi; katılımcı ID'si içermez)

**Plan:** `docs/EVAL016_PLAN.md` (D-025, D-037 KABUL). **Bu rapor Aşama 1'dir:** N3 (yalnız arka plan) Aşama 2'dedir; C1 hiçbir backbone için bu raporla tamamlanmış sayılmaz.

**Kohort:** EXP-016 kohortu 342 → T1 kohortu 339 (282 astım / 57 sağlıklı); saat / tarih / yaş eksik olduğu için dışlanan 3 {'astım': 1, 'sağlıklı': 2}; kayıt günü 65.

**ŝ:** 5 tekrarın dış-test füzyon logit'lerinin ortalaması; z-standardize, OR = ŝ'nin 1 SD'si başına. Bootstrap B = 2000; Spisak permütasyon = 1000; spline df = 4 (duyarlılık 6).

**Uygulama protokolü:** EVAL-016 uygulama v1.2 (D-038) — durum **KABUL** · T1 fit: firth · C2 kısmi test: within_label_permutation · C1 koşul 3: cluster_cv3 · N5 C1'de: False

**Çalıştırma kipi:** gerçek veri kipi (protokol KABUL, ayarlar sabit)

## 0. Kalibrasyon tanılaması (gerçek etiket + gerçek bağlam, sinyalsiz yapay skor, γ = 0; ses skoru kullanılmaz)

Yanlış pozitif oranları (iki yönlü). İdeal: 0.05 / 0.01. T1-PLR birincil test; CR1 = Firth sandviç (G/(G−1); betimsel); CV3 = gün-kümeli jackknife, t(G−1) (C1 koşul 3, D-038). S2 (basamak) ve S3 (ölçülmemiş gün etkisi) yanlış belirtilmiş model senaryolarıdır (SIM-001).

| senaryo | PLR p < 0.05 [Wilson %95] | PLR p < 0.01 | CR1 p < 0.05 | CV3 p < 0.05 [Wilson %95] | CV3 p < 0.01 | n_sim |
|---|---|---|---|---|---|---|
| S1_smooth | 0.046 [0.031, 0.068] | 0.010 | 0.102 | 0.026 [0.015, 0.044] | 0.002 | 500 |
| S2_step | 0.044 [0.029, 0.066] | 0.014 | 0.104 | 0.026 [0.015, 0.044] | 0.004 | 500 |
| S3_day | 0.148 [0.120, 0.182] | 0.054 | 0.162 | 0.036 [0.023, 0.056] | 0.012 | 500 |

**K2 geçerlilik kuralı (önceden yazıldı, D-038):** kalibrasyon tanılamasında CV3'ün iki yönlü p < 0.05 ret oranının Wilson %95 alt sınırı S1 ve S3'te ≤ 0.075 (Bradley'nin liberal ölçütü, 1.5α); değilse koşul 3 geçersiz → C1 'kurulamadı (K2)'. Sonuç: CV3 Wilson alt sınırları S1_smooth 0.015, S3_day 0.023 → **koşul 3 geçerli**.

## 1. T1 (L2, birincil) — onaylayıcı aile: 6 backbone, Holm; MFCC yalnız referans

Karar istatistiği: **Firth cezalı olabilirlik oranı testi** (PLR; EVAL016_PLAN S9 — tarih bağlamı etiketi neredeyse ayırdığı için klasik ML fit'i yakınsamayabilir). Klasik ML LRT yalnız karşılaştırma için.

| kol | PLR χ² | p (PLR) | Holm p | OR / SD [bootstrap %95 CI] | Firth Wald p | gün-kümeli CR1 p (betimsel) | **gün-kümeli CV3 p** (koşul 3) | OR df 6 (p) | ML LRT p (yakınsadı) | tekrar başına PLR aralığı | CV ΔAUC (betimsel) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cnn10 | 12.351 | <0.001 | 0.002 | 3.064 [1.804, 9.671] (başarısız 0) | 0.001 | <0.001 | 0.003 | 3.244 (<0.001) | <0.001 (hayır) | 9.452–13.027 | +0.0065 ± 0.0099 |
| cnn14 | 7.500 | 0.006 | 0.012 | 2.145 [1.299, 5.114] (başarısız 0) | 0.010 | 0.002 | 0.015 | 2.268 (0.003) | 0.004 (hayır) | 3.510–9.089 | +0.0033 ± 0.0081 |
| cnn14_16k | 2.127 | 0.145 | 0.145 | 1.551 [1.008, 2.877] (başarısız 0) | 0.144 | 0.054 | 0.192 | 1.638 (0.078) | 0.122 (hayır) | 0.022–4.493 | +0.0006 ± 0.0043 |
| beats | 17.576 | <0.001 | <0.001 | 4.691 [2.664, 18.760] (başarısız 0) | <0.001 | <0.001 | 0.020 | 4.598 (<0.001) | <0.001 (hayır) | 13.899–20.847 | +0.0092 ± 0.0105 |
| wavlm_base_plus | 9.487 | 0.002 | 0.006 | 2.747 [1.446, 9.394] (başarısız 0) | 0.004 | 0.006 | 0.068 | 2.825 (0.001) | 0.001 (hayır) | 5.563–11.897 | +0.0056 ± 0.0087 |
| wavlm_large | 15.829 | <0.001 | <0.001 | 3.834 [1.971, 14.730] (başarısız 0) | <0.001 | <0.001 | 0.017 | 3.965 (<0.001) | <0.001 (hayır) | 10.567–18.982 | +0.0090 ± 0.0108 |
| mfcc_lr | 6.819 | 0.009 | referans | 2.105 [1.316, 5.300] (başarısız 0) | 0.011 | 0.006 | 0.103 | 2.040 (0.010) | 0.005 (hayır) | 4.456–8.802 | +0.0029 ± 0.0070 |
| mfcc_mlp | 5.269 | 0.022 | referans | 2.030 [1.232, 6.525] (başarısız 0) | 0.026 | 0.008 | 0.152 | 2.060 (0.015) | 0.012 (hayır) | 3.330–6.458 | +0.0020 ± 0.0073 |

## 2. E2h-KLR (L4) ve tabakalı AUC'ler (L1, betimsel)

| kol | E2h n (astım/sağlıklı) | OR / SD [%95 CI] | p |
|---|---|---|---|
| cnn10 | 65/50 | 4.914 [1.924, 12.554] | <0.001 |
| cnn14 | 65/50 | 2.729 [1.342, 5.550] | 0.006 |
| cnn14_16k | 65/50 | 2.157 [1.051, 4.430] | 0.036 |
| beats | 65/50 | 7.166 [2.523, 20.354] | <0.001 |
| wavlm_base_plus | 65/50 | 3.656 [1.569, 8.521] | 0.003 |
| wavlm_large | 65/50 | 5.049 [2.015, 12.654] | <0.001 |
| mfcc_lr | 65/50 | 2.687 [1.378, 5.240] | 0.004 |
| mfcc_mlp | 65/50 | 2.295 [1.144, 4.603] | 0.019 |

Tabakalı AUC (yalnız aynı tabakadaki astım–sağlıklı çiftleri). **0.5'e karşı test edilmez**; her tasarımın gerçek şans çizgisi aşağıdaki yalnız-bağlam referanslarıdır (CONFOUND 2.3).

| kol | E1 | E1h | E2 | E2h | E3 | E3h |
|---|---|---|---|---|---|---|
| cnn10 | 0.855 | 0.863 | 0.828 | 0.815 | 0.842 | 0.929 |
| cnn14 | 0.805 | 0.774 | 0.792 | 0.749 | 0.836 | 0.929 |
| cnn14_16k | 0.728 | 0.719 | 0.733 | 0.722 | 0.649 | 0.929 |
| beats | 0.922 | 0.928 | 0.900 | 0.901 | 0.877 | 0.857 |
| wavlm_base_plus | 0.886 | 0.886 | 0.859 | 0.843 | 0.895 | 0.929 |
| wavlm_large | 0.911 | 0.911 | 0.890 | 0.873 | 0.883 | 0.857 |
| mfcc_lr | 0.774 | 0.804 | 0.740 | 0.775 | 0.737 | 0.786 |
| mfcc_mlp | 0.790 | 0.805 | 0.738 | 0.749 | 0.725 | 0.786 |
| *referans: yalnız yaş* | 0.674 | 0.703 | 0.664 | 0.703 | 0.693 | 0.918 |
| *referans: yalnız saat + tarih* | 0.928 | 0.817 | 0.809 | 0.352 | 0.883 | 0.646 |
| *referans: yalnız yaş + saat + tarih* | 0.930 | 0.854 | 0.827 | 0.497 | 0.918 | 0.861 |

## 3. Bağlam kullanımı: Spisak (L3), karşıt hücreler (L5), N2

Spisak: mlconfound 0.21.3, cond_dist_method='linear' (gam güncel scipy ile çalışmıyor; EVAL016_PLAN S3). Tam test T1'in destekleyicisidir. mlconfound KISMİ testi yalnız bilgi amaçlı: sentetik testte bağlamsız güçlü bir skoru yanlışlıkla işaretledi; C2 kararı etiket-içi permütasyon testine dayanır (S10).

| kol | **L3 kısmi (C2): etiket-içi permütasyon p**, C = saat + tarih [+ yaş] | Spisak tam p (H0: ŝ ⊥ Y \| C) | mlconfound kısmi p (yalnız bilgi) | hasta-içi PM katsayısı (z) [%95 CI] | sağlıklı AM − PM (z) [%95 CI] | N2 tarih p (hastalar) |
|---|---|---|---|---|---|---|
| cnn10 | 0.212 [<0.001] | <0.001 | 0.033 | +0.104 [-0.149, +0.356] | +0.204 [-0.151, +0.549] (12/45) | 0.128 |
| cnn14 | 0.129 [<0.001] | <0.001 | 0.155 | +0.119 [-0.128, +0.367] | +0.467 [-0.086, +1.062] (12/45) | 0.450 |
| cnn14_16k | 0.082 [<0.001] | 0.040 | 0.133 | +0.046 [-0.223, +0.316] | +0.369 [-0.026, +0.803] (12/45) | 0.192 |
| beats | 0.092 [<0.001] | <0.001 | 0.003 | -0.015 [-0.238, +0.208] | +0.402 [+0.070, +0.746] (12/45) | 0.002 |
| wavlm_base_plus | 0.188 [<0.001] | <0.001 | <0.001 | -0.072 [-0.287, +0.143] | +0.411 [-0.077, +0.925] (12/45) | 0.016 |
| wavlm_large | 0.137 [<0.001] | <0.001 | 0.001 | -0.005 [-0.221, +0.211] | +0.251 [-0.146, +0.667] (12/45) | 0.003 |
| mfcc_lr | 0.050 [<0.001] | <0.001 | 0.027 | -0.096 [-0.325, +0.134] | -0.125 [-0.701, +0.431] (12/45) | 0.029 |
| mfcc_mlp | 0.328 [0.022] | 0.002 | 0.177 | -0.085 [-0.320, +0.150] | -0.080 [-0.542, +0.343] (12/45) | 0.004 |

## 4. EXP-020 G1 bağlam probu (gömmeler) ve N5

| backbone | (a) hastalarda PM: OOF AUC | LR p (tarih ayarlı) | Holm p | OR / SD | G1 | (b) geç dönem AUC [%95 CI] (klinik ref. 0.855) | N5 p (vekil skor T1'de) |
|---|---|---|---|---|---|---|---|
| cnn10 | 0.583 | 0.044 | 0.266 | 1.319 | − | 0.643 [0.575, 0.714] | bilgisiz (G1−) |
| cnn14 | 0.550 | 0.626 | 1.000 | 0.947 | − | 0.621 [0.553, 0.686] | bilgisiz (G1−) |
| cnn14_16k | 0.515 | 0.483 | 1.000 | 0.913 | − | 0.644 [0.576, 0.712] | bilgisiz (G1−) |
| beats | 0.561 | 0.176 | 0.881 | 1.203 | − | 0.725 [0.662, 0.788] | bilgisiz (G1−) |
| wavlm_base_plus | 0.541 | 0.307 | 1.000 | 1.147 | − | 0.585 [0.516, 0.657] | bilgisiz (G1−) |
| wavlm_large | 0.532 | 0.546 | 1.000 | 1.084 | − | 0.703 [0.641, 0.766] | bilgisiz (G1−) |

## 5. Codec zinciri (L7, betimsel)

| kol | karışık zincirli katılımcı | ffmpeg − apple logit [%95 CI] | zincir probu AUC [%95 CI] |
|---|---|---|---|
| cnn10 | 284 | -0.240 [-0.371, -0.118] | 0.584 [0.557, 0.610] |
| cnn14 | 284 | -0.166 [-0.280, -0.047] | 0.563 [0.539, 0.589] |
| cnn14_16k | 284 | -0.050 [-0.148, +0.068] | 0.578 [0.554, 0.602] |
| beats | 284 | -0.101 [-0.290, +0.107] | 0.710 [0.686, 0.736] |
| wavlm_base_plus | 284 | -0.053 [-0.204, +0.098] | 0.594 [0.570, 0.619] |
| wavlm_large | 284 | -0.149 [-0.290, -0.011] | 0.603 [0.576, 0.631] |
| mfcc_lr | 284 | +0.049 [-0.040, +0.134] | — [—, —] |
| mfcc_mlp | 284 | +0.269 [-0.077, +0.612] | — [—, —] |

## 6. Oynaklık (L8)

| kol | fold AUC ort ± SD (25) | tekrar AUC aralığı |
|---|---|---|
| cnn10 | 0.855 ± 0.047 | 0.842–0.865 |
| cnn14 | 0.791 ± 0.063 | 0.768–0.798 |
| cnn14_16k | 0.729 ± 0.067 | 0.685–0.747 |
| beats | 0.921 ± 0.046 | 0.909–0.921 |
| wavlm_base_plus | 0.884 ± 0.039 | 0.865–0.890 |
| wavlm_large | 0.909 ± 0.041 | 0.888–0.912 |
| mfcc_lr | 0.771 ± 0.078 | 0.759–0.773 |
| mfcc_mlp | 0.770 ± 0.058 | 0.759–0.783 |

## 7. Önceden yazılmış kurallara göre Aşama 1 kararları

C1 koşulları: 1 Holm-T1 (PLR) p < 0.05 ve OR > 1 · 2 bootstrap CI 1'i dışlıyor · 3 gün-kümeli: jackknife (CV3) p < 0.05 ve OR > 1 (K2, D-038) · 4 E2h-KLR OR > 1 · 5 df 4 ve df 6 aynı yön · 6 N3 (Aşama 2) · 7 G1+ ise N5 anlamlı değil (S11: yalnız betimsel, C1'e dahil değil).

| backbone | 1 | 2 | 3 | 4 | 5 | 7 | Aşama-1 koşulları | C1 | Aşama 2 (kabul edilen kural: Holm-T1 pozitif) | C2 (bağlam kullanımı) |
|---|---|---|---|---|---|---|---|---|---|---|
| cnn10 | ✓ | ✓ | ✓ | ✓ | ✓ | n/a | hepsi ✓ | Aşama 2 bekliyor (N3) | evet | gösterilemedi |
| cnn14 | ✓ | ✓ | ✓ | ✓ | ✓ | n/a | hepsi ✓ | Aşama 2 bekliyor (N3) | evet | gösterilemedi |
| cnn14_16k | ✗ | ✓ | ✗ | ✓ | ✓ | n/a | eksik | sağlanmadı | hayır | gösterilemedi |
| beats | ✓ | ✓ | ✓ | ✓ | ✓ | n/a | hepsi ✓ | Aşama 2 bekliyor (N3) | evet | gösterilemedi |
| wavlm_base_plus | ✓ | ✓ | ✗ | ✓ | ✓ | n/a | eksik | sağlanmadı | evet | gösterilemedi |
| wavlm_large | ✓ | ✓ | ✓ | ✓ | ✓ | n/a | hepsi ✓ | Aşama 2 bekliyor (N3) | evet | gösterilemedi |

## 8. Bu rapor neyi gösterebilir, neyi gösteremez (önceden yazılmış metin)

- Aşama-1 koşullarını sağlayan backbone(lar) için C1 **henüz kurulmadı**: N3 (Aşama 2) bekleniyor.
- T1 yalnız **ölçülmüş** bağlamı (saat, tarih) ve yaşı ayarlar. Ölçülmemiş, güne özgü bağlamda T1-PLR'nin yanlış pozitif oranı ~%11 (p < 0.05) olabilir (SIM-001 S3; Bölüm 0). Sandviç CR1 bunu düzeltmez; C1 koşul 3 (CV3) simülasyonda bu senaryoyu sınırlıyor, ama yalnız **sentetik** gün etkisi için gösterildi.
- İstatistiksel anlamlılık (koşul 1) tek başına "bağlamdan bağımsız ses sinyali" kanıtı değildir; o iddia yalnız C1'in bütün koşulları ve Aşama 2 (N3) ile kurulabilir. Klinik yorum bu raporun kapsamı dışındadır.
- Pozitiflik sınırı: sabah kaydedilen sağlıklı sayısı çok küçük, geç dönemde sağlıklı yok; bilgi yalnız örtüşme bölgesinden gelir.
- Spirometri / bronkodilatör sonrası kayıt olasılığı (S13) saat kontrolüyle ayrılamaz; veri ekibinden cevap bekleniyor.
- Pozitif bir C1 bile klinik tanı aracı anlamına gelmez (vaka-kontrol, tek merkez, dış doğrulama yok).
- Bu sayılar EXP-016'nın dondurulmuş probları içindir; fine-tune modellerine genellenemez (C3 ayrı değerlendirilir).

