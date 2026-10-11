# EVAL-016 — POST HOC duyarlılıklar (PH1–PH3; otomatik üretildi; katılımcı ID'si içermez)

**Aşama 1 sonuçları görüldükten sonra önerildi; C1 / C2 kararlarını değiştirmez.** Tanımlar ve yorum kuralı çalıştırmadan önce yazıldı (`docs/EVAL016_STAGE2_PLAN.md` Bölüm 5).

**PH0 bütünlük:** kaydedilmiş ŝ ile birincil T1 Aşama 1 raporunu yeniden üretti (en büyük |Δ LR| = 8.53e-14). Kohort 339 (282 / 57).

## PH1 / PH2 — T1 farklı bağlam modelleriyle

| kol | birincil: Holm p · OR · CV3 p | PH1 yaş spline: Holm p · OR · CV3 p | PH2 + cinsiyet + sigara: Holm p · OR · CV3 p | işaret |
|---|---|---|---|---|
| cnn10 | 0.002 · 3.064 · 0.003 | 0.002 · 3.190 · 0.018 | 0.005 · 2.871 · 0.003 | PH1–PH2'ye dayanıklı |
| cnn14 | 0.012 · 2.145 · 0.015 | 0.010 · 2.233 · 0.067 | 0.036 · 2.004 · 0.025 | **yaş / demografi ayarına duyarlı** |
| cnn14_16k | 0.145 · 1.551 · 0.192 | 0.128 · 1.594 · 0.274 | 0.289 · 1.395 · 0.263 | Aşama 1'de sağlanmadı |
| beats | <0.001 · 4.691 · 0.020 | <0.001 · 4.937 · 0.045 | <0.001 · 4.924 · 0.027 | PH1–PH2'ye dayanıklı |
| wavlm_base_plus | 0.006 · 2.747 · 0.068 | 0.005 · 2.891 · 0.112 | 0.014 · 2.582 · 0.062 | Aşama 1'de sağlanmadı |
| wavlm_large | <0.001 · 3.834 · 0.017 | <0.001 · 3.859 · 0.028 | 0.001 · 3.634 · 0.038 | PH1–PH2'ye dayanıklı |
| mfcc_lr | ref. 0.009 · 2.105 · 0.103 | ref. 0.009 · 2.096 · 0.171 | ref. 0.006 · 2.229 · 0.093 | — |
| mfcc_mlp | ref. 0.022 · 2.030 · 0.152 | ref. 0.024 · 2.018 · 0.198 | ref. 0.015 · 2.103 · 0.124 | — |

İşaret (önceden yazılmış kural): Aşama-1 koşullarını sağlayan bir backbone PH1 ya da PH2'de Holm p ≥ 0.05 ya da CV3 p ≥ 0.05 çıkarsa **yaş / demografi ayarına duyarlı**. MFCC kolları referans (Holm ailesinde değil).

## PH3 — K19: E2h cezalı koşullu logitin sayısal yeniden doğrulaması

| kol | Aşama 1: OR (yakınsadı) | PH3: OR [%95] | p | yakınsadı (yerel min., uyuşan başlangıç) | başlangıçlar arası en büyük fark | ML OR |
|---|---|---|---|---|---|---|
| cnn10 | 4.914 (hayır) | 4.914 [1.924, 12.554] | <0.001 | evet (evet, 3/3) | 3.6e-05 | 5.897 |
| cnn14 | 2.729 (evet) | 2.729 [1.342, 5.550] | 0.006 | evet (evet, 3/3) | 3.5e-05 | 2.971 |
| cnn14_16k | 2.157 (evet) | 2.157 [1.051, 4.430] | 0.036 | evet (evet, 3/3) | 1.2e-04 | 2.276 |
| beats | 7.166 (hayır) | 7.166 [2.523, 20.354] | <0.001 | evet (evet, 3/3) | 5.5e-05 | 8.903 |
| wavlm_base_plus | 3.656 (evet) | 3.656 [1.569, 8.521] | 0.003 | evet (evet, 3/3) | 1.7e-05 | 4.243 |
| wavlm_large | 5.049 (hayır) | 5.049 [2.015, 12.654] | <0.001 | evet (evet, 3/3) | 8.6e-05 | 6.096 |
| mfcc_lr | 2.687 (evet) | 2.687 [1.378, 5.240] | 0.004 | evet (evet, 3/3) | 9.1e-05 | 2.917 |
| mfcc_mlp | 2.295 (evet) | 2.295 [1.144, 4.603] | 0.019 | evet (evet, 3/3) | 7.4e-05 | 2.459 |

K19 sonucu: yön her kolda aynı; bütün yeni fit'ler yakınsadı

Süre 931.4 s.
