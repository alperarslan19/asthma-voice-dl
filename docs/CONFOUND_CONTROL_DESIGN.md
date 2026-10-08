# Kayıt bağlamı confounder'ı: yöntemlerin değerlendirmesi ve deney tasarımı

**Durum:** ÖNERİLDİ — onay bekliyor (D-025 test spesifikasyonu, D-026 kontrol yığını, D-027 fine-tuning protokolü) · **Tarih:** 2026-10-08 (6. tur) · **Ses modeli sonuçları görülmeden yazıldı**

> **7. tur — D-028 (KABUL):** Confounder ayrı bir araştırma başlığı. Bu belgedeki kontrol deneyleri (EXP-020, EXP-022), azaltma yöntemleri ve D-025–027 önerileri **model geliştirme sırasında uygulanmaz**. Model geliştirme bittiğinde, kaydedilmiş tahminler üzerinde yapılacak değerlendirmenin yol haritası olarak kullanılır. Geliştirme sırasında yalnız D-028'deki kayıt yükümlülükleri ve "üst sınır" raporlama kuralı geçerlidir.
**İlgili:** D-017, D-020, D-021, D-022, D-023 · EXP-003, EXP-004, EXP-004b, SIM-001 · rapor Bölüm 6.9–6.10

> Bu belge şu soruya cevap verir: *Yeni veri toplamadan, ham ses + pretrained model + fine-tuning ile "model gerçekten astım bilgisi mi öğreniyor, yoksa kayıt bağlamını mı?" sorusunu nasıl savunulabilir biçimde test ederiz?*

---

## 0. Kısa cevap

| # | Yöntem | Confounding'i gerçekten azaltır mı? | Hüküm | Bu projedeki rolü |
|---|---|---|---|---|
| 1 | Ses ön işleme / harmonizasyon | **Yalnız teknik kanalları** (codec zinciri, bant genişliği, seviye). Saat / tarih / yaş yolunu **azaltmaz**. | Hijyen; saat-tarih confounding'i açısından **kozmetik**. Gürültü giderme ve sabit gürültü ekleme **zararlı**. | Her zaman (D-021), etkisi zincir probuyla ölçülür. |
| 2 | Zaman / tarih / yaş eşleştirme, E2h | **Evet, ölçülen bağlam için ve yalnız tasarım içinde.** Ama E2h'de yaş dengesiz (0.703) ve saat/tarih kısmen dengesiz. | Gerçek kontrol, ama **ham E2h AUC'si yorumlanamaz**; ayarlı analiz şart. | Destekleyici test: E2h içinde koşullu lojistik regresyon. |
| 3 | Bağlam-only vs ses-only vs bağlam+ses (artımlı) | Modelin kullandığı confounding'i azaltmaz; **tahmin edilen sayıyı** confounding'den arındırır. Doğru soruyu sorar. | **Birincil çıkarım aracı** (T1). Karar istatistiği ΔAUC değil, olabilirlik temelli test (SIM-001). | Birincil test. |
| 4 | Residualization / confound regression | Yalnız **CV içinde** ve yalnız **doğrusal** kısmı. Tüm veride yapılırsa yanlılık üretir. Ham dalga formuna uygulanamaz. | Dondurulmuş gömmelerde **duyarlılık analizi**. Fine-tuning'de yok. | EXP-022 (koşullu), yalnız hastalarda tahmin edilen bağlam yönleriyle. |
| 5 | Adversarial debiasing / gradient reversal | Marjinal hâli **astım sinyalini de siler** (bağlam etiketi 0.93 ile ele veriyor). Etiket-koşullu hâli teoride doğru, ama küçük n'de kararsız ve kaldırma eksik kalır. | Birincil hatta **yok**. Yalnız araştırma uzantısı. | Planlanmadı; koşullar Bölüm 3.5'te. |
| 6 | Katılımcı düzeyinde split, grup temelli doğrulama | Katılımcı split'i **confounding kontrolü değildir** (kimlik sızıntısını önler; bağlam katılımcı düzeyinde olduğu için confounding train ve testte aynen kalır). **Gün-gruplu** CV, güne özgü akustik izi kısmen kontrol eder. | Katılımcı split: zorunlu, ama tek başına değersiz. Gün-gruplu CV: duyarlılık. | Her deney (zorunlu) + EXP-021'de gün-gruplu duyarlılık. |
| 7 | Bilinen kayıt değişkenleri (codec / cihaz / oda) | Cihaz, mesafe ve konum sabit. Değişen tek bilinen değişken codec zinciri, ama etiketle ilişkili değil (p = 0.22) ve hastalarda saatle de ilişkili değil (EXP-004). | Doğru ve gerekli, ama **ana confounder'a dokunmuyor**. | Harmonizasyon + zincir probu + katılımcı içi zincir duyarlılığı. |

**Birlikte kullanılması gerekenler:** 1 + 6 (hijyen) → 3 (birincil test) + 2 (ayarlı, destekleyici) → negatif kontroller → (gerekirse) 4 veya yeniden ağırlıklandırma (azaltma). Azaltma yöntemleri **kanıt üretmez**; işe yarayıp yaramadıkları yine 3 ve 2 ile ölçülür.

**Hiçbir yöntemin aşamayacağı sınır (pozitiflik):** Sabah kaydedilen 12 sağlıklı ve öğleden sonra kaydedilen 86 hasta dışında, bağlam etiketi neredeyse belirliyor. Nisan 2024 sonrasındaki 184 hastanın ise hiç tarih-eşi sağlıklısı yok. Bu bölgelerde sesin astım bilgisini bağlamdan ayırmak **mümkün değildir**: veri buna izin vermiyor. Bilgi yalnız örtüşme bölgesinden gelir. SIM-001 bunu sayıyla gösteriyor: tüm kohortu (339 kişi) kullanan T1'in gücü, yalnız E2h'yi (115 kişi) kullanan testin gücüyle aşağı yukarı aynı. Kestirme basamak biçimindeyse E2h testi daha da güçlü. [FACT + INFERENCE]

---

## 1. Sorunun yapısı: hangi okun üzerinde çalışıyoruz?

```
                 ┌──────────── fizyoloji (istediğimiz yol) ─────────────┐
                 │                                                        ▼
   Astım (Y) ────┤                                                     Ses (A) ──► model skoru (ŝ)
                 │   işe alım / klinik akışı                              ▲  ▲
                 └──────────────► Kayıt bağlamı (C: tarih, saat) ─────────┘  │
                                   (oda gürültüsü, sabah sesi,               │
                                    iOS sürümü, klinik yoğunluğu …)          │
   Yaş ─────────► Y ;  Yaş ─────────────────────────────────────────────────┘
   Y ──► prosedür (spirometri / bronkodilatör, S13) ──► A        ← ayrı bir sorun; saat kontrolü bunu çözmez
```

Her yöntem şu üç yerden birine müdahale eder:

| Müdahale yeri | Örnek yöntemler | Ne kazandırır | Ne kazandırmaz |
|---|---|---|---|
| **(a) Veri:** C → A okunu kesmek | harmonizasyon, residualization (girdi), gürültü işlemleri | Modelin bağlamı "görmesini" zorlaştırır | Kanıt. İşe yarayıp yaramadığı ancak (c) ile ölçülür. |
| **(b) Öğrenme:** modelin C'yi kullanmasını cezalandırmak | yeniden ağırlıklandırma, adversarial, etiket-koşullu kayıplar | Bağlama dayanıklı model (genelleme) | Kanıt. Aynı nedenle (c) gerekir. |
| **(c) Değerlendirme:** soruyu C'ye koşullu sormak | artımlı test (T1), ayarlı E2h, Spisak testleri, negatif kontroller | **Kanıt**: "ses, bağlamın ötesinde bilgi taşıyor mu?" | Modeli düzeltmez; yalnız dürüstçe ölçer. |

Bu yüzden tasarımın omurgası (c)'dir. (a) ve (b) ancak (c) bir sorun gösterirse ve sorunun mekanizması belliyse eklenir. [DECISION]

**Hangi değişken confounder, hangisi aracı (mediator)?** [INFERENCE]
- *Saat:* Hastaların sabah gelmesi klinik iş akışının sonucudur. "Sabah sesi" ise iki grubu da etkiler. Yani saat bir **confounder**'dır; onu kontrol etmek doğrudur, aşırı-ayarlama (over-adjustment) değildir.
- *Saat kontrolünün bedeli:* Astımın gerçek ses etkisinin bağlamla doğrusal olarak örtüşen kısmı bağlama atfedilir. Bu, sonucu **muhafazakâr** yapar ama yanlı değil; sorun tanımlanabilirliktir.
- *Bronkodilatör / spirometri sonrası kayıt (S13):* Bu, saatten bağımsız bir **prosedür yolu**dur. Hiçbir saat kontrolü onu ayırmaz. Veri ekibinin cevabı gelene kadar her pozitif sonucun yanında açık bir sınırlama olarak yazılır.
- *Yaş:* Klasik confounder. Ses modelleri yaşı çok iyi kodlar. Her testte ayarlanır.

---

## 2. Bu turda eklenen veri kontrolleri

Hepsi `scripts/analyze_recording_context.py` (E, E-b, F) ve `scripts/simulate_t1_power.py` ile tekrar üretilebilir; çıktılar `reports/recording_context/` altında, katılımcı ID'si içermez.

### 2.1 EXP-004 — Saatin kaba akustik izi (etiket içinde)
Hastalarda sabah ve öğleden sonra kayıtları, denetim ölçümleriyle (gürültü tabanı, seviye, SNR, bant genişliği, süre, sessizlikler, bitrate, zincir) ayrılamıyor:

| Grup | n (öğleden sonra) | Tüm ölçümler, CV AUC | Şanstan ayrılan tekil ölçüm |
|---|---|---|---|
| Astım | 282 (86) | 0.549 ± 0.067 | yok |
| Sağlıklı | 58 (46) | 0.525 ± 0.179 | yok (sabah yalnız 12 kişi → çok düşük güç) |

**Yorum:** Odada kaba, ölçülebilir bir "sabah / öğleden sonra" farkı (gürültü tabanı, seviye) görünmüyor. [FACT] Bu, harmonizasyonun saat confounding'ine karşı **neden kozmetik kalacağını** da açıklıyor: kaldırılacak kaba bir teknik iz yok. Ama bu, derin gömmelerin saati kodlamadığı anlamına **gelmez**: ince spektral veya prozodik izler bu ölçümlerde görünmez. Bu soru EXP-020'de (G1) gömmelerle test edilecek. [INFERENCE]

### 2.2 EXP-004b — Sabah ve öğleden sonra hastaları klinik olarak farklı mı?
Klinik profil (tedavi basamağı, SFT, GINA, ACT, FEV1%, FVC%, FEV1/FVC, ICS, rinit, sigara, cinsiyet, yaş) → öğleden sonra: **CV AUC 0.528 ± 0.071** (fold %2.5–97.5: 0.37–0.64). Düzeltmesiz p < 0.05 olan tek değişken dönem (öğleden sonra hastaları daha çok geç dönemde; p = 0.023). [FACT]

**Neden önemli:** "Etiket içinde skor ↔ saat" ilişkisi (Spisak kısmi testi, N5 bağlam-vekili kontrolü), sabah ve öğleden sonra hastaları klinik olarak benzerse "kayıt bağlamı" diye okunabilir. Benzerler. Tek fark dönem; bu yüzden bu kontrastlarda tarih her zaman kovaryat olacak. [INFERENCE]

### 2.3 Her tasarımın gerçek "şans çizgisi" (rapor Bölüm F)
Bağlamı mükemmel kodlayan ama hiç astım bilgisi taşımayan bir modelin her tasarımda alacağı tabakalı AUC. Bağlam-only lojistik modelin OOF skoru kullanıldı (5-fold × 20; ± tekrarlar arası SD):

| Tasarım | yalnız yaş | saat + tarih | yaş + saat + tarih |
|---|---|---|---|
| E1 tam kohort | 0.674 | 0.930 | 0.930 |
| E1h + aynı 1 saat dilimi | 0.703 | 0.817 | 0.854 |
| E2 zaman penceresi | 0.665 | 0.810 | 0.827 |
| **E2h pencere + 1 saat dilimi** | **0.703** | **0.351** | 0.497 |
| E3 aynı gün | 0.691 | 0.885 | 0.918 |
| E3h aynı gün + 1 saat dilimi | 0.914 | 0.639 | 0.861 |

**Bu tablo D-023'ü düzeltiyor:** [FACT + INFERENCE]
- E2h'de **yalnız yaşı kodlayan bir ses modeli 0.70 alır.** Ses modelleri yaşı iyi kodlar → "E2h > 0.5" kanıt değildir.
- Tam kohortta bağlamı öğrenmiş bir model E2h'de **0.35** alır, yani ters yönde. Tam kohortta "geç tarih → astım" doğruyken, pencere içinde hastalar daha erken tarihlidir. Kestirme öğrenmiş bir modelin E2h sonucu, hangi kestirmeyi öğrendiğine göre 0.35 ile 0.70 arasında herhangi bir yere düşebilir.
- **Sonuç:** Ham E2h AUC'si tek başına yorumlanamaz. E2h, tabakalar içinde yaş, saat ve tarihe göre ayarlanmış **koşullu lojistik regresyonla** değerlendirilecek (D-025; D-023 güncellemesi). Ham E2h AUC'si yine raporlanır, ama yanında bu üç referans çizgisiyle.

### 2.4 SIM-001 — T1'in gücü ve yanlış pozitif riski (ses olmadan)
Gerçek etiketler ve gerçek bağlam (saat, tarih, yaş, kayıt günü) kullanıldı; yalnız ses skoru yapay:
`skor = γ·astım + 1·kestirme(bağlam) + gürültü`.
- γ = bağlamdan bağımsız gerçek sinyal. Aynı bağlamdaki ayrım gücü AUC = Φ(γ/√2).
- Kestirme gücü bilinçli olarak yüksek seçildi (λ = 1).
- Her hücrede 500 simülasyon. Ret ölçütü p < 0.05.

Hücreler: p < 0.05 ile ret oranı; parantez içinde p < 0.01 (≈ 6 backbone için Holm'un en sıkı eşiği, 0.0083). **γ = 0 satırları yanlış pozitif oranıdır** (ideal: 0.05 / 0.01); γ > 0 satırları güçtür.

| senaryo | gamma | aynı bağlamda AUC | T1 kuadratik | T1 spline | T1 spline + gün kümesi | E2h KLR | ort. ΔAUC (spline) |
|---|---|---|---|---|---|---|---|
| S1_smooth | 0.0 | 0.5 | 0.068 (0.012) | 0.046 (0.01) | 0.078 (0.018) | 0.062 (0.006) | 0.0005 |
| S1_smooth | 0.25 | 0.57 | 0.23 (0.082) | 0.15 (0.03) | 0.182 (0.08) | 0.124 (0.03) | 0.0008 |
| S1_smooth | 0.5 | 0.638 | 0.708 (0.474) | 0.512 (0.26) | 0.544 (0.334) | 0.482 (0.224) | 0.0023 |
| S1_smooth | 0.75 | 0.702 | 0.97 (0.876) | 0.84 (0.576) | 0.852 (0.606) | 0.824 (0.546) | 0.0041 |
| S1_smooth | 1.0 | 0.76 | 1.0 (0.994) | 0.972 (0.89) | 0.968 (0.878) | 0.966 (0.864) | 0.0069 |
| S2_step | 0.0 | 0.5 | 0.216 (0.066) | 0.06 (0.012) | 0.09 (0.022) | 0.06 (0.014) | 0.0005 |
| S2_step | 0.25 | 0.57 | 0.628 (0.356) | 0.08 (0.022) | 0.106 (0.032) | 0.16 (0.03) | 0.0006 |
| S2_step | 0.5 | 0.638 | 0.952 (0.806) | 0.34 (0.128) | 0.364 (0.156) | 0.482 (0.182) | 0.0016 |
| S2_step | 0.75 | 0.702 | 1.0 (0.984) | 0.764 (0.426) | 0.712 (0.452) | 0.8 (0.538) | 0.0033 |
| S2_step | 1.0 | 0.76 | 1.0 (1.0) | 0.924 (0.776) | 0.916 (0.722) | 0.962 (0.818) | 0.0055 |
| S3_day | 0.0 | 0.5 | 0.212 (0.076) | 0.122 (0.034) | 0.11 (0.044) | 0.142 (0.03) | 0.0007 |
| S3_day | 0.25 | 0.57 | 0.296 (0.158) | 0.204 (0.088) | 0.182 (0.086) | 0.212 (0.082) | 0.001 |
| S3_day | 0.5 | 0.638 | 0.464 (0.27) | 0.32 (0.164) | 0.308 (0.158) | 0.312 (0.138) | 0.0016 |
| S3_day | 0.75 | 0.702 | 0.688 (0.5) | 0.558 (0.32) | 0.492 (0.29) | 0.51 (0.3) | 0.0028 |
| S3_day | 1.0 | 0.76 | 0.866 (0.752) | 0.732 (0.566) | 0.69 (0.49) | 0.704 (0.492) | 0.0042 |

**Ne öğrendik?** [FACT, simülasyon; INFERENCE, gerçek modele taşınması]
1. **Kuadratik bağlam modeli güvenli değil.**
   - Ses yalnız "sabah mı / geç dönem mi" basamaklarını kodluyorsa (S2), EXP-003 biçimindeki kuadratik saat + doğrusal tarih modeliyle T1 yanlış pozitif oranı **0.216** (p < 0.01'de bile 0.066).
   - Spline bağlam modeli (4 df) ve E2h-KLR kalibre: 0.060.
   - → T1'in bağlam kısmı spline (D-025).
2. **Ölçülmemiş, güne özgü bağlam çıkarımla düzeltilemiyor.**
   - Güçlü bir gün etkisi varsa (S3) yanlış pozitif spline'da 0.122, gün-kümeli SE ile 0.110, E2h-KLR'de 0.142.
   - Sebep: 65 günün 53'ü tek etiketli, yani gün etkisiyle etiket pratikte ayrılamıyor.
   - Önlemler:
     - modelin gün imzasını sistematik olarak öğrenmesini önlemek için gün-gruplu CV;
     - N3 ve N5 kontrolleri;
     - Holm eşiği: p < 0.01'de S3 yanlış pozitifi 0.034'e iniyor.
   - Bu kalıntı risk her pozitif sonuçta açıkça yazılacak.
3. **ΔAUC karar istatistiği olamaz.** Güç %84–97 iken bile ortalama ΔAUC yalnız 0.004–0.007. D-023'ün "ΔAUC CI'ı 0'ı dışlasın" ölçütü pratikte güçsüzdü. Karar olabilirlik temelli testle verilir; ΔAUC betimsel kalır.
4. **Bilgi örtüşme bölgesinde.** Tam kohort T1 (339 kişi) ile E2h-KLR (115 kişi) neredeyse aynı güce sahip. Basamak kestirmede E2h-KLR daha güçlü (γ = 0.5'te 0.48 vs 0.34).
5. **Minimum saptanabilir etki (%80 güç):**

   | Eşik | Aynı-bağlam AUC'si (ölçülen bağlam sabitken ses skorunun astımı ayırma gücü) |
   |---|---|
   | α = 0.05 | ≈ 0.69 |
   | Holm düzeyi (≈ 0.01) | ≈ 0.74 |

   Aynı-bağlam AUC'si 0.64 olan bir sinyal yaklaşık yarı yarıya (α = 0.05) ya da dörtte bir (α = 0.01) yakalanır.
   
   Bu sayılar **iyimser**: gerçek OOF skoru tek boyutlu, normal gürültülü bir skordan daha gürültülü olacak. Olumsuz bir sonuç bu güç sınırıyla birlikte raporlanır.

---

## 3. Yedi yöntemin ayrıntılı değerlendirmesi

### 3.1 Ses ön işleme / harmonizasyon
- **Ne yapar:** Kayıpsız çözme, ortak SR, 11.0 kHz alçak geçiren (32 kHz yolu), kenar kırpma, tepe normalizasyonu (D-021). Codec zincirinin ve bant genişliğinin izini siler.
- **Neden saat / tarih confounding'ini azaltmaz:** Saat ve tarihin sesteki olası izleri konuşma bandındadır: oda tonu, klima, klinik gürültüsü, sabah sesi, mevsim (soğuk algınlığı, polen). Bu izleri sesi bozmadan ayırmak mümkün değil. EXP-004'e göre de kaldırılacak kaba bir teknik iz yok. Codec zinciri ise etiketle (p = 0.22) ve hastalarda saatle ilişkili değil. [FACT + INFERENCE]
- **Zararlı varyantlar:**
  - *Gürültü giderme* nefesli ses, hırıltı ve türbülans gibi olası hastalık işaretlerini silebilir ve girişe bağlı yeni artefaktlar üretir. Ayrıca gürültü tahmini bağlama göre değiştiği için confound'u kaldırmaz, biçimini değiştirir.
  - *Sabit gürültü ekleme* herkesin SNR'ını düşürür ama bağlam farkını yalnız kısmen maskeler.
  - *DAW'dan yeniden export* kayıplı yeniden kodlama ekler. [INFERENCE, D-021]
- **Tepe normalizasyonu hakkında not:** Mutlak ses şiddeti bilgisini siler. Şiddet hem hastalık (zayıf ses) hem bağlam (operatör talimatı) taşıyabilir. Pretrained modeller için standart bir adım; MFCC baseline'ında şiddet özellikleri ayrıca raporlanır. [DECISION]
- **Hüküm:** Yapılır, çünkü ucuz, deterministik ve teknik bir istikrarsızlık kaynağını kaldırır. Ama **"confounder'ı kontrol ettik" iddiası için kullanılmaz.** Etkisi zincir probuyla ölçülür (EXP-020).

### 3.2 Zaman / tarih / yaş eşleştirme ve E2h
- **Ne yapar:** Karşılaştırmayı, ölçülen bağlamın benzer olduğu katılımcılarla sınırlar. Ölçülen değişkenler için gerçek, tasarım temelli bir kontroldür.
- **Sınırları:** [FACT]
  - Artık dengesizlik: E2h'de yaş 0.703, saat 0.378, tarih 0.375 (Bölüm 2.3).
  - Küçük n: bilgi taşıyan tabakalarda 65 / 50 kişi.
  - Ölçülmeyen bağlam (dakika düzeyi, operatör, o günün klinik yoğunluğu) eşleşmez.
- **Eğitimde mi, değerlendirmede mi?**
  - *Yalnız değerlendirmede eşleştirmek* (tüm kohortta eğit, E2h tabakalarında değerlendir): eğitim dağılımını değiştirmez. Kestirme öğrenmiş bir model E2h'de zayıf görünebilir; bu muhafazakâr bir yanlılıktır.
  - *Eşleştirilmiş alt kümede eğitmek* (D-017'deki E4): kestirme ödüllendirilmez, ama fold başına ~90 katılımcıyla eğitim çok gürültülüdür. Snoek ve ark. alt örnekleme ile karşı-dengelemenin (counterbalancing) simülasyonlarında pozitif yanlılık ürettiğini bildirir [FROM PAPER; mekanizmanın bizim veriye taşınması NEEDS VERIFICATION]. Bu yüzden E4 yalnız dondurulmuş problarda bir duyarlılık analizidir, fine-tuning'de kullanılmaz.
- **Hüküm:** Gerçek kontrol. Ama ham E2h AUC'si yerine **E2h içinde koşullu lojistik regresyon** (tabaka = 1 saatlik dilim; kovaryatlar yaş, saat, tarih; ilgi katsayısı ses skoru) kullanılır.

### 3.3 Bağlam-only vs ses-only vs bağlam + ses (artımlı test, T1)
- **Ne yapar:** "Bağlam biliniyorken ses ne ekliyor?" sorusunu doğrudan test eder. Dinga ve ark.'nın önerisiyle uyumludur: confound'u girdiden temizlemek yerine model tahminleri üzerinde kontrol edilir [FROM PAPER]. Tüm katılımcıları kullanır.
- **Riskler ve çözümler:**

| Risk | Mekanizma | Çözüm | Kanıt |
|---|---|---|---|
| Tavan etkisi | Bağlam AUC 0.93; gerçek sinyal olsa bile ΔAUC binde birler düzeyinde | Karar istatistiği **olabilirlik temelli** (ses skoru katsayısının LR / Wald testi, SD başına OR). ΔAUC yalnız betimsel. | SIM-001: γ = 0.75'te güç ~%84, ortalama ΔAUC ≈ 0.004 |
| Bağlam modelinin yanlış belirtilmesi | Bağlam modeli bağlamın etiket ilişkisini tam yakalamazsa, bağlamı kodlayan ses skoru "fazladan bilgi" gibi görünür → **yanlış pozitif** | Esnek bağlam modeli (saat ve tarih için kısıtlı kübik spline, 4 df); duyarlılık 6 df. Kuadratik saat + doğrusal tarih modeli **kullanılmaz** | SIM-001 S2: kuadratik modelde yanlış pozitif 0.216, spline'da 0.060 |
| Ölçülmemiş, güne özgü bağlam | Aynı gün kaydedilenler bağımsız değil | Kayıt gününe göre küme-dayanıklı çıkarım (duyarlılık) + gün-gruplu CV | SIM-001 S3 |
| Sızıntı | Ses skoru katılımcının kendi etiketini görmüş olursa | Ses skoru **yalnız dış-fold OOF** tahmininden; T1 regresyonu bu skoru dışsal bir kovaryat olarak kullanır | yapısal |
| Pozitiflik | Örtüşme olmayan bölgelerde ayrım tanımsız | Kaçınılmaz; E2h-KLR ile birlikte yorumlanır | SIM-001 |

- **Hüküm:** **Birincil çıkarım aracı.** Spisak'ın *tam confounder testi* aynı soruyu parametrik olmayan biçimde sorar (H0: ŝ ⊥ Y | C, yani "model tamamen bağlam güdümlü" [FROM OFFICIAL DOCS, mlconfound]). Destekleyici test olarak eklenir.

### 3.4 Residualization / confound regression
- **Bilinenler:** [FROM PAPER]
  - Snoek ve ark. (2019): Confound regresyonunu **tüm veride** (CV'den önce) yapmak, simülasyonlarda şansın altında doğruluğa (negatif yanlılık) yol açtı. **Fold içinde** (train'de fit, teste uygula) yapıldığında bu yanlılık ortadan kalktı.
  - Dinga ve ark. (2020): Girdiden doğrusal residualization yetersizdir. Doğrusal olmayan modeller, confound'un varyans veya etkileşimlerdeki izini hâlâ kullanabilir.
- **Bizim veride ek sorun — eşdoğrusallık:** C, Y'yi 0.93 ile ele verdiği için C'yi gömmelerden tüm katılımcılarda regresyonla çıkarmak, gerçek astım sinyalinin C ile örtüşen kısmını da çıkarır. **Daha iyi varyant:** bağlamın gömmeler üzerindeki etkisini **yalnız hastalarda** tahmin et (etiket sabit, saat 196 / 86 değişkenlik gösteriyor), sonra bu yönü herkesten çıkar. Bu, Zhao ve ark.'nın etiket-koşullu yaklaşımıyla aynı mantıktır. Varsayımı, bağlam etkisinin sağlıklılarda da aynı olmasıdır. [INFERENCE + HYPOTHESIS]
- **Ham dalga formu + fine-tuning'e uygulanamaz:** Dalga formundan "saat" regresyonla çıkarılamaz. Ara katmanlarda yapmak ise eğitim dinamiğini değiştirir ve bu artık adversarial / koşullu kayıp ailesine girer.
- **Hüküm:** Yalnız dondurulmuş gömmelerde, fold içinde, hasta-içi tahmin edilen doğrusal bağlam yönleriyle; **duyarlılık analizi** olarak. Sonucu T1 ile değerlendirilir. EXP-022, yalnız G1 pozitifse.

### 3.5 Adversarial debiasing / gradient reversal
- **Marjinal GRL** ("gömme saati tahmin edemesin"): Saat etiketi güçlü biçimde ele verdiği için, gömmeyi saatten bağımsız yapmak onu büyük ölçüde etiketten de bağımsız yapar ve gerçek sinyal silinir. Zhao ve ark. (2020) tam da bu nedenle düşmanı yalnız etiket-koşullu bir alt kohortta eğitir ve koşulsuz sürümün daha kötü olduğunu bildirir [FROM PAPER]. **Bizde marjinal GRL yanlış hedeftir.** [INFERENCE]
- **Etiket-koşullu düşman** (ör. yalnız hastalarda saati tahmin etmeye çalışan düşman) teoride doğru hedeftir: ŝ ⊥ C | Y. Pratik sorunlar:
  - Min-max eğitimi küçük n'de kararsızdır.
  - λ ayarı bir doğrulama metriği ister; o metrik de confounded'dır.
  - Elazar ve Goldberg (2018), düşman eğitim sırasında şans düzeyindeyken bile, sonradan eğitilen yeni bir probun korunan özniteliği geri bulabildiğini gösterdi [FROM PAPER] → "düşman şans düzeyinde" ≠ "gömme bağlamdan arınmış".
  - Sağlıklılarda koşullu düşman yalnız 12 sabah kaydıyla eğitilebilirdi; bu pratikte imkânsız.
- **Hüküm:** Birincil hatta **yok**. Yalnız şu üç koşul birlikte sağlanırsa araştırma uzantısı olarak düşünülür:
  1. G1 güçlü pozitif;
  2. yeniden ağırlıklandırma yetersiz;
  3. sonuç yeni, sonradan eğitilen problarla doğrulanacak.

### 3.6 Katılımcı düzeyinde split ve grup temelli doğrulama
- **Katılımcı split'i zorunlu ama confounding kontrolü değil.** Kişi başı 7 kayıt var; kayıt veya segment düzeyinde split konuşmacı kimliğini sızdırır. Bağlam ise katılımcı düzeyinde bir özelliktir (herkesin tek bir seansı var). Bu yüzden katılımcı split'inde bağlam ↔ etiket ilişkisi train'de ve testte **aynen** kalır, ve kestirme test metriğinde de ödüllendirilir. Coppock ve ark. (2024) COVID-19 ses verisinde tam olarak bunu gösterdi: rastgele split'te AUC 0.846, ölçülen confounder'lar eşleştirilmiş test kümesinde ~0.62 [FROM PAPER].
- **Gün-gruplu CV** (aynı gün kaydedilenler aynı fold'da): Model, "bu günün akustiği = hasta günü" bilgisini aynı günün başka katılımcılarından öğrenemez. 65 kayıt gününün 53'ünde yalnız tek bir etiket var. [FACT] Bu yüzden güne özgü iz gerçek bir kestirme riskidir. Gün-gruplu CV bu riski kontrol eder; düzgün tarih eğilimi ve saat kestirmesini kontrol etmez. **Duyarlılık analizi** (EXP-021): gün-gruplu T1, katılımcı-gruplu T1'den belirgin düşükse, güne özgü sızıntı var demektir.
- **Bağlam-karşıt test kümesi** (sabah sağlıklılar + öğleden sonra hastalar üzerinde test): fikir doğru [INFERENCE; Chyzhyk ve ark.'nın confound'u en aza indiren test kümesi önerisiyle aynı mantık, ayrıntı NEEDS VERIFICATION]. Ama sabah sağlıklı yalnız 12 kişi. Bunun yerine aynı bilgiyi veren **karşıt-hücre tablosu** (Bölüm 4, L5) kullanılır.

### 3.7 Bilinen kayıt değişkenlerinin kontrolü (codec / cihaz / oda)
- Cihaz (iPhone 14), mesafe (10 cm) ve konum protokolde sabit [FACT, veri ekibi]. "Aynı konum" aynı akustik koşul demek değil (kapı, klima, yoğunluk) [INFERENCE].
- Codec zinciri dosya düzeyinde karışık: 285 katılımcıda iki zincir birlikte var. Etiketle ilişkili değil (p = 0.22), hastalarda saatle de değil (EXP-004). Yani bir **gürültü değişkeni**, confounder değil. [FACT]
- **Kontrol:**
  1. harmonizasyon;
  2. zincir probu (gömmeden zincir tahmin edilebiliyor mu, harmonizasyon öncesi ve sonrası);
  3. **katılımcı içi zincir duyarlılığı**: aynı kişinin Apple ve FFmpeg kayıtlarının model skorları sistematik olarak farklı mı? (eşleştirilmiş, 285 kişi).
- **Hüküm:** Gerekli, ama ana confounder'a dokunmuyor.

---

## 4. Birlikte kullanım: katmanlı yığın ve her modelde aynı "lens seti"

**Katman 0 — Hijyen (her deney):**
- Katılımcı düzeyinde tekrarlı tabakalı 5-fold. Tabakalar etiket × dönem × sabah/öğleden sonra; yaş grubu yalnız her hücrede ≥ 5 kişi kalıyorsa.
- D-021 harmonizasyonu; meta veri atılır.

**Katman 1 — Lens seti:** Her model (MFCC, dondurulmuş, fine-tune) **aynı OOF katılımcı skorlarından** şu çıktıları üretir:

| Lens | Ne | Neyi yanıtlar | Rol |
|---|---|---|---|
| L1 | E1 AUC (+ E1h, E2, E3), Bölüm 2.3'teki referans çizgileriyle | Makaleyle karşılaştırma | Betimsel; **kanıt değil** |
| L2 | **T1:** `y ~ spline(saat) + spline(tarih) + yaş + ŝ` lojistik regresyonu. ŝ = tekrarlar boyunca ortalaması alınmış OOF logit. Ses için LR testi; SD başına OR ve katılımcı-bootstrap %95 CI; gün-kümeli SE ile ve spline 6 df bağlam modeliyle duyarlılık. İkinci düzey CV ΔAUC betimsel. | Ses, ölçülen bağlam ve yaşın ötesinde bilgi taşıyor mu? | **Birincil** |
| L3 | Spisak testleri (`mlconfound`). Tam test, H0: ŝ ⊥ Y \| C. Kısmi test, H0: ŝ ⊥ C \| Y. | Tam: L2'nin parametrik olmayan teyidi. Kısmi: model bağlamı kullanıyor mu? | Destekleyici |
| L4 | **E2h-KLR:** E2h içinde, 1 saatlik dilimlere göre koşullu lojistik regresyon `y ~ ŝ + yaş + saat + tarih` (OR, CI). Ham E2h AUC'si referans çizgileriyle birlikte. | Az model varsayımıyla, örtüşme bölgesinde aynı soru | Destekleyici |
| L5 | **Karşıt-hücre tablosu:** etiket × sabah/öğleden sonra hücrelerinde ortalama skor (CI); hastalarda sabah vs öğleden sonra skor farkı (tarih ayarlı). | Model saati kullanıyorsa öğleden sonra hastaları "sağlıklı gibi" görünür | Kısmi testin yorumlanabilir hâli |
| L6 | Negatif kontroller N1–N5 (aşağıda) | Alternatif açıklamalar | Zorunlu |
| L7 | Zincir duyarlılığı (katılımcı içi eşleştirilmiş skor farkı) ve zincir probu | Teknik kanal | Zorunlu |
| L8 | Fold-fold SD, tekrarlar arası SD, tohum SD | Kararlılık | Zorunlu |

**Negatif kontroller:**
- **N1** dönem probu (D-020).
- **N2** etiket içinde skor ↔ tarih.
- **N3** yalnız arka plan (en düşük enerjili kareler).
- **N4** referans çizgileri: yaş-only, bağlam-only ve Bölüm 2.3.
- **N5 (yeni) bağlam-vekili kontrolü:**
  - *Kurulum:* Gömmelerden **yalnız hastalarda** sabah/öğleden sonra tahmin eden bir model eğitilir. Bu modelin OOF skoru, ses skorunun yerine T1'e konur.
  - *Beklenti:* Bilgi eklememesi. Eklerse, ses bağlamı bizim ölçtüğümüz saat ve tarihten daha ince kodluyor demektir; bağlam modeli yetersiz ayarlıyordur ve T1'in pozitif sonucu tek başına astım bilgisi diye okunamaz.
  - *Ne zaman bilgilidir:* Yalnız vekil modelin kendisi şanstan iyiyse (G1 pozitif). G1 negatifse N5 kendiliğinden boş çıkar ve bilgi taşımaz.
  - *Neden mümkün:* EXP-004b sabah ve öğleden sonra hastalarının klinik olarak benzer olduğunu gösterdi.

**Katman 2 — Azaltma (koşullu):** Yalnız G1 pozitifse veya L3 kısmi testi fine-tune modelinde anlamlıysa:
- **Bağlam-dengeli yeniden ağırlıklandırma:** ağırlık ∝ 1 / p̂(Y | C), train fold içinde çapraz-fit edilir, %1–99'da kırpılır, etkin örneklem büyüklüğü raporlanır. Fine-tuning'e doğrudan uygulanabilen tek yöntem bu: yalnız kayıp ağırlığıdır.
- **Hasta-içi residualization:** yalnız dondurulmuş gömmelerde (3.4).
- Etkinlikleri yine L2–L5 ile ölçülür.

---

## 5. Küçük örneklem riskleri

**E2h (~115 kişi: 65 astım / 50 sağlıklı):**
1. **Kesinlik:** Gerçek AUC 0.75'te %95 aralığın yarı genişliği ±0.125 (EXP-003 simülasyonu). Aynı-bağlam AUC'si 0.64 olan bir sinyali E2h-KLR yalnız yaklaşık yarı yarıya yakalar (SIM-001).
2. **Şans çizgisi 0.5 değil** (Bölüm 2.3) → ayarlı model gerekiyor. 50 sağlıklı olayla 4 parametre (ses + 3 kovaryat): olay / değişken oranı ~12, sınırda kabul edilebilir. Daha fazla kovaryat eklenmez. [INFERENCE]
3. **Artık dengesizlik** dilim içinde sürüyor (saat 0.378, tarih 0.375).
4. **Spektrum / seçilim:** E2h hastaları erken dönemin öğleden sonra hastalarıdır. EXP-002'ye göre erken ve geç dönem hastaları klinik olarak farklı. E2h sonucu "bağlamdan bağımsız sinyal var mı?" sorusunu test eder; ama etki büyüklüğü tüm hasta popülasyonuna genellenemez.
5. **E2h'de eğitim:** Fold başına ~90 katılımcı. Fine-tuning için kabul edilemez varyans; yalnız dondurulmuş probla duyarlılık.
6. **Araştırmacı serbestlik derecesi:** Dilim genişliği (1 saat) ve pencere sınırları D-023'te önceden sabitlendi. 2 saatlik dilim yalnız etiketli bir duyarlılık analizi olabilir; sonuca bakıp değiştirilmez.

**Tüm kohort için:**
7. **Sağlıklı n = 58–59; sabah sağlıklı 12.** İç doğrulama fold'unda ~9 sağlıklı var → eşik ve early stopping çok gürültülü. Birincil metrikler eşiksizdir; eşik yalnız ikincil metrikler için ve iç CV'den seçilir.
8. **Gün kümelenmesi:** 65 gün, 53'ü tek etiketli → tarihle ilgili etkiler için etkin örneklem 340'tan çok küçük. Gün-kümeli çıkarım duyarlılık olarak raporlanır (SIM-001 S3).
9. **Çoklu karşılaştırma:** 6 backbone × 7 görev × birçok tasarım. Onaylayıcı aile (confirmatory family): **katılımcı düzeyinde toplanmış 6 dondurulmuş backbone için T1**, Holm düzeltmesiyle. Görev bazlı ve tasarım bazlı her şey keşifsel (exploratory).
10. **Tohum varyansı:** Fine-tuning'de tohumlar arası fark, modeller arası farktan büyük olabilir. 3 tekrar × 5 fold, her tekrar kendi split'i ve tohumuyla; SD raporlanır.
11. **Seçim iyimserliği:** Fine-tune edilecek backbone aynı verideki dondurulmuş sonuçlara bakılarak seçilir (kilitli test seti yok, D-009 A). Fine-tune sonuçları bu yüzden "seçim sonrası" diye etiketlenir; onaylayıcı iddia dondurulmuş aileden gelir.

---

## 6. Önerilen deney tasarımı (ham ses → pretrained → fine-tuning)

Ön koşul: Faz 1 (split dosyaları, harmonize önbellek, MFCC EXP-010/011, T1 altyapısı; `scripts/evaluate_context.py` lens setini üretir).

### EXP-020 — Dondurulmuş gömmelerde bağlam probları (karar kapısı G1)
- **Soru:** Her backbone'un gömmesi, **etiket sabitken** bağlamı kodluyor mu?
- **Problar** (lojistik regresyon, katılımcı ortalaması gömme, CV, permütasyon eşiği):
  - (a) hastalarda sabah / öğleden sonra, tarih kovaryatlı: tarih-only modele karşı artımlı;
  - (b) hastalarda erken / geç dönem (N1, klinik referans 0.855 ile);
  - (c) dosya düzeyinde zincir (katılımcı-gruplu CV), harmonizasyon öncesi ve sonrası.
- **G1 pozitif:** (a)'da artımlı AUC'nin alt %95 sınırı permütasyon dağılımının %97.5'ini aşıyor.
- **Sonuçlar:**
  - G1+ → N5 bilgilidir; EXP-022 ve fine-tune'da yeniden ağırlıklandırma kolu planlanır.
  - G1− → bağlam kestirmesi dondurulmuş gömmelerde zayıf; fine-tune yine L3 kısmi testiyle izlenir (fine-tuning, alt katmanlarda bulunan bağlam bilgisini öne çıkarabilir).
- **Maliyet:** Gömme çıkarımı T4'te backbone başına dakikalar; problar CPU'da.

### EXP-021 — Dondurulmuş gömme + lineer prob, tüm lens setiyle (onaylayıcı aile)
- 6 backbone: CNN10, CNN14, CNN14_16k, BEATs, WavLM Base+, WavLM Large.
- Katılımcı düzeyinde toplanmış (7 görev logit ortalaması) = **onaylayıcı**; görev bazlı = keşifsel (RQ4/RQ5).
- Lojistik regresyon; C iç CV'de log-loss ile, küçük ızgara.
- Lens L1–L8. Gün-gruplu CV duyarlılığı. E4 (pencerede eğitim) duyarlılığı.
- **Fine-tune'a geçecek backbone'lar L2'deki (T1) sırayla seçilir, E1 ile değil.** E1'e göre seçmek en çok kestirme öğrenen modeli seçmek olabilir. Eşitlikte düşük kapasiteli olan seçilir.

### EXP-022 — Koşullu azaltma kolları (yalnız G1+)
- Hasta-içi residualization ve bağlam-dengeli yeniden ağırlıklandırma, dondurulmuş gömmelerde.
- Aynı lens seti. Beklenti [HYPOTHESIS]: L3 kısmi test p'si büyür (daha az bağlam kullanımı), L2 sabit kalır. L2 de düşerse, "astım sinyali" sandığımız şeyin bir kısmı bağlamdı.
- Adversarial: planlanmadı (3.5).

### EXP-030 — Fine-tuning (D-027)
| Öğe | Karar |
|---|---|
| Backbone | EXP-021'de T1'e göre ilk 1–2 (PANNs, Boll ile karşılaştırma için tercihen biri) |
| Girdi | Harmonize ham dalga formu, modelin kendi SR'ında. Frontend'i model hesaplar (D-007). |
| Birim | Her kayıt bir örnek; katılımcı skoru = kayıt logit'lerinin ortalaması. Bir katılımcının tüm kayıtları aynı fold'da. |
| Görev | Varsayılan: 7 görevin hepsi tek modelde (~1 900 train kaydı / fold). Görev bazlı fine-tune yalnız keşifsel. |
| Ne kadarı eğitilir | **Kısmi fine-tune:** son blok(lar) + sınıflandırma başı. Blok sayısı mimari başına önceden yazılır, smoke testte bellek ve süreyle doğrulanır. Tam fine-tune yalnız ablasyon. |
| Hiperparametreler | Sabit (Boll): Adam, lr 1e-4, wd 1e-4, batch 16. Ayar araması yok. |
| Epoch / early stopping | **Sabit epoch sayısı**, önceden yazılır. Bağlam-confounded bir doğrulama AUC'siyle early stopping veya checkpoint seçimi **yapılmaz**, çünkü o metrik kestirmeyi ödüllendirir. Epoch sayısı yalnız fold-0'ın iç train/val'ında, kayıp eğrisinin platosuna bakılarak bir kez belirlenir ve dondurulur [NEEDS VERIFICATION, smoke test]. İç doğrulama yalnız izleme içindir (log-loss, L5 tablosu). |
| Sınıf dengesizliği | Sınıf-dengeli kayıp ağırlığı (problarla aynı) |
| Kollar | **A:** standart. **B:** + bağlam-dengeli ağırlık. B yalnız G1+ ise ya da A'da L3 kısmi testi anlamlıysa. |
| Tekrar | 3 tekrar × 5 fold; tekrar r, EXP-021'in r. split'ini ve r. tohumunu kullanır → dondurulmuş ve fine-tune eşleştirilmiş. |
| Değerlendirme | Aynı lens seti + **eşleştirilmiş karşılaştırma:** fine-tune vs dondurulmuş, aynı katılımcılarda T1 sapma (deviance) azalması farkı (katılımcı bootstrap) ve L3 kısmi test. |
| Kayıt ve kurtarma | Her epoch `latest.pt` (atomik), periyodik checkpoint, epoch logu (kayıp, lr, süre, iç izleme metrikleri), config, tohum, git commit. Sabit epoch sayısı olduğu için "en iyi val" checkpoint'i birincil değil; son epoch kullanılır. |
| Smoke test (zorunlu) | 1 batch forward, 1 train adımı, 1 doğrulama adımı, şekil / dtype / cihaz assert'leri, GPU bellek, epoch süresi ölçümü → bütçe tahmini |

**Önceden yazılmış beklenti [HYPOTHESIS]:** Fine-tuning, E1'i T1'den daha çok artıracak ve L3 kısmi confounding'i artıracak. Çünkü kodlayıcı, etiketi ayıran her şeye uyum sağlar, bağlam dahil. Böyle çıkarsa bu bir **olumsuz ama değerli sonuçtur**: "Fine-tuning burada kestirmeyi güçlendirdi."

### Önceden yazılmış iddia kuralları
- **İddia C1 — "ses, ölçülen bağlam ve yaşın ötesinde astım bilgisi taşıyor":** Bir backbone için aşağıdakilerin hepsi:
  1. L2'de ses katsayısı LR testi, Holm düzeltmeli (6 backbone) p < 0.05;
  2. OR'un katılımcı-bootstrap CI'ı 1'i dışlıyor ve gün-kümeli duyarlılıkta yön korunuyor;
  3. L4 E2h-KLR OR'u aynı yönde (CI'ın 1'i dışlaması **şart değil**, güç düşük);
  4. N3 etkiyi yeniden üretmiyor; N5 bilgiliyse anlamlı değil;
  5. spline 4 df ve 6 df bağlam modelleri aynı yönü veriyor.
- **İddia C2 — "model bağlamı kullanıyor":** L3 kısmi test p < 0.05 veya L5'te hasta-içi sabah / öğleden sonra skor farkının CI'ı 0'ı dışlıyor. **C1 ve C2 birlikte doğru olabilir:** model hem gerçek sinyal hem kestirme taşıyabilir.
- **İddia C3 — "fine-tuning bağlamdan bağımsız bilgiyi artırdı":** Eşleştirilmiş T1 sapma azalması farkının CI'ı 0'ı dışlıyor. Seçim sonrası olduğu için keşifsel etiketiyle raporlanır.
- **C1 sağlanmazsa:** "Bu veriyle bağlamdan bağımsız bir ses sinyali gösterilemedi; SIM-001'e göre aynı-bağlam AUC'si ≈ 0.69 (α = 0.05) / 0.74 (Holm düzeyi) altındaki etkileri saptama gücümüz düşüktü" yazılır. Bu da yayınlanabilir, değerli bir sonuçtur.

---

## 7. Bilinçli olarak yapmayacaklarımız
- Gürültü giderme, sabit gürültü ekleme, DAW'dan yeniden export (3.1).
- Tüm veride (CV öncesi) confound regresyonu (Snoek: negatif yanlılık).
- Marjinal gradient reversal (3.5).
- Ham E2h AUC'sini 0.5'e karşı iddia olarak kullanmak (2.3).
- Fine-tune backbone'unu E1'e göre seçmek; bağlam-confounded doğrulama metriğiyle early stopping.
- Saat veya tarihi modele **girdi** olarak vermek: soru sesin katkısıdır; bağlam yalnız değerlendirmede ayarlanır.

---

## 8. Açık noktalar ve doğrulanacaklar
- [NEEDS VERIFICATION] `mlconfound`'un çok değişkenli C'yi kabul edip etmediği. Etmiyorsa C = çapraz-fit edilmiş bağlam eğilim skorunun (`logit p̂(Y | yaş, saat, tarih)`) tek boyutlu özeti kullanılır [DECISION]. `cat_y=True`, olasılık skoru için `cat_yhat=False` [FROM OFFICIAL DOCS].
- [NEEDS VERIFICATION] Spline df (4) önceden sabitlendi; SIM-001 bu seçimle yapıldı.
- [NEEDS VERIFICATION] Kısmi fine-tune blok sayıları ve sabit epoch sayısı → smoke test.
- [NEEDS VERIFICATION, veri ekibi S13] Ses kaydı bronkodilatör testinden önce mi sonra mı alındı? Sonra ise bu, saat kontrolünün çözemeyeceği bir prosedür yoludur.
- Kalıcı çözüm hâlâ veri toplamaktır: sabah sağlıklı ve öğleden sonra hasta kayıtları, aynı protokolle (rapor 6.6). Bu belge yalnız mevcut veriyle yapılabilecek en dürüst analizi tanımlar.
