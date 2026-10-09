# Model seçimi, SMOTE ve dengesizlik: metodolojik değerlendirme

**Tarih:** 2026-10-09 (11. tur) · **Bağlam:** Alper'in EXP-010/011 sonrası soruları · **Deneyler:** EXP-013 (etiket dengesizliği), EXP-014 (bağlam dengesizliği), EXP-015 (iç içe CV). Üçü de **keşifsel**: sorulara cevap olarak, sonuçlar görüldükten sonra tasarlandı. Bütün AUC'ler üst sınırdır (D-028).
**Kod:** `scripts/analyze_imbalance_selection.py` · **Raporlar:** `reports/mfcc/EXP-013_imbalance.md`, `EXP-014_context_balancing.md`, `EXP-015_nested_cv.md`

---

## 0. Kısa cevaplar

1. **"EXP-011'i makalenin AUC'leriyle kıyaslamak doğru olmaz mı?"**
   - Makalenin 5-fold CV'si sabit bir model için doğru bir ölçüm. Ama yayınlanan sayılar CV'ye ek olarak iki şey içeriyor: 14 model arasından **test sonucuna bakarak seçim** ve **tek bir bölünmenin şansı**. Bunlar ayrı düşünülmeli.
   - Seçimi dürüstçe yapan iç içe CV, makalenin prosedürü için görev başına ortalama **0.654** veriyor. Makalede 0.709, EXP-011 LR'de 0.660.
   - Yani seçim iyimserliği çıkarılınca **bulgularımız makaleyle uyumlu**. Fark ~0.05 iyimserlikten geliyor (EXP-015).
2. **"İç içe CV ne öğretir?"**
   - Makalenin prosedürünün gerçek performansını (yukarıdaki 0.654).
   - Seçimin kararsız olduğunu: 7 görevin 6'sında 25 fold boyunca 5–8 farklı model seçildi.
   - Faz 2 için ders: 6 backbone arasından seçim yaparken aynı tuzak var.
3. **"SMOTE'u kullanmadık mı? Sınırlılıkları?"**
   - EXP-011'de kullanmadık (sınıf ağırlığı). EXP-010'da makaleyi taklit etmek için kullandık.
   - EXP-013: SMOTE, sınıf ağırlığı, rastgele aşırı örnekleme ve hiçbiri **aynı AUC'yi** veriyor (fark ≤ 0.01, CI'lar 0'ı içeriyor).
   - SMOTE ve ağırlık **kalibrasyonu bozuyor**: LR'de astım olasılığı ortalama 0.25 düşük tahmin ediliyor.
   - Dengeli doğrulukta da avantaj yok: eşik iç CV'den seçilince "hiçbiri" de aynı sonucu veriyor (0.698).
   - Makaleyle aramızdaki duyarlılık / özgüllük farkları **çalışma noktası** farkıdır, ayırma gücü farkı değil.
4. **"SMOTE ile saat / tarih dengesizliği düzeltilebilir mi?"**
   - Hayır, örtüşme bölgesinin ötesinde düzeltilemez. Geç dönemde hiç sağlıklı yok; SMOTE yalnızca var olan örneklerin arasına nokta koyar.
   - EXP-014: hücre-içi SMOTE ve ağırlık fark yaratmıyor. Yalnız örtüşmede eğitim AUC'yi ~0.07 düşürüyor.
   - Doğru yer eğitim değil **değerlendirme**.
5. **"Sınıf ağırlığı ya da başka bir kayıp?"**
   - Derin modellerde **ağırlıklı BCE**, eşik iç CV'den ve kalibrasyon raporlanır.
   - Focal / LDAM / logit ayarlaması bu dengesizlik düzeyinde (1:4.8) gerekmiyor.
   - Grup-dengeli kayıplar (Group DRO vb.) yalnız değerlendirme aşamasında ve örtüşmede düşünülebilir (Bölüm 5).

---

## 1. İki ayrı iş: performansı *tahmin etmek* ve model *seçmek*

**Çapraz doğrulama (CV) bir ölçüm aracıdır.** Sabit bir yöntemin, hiç görmediği katılımcılarda ne kadar başarılı olacağını tahmin eder. Makalenin 5-fold CV'si katılımcı düzeyinde ve sızıntısız; tek bir sabit model için doğru bir tahmin aracıdır. [FROM PAPER + FACT]

**Sorun CV'nin kendisinde değil, üstüne eklenen iki adımda:**

| Adım | Makalede | Etkisi | Bizde nasıl ölçtük |
|---|---|---|---|
| Tek bir 5-fold bölünmesi | `random_state=42`, bir kez | Sonuç, katılımcıların hangi fold'a düştüğüne bağlı: **split şansı** | EXP-012 C: 50 farklı bölünme → görev başına ortalamanın SD'si 0.02–0.03, aralık 0.08–0.14 |
| 14 model arasından **aynı test sonuçlarına bakarak** "en iyi"yi seçmek | Her görevde ayrı | Seçilen sayı sistematik olarak yüksek: **seçim iyimserliği** (Cawley & Talbot 2010; Varma & Simon 2006) | EXP-015: iç içe CV |
| Başlıkta 7 görev × 14 model arasından en yüksek sayıyı vermek | "gelecek, AUC 0.769" | İkinci kat seçim | Aynı mantık |

Seçim iyimserliğinin kaynağı: 14 modelin gerçek performansları birbirine yakınsa, hangisinin "kazandığını" büyük ölçüde o bölünmedeki şans belirler. Kazanan, şansı en çok yaver gidendir. Aynı model yeni katılımcılarda bu şansı tekrar yaşamaz. [FROM PAPER: Cawley & Talbot 2010]

**Bu yüzden "EXP-011'i makalenin AUC'leriyle kıyaslayalım mı?" sorusunun cevabı şu:** Kıyaslanabilir, ama tek bir farkı ölçmez. EXP-011 makaleden **dört** şeyle ayrılıyor:
1. girdi (harmonize önbellek);
2. sınıflandırıcı (önceden seçilmiş LR; 14 modelden en iyisi değil);
3. dengesizlik yöntemi (sınıf ağırlığı; SMOTE değil);
4. tekrarlı bölünme (5×5; tek 5-fold değil).

Fark bu dördünün toplamıdır. Doğru kıyaslar ayrı sorulara ayrı cevap verir:

| Soru | Doğru kıyas | Cevap |
|---|---|---|
| Verimiz ve özellik tarifimiz makaleyle tutarlı mı? | Makale protokolü, aynı veride (EXP-010) | Evet: en iyi modellerin ortalaması 0.702, makalede 0.709 |
| Makalenin **kendi prosedürü** (14 model + seçim) dürüstçe ne verir? | İç içe CV (EXP-015) | Görev başına ortalama **0.654** (EXP-015). Makaledeki 0.709'dan ~0.05 düşük; EXP-011 LR'yle (0.660) neredeyse aynı |
| Sabit, önceden seçilmiş basit bir model ne verir? | EXP-011 LR | Görev başına 0.64–0.69; füzyon 0.771 |
| Girdi (harmonizasyon) farkı ne kadar? | Aynı protokol, iki girdi (EXP-012 B) | Ortalamada fark yok (0.664 vs 0.660) |
| SMOTE farkı ne kadar? | Aynı model, aynı bölünme, yalnız dengesizlik yöntemi farklı (EXP-013) | Hiç: AUC farkı ≤ 0.01 (EXP-013). Fark kalibrasyonda ve 0.5 eşiğindeki metriklerde |

---

## 2. İç içe (nested) CV — EXP-015

**Ne yapar?**
- **Dış döngü:** Yalnız raporlanacak tahmin için kullanılır. Bizim 25 fold'umuz.
- **İç döngü:** Her dış eğitim kümesinin içinde 5-fold. 10 modelin her biri iç döngüde değerlendirilir. İç ortalama AUC'si en yüksek olan seçilir, bütün dış eğitim kümesiyle yeniden eğitilir ve dış test fold'unda **bir kez** ölçülür.

Dış test fold'u seçime hiç karışmadığı için çıkan sayı, "14 modelden en iyisini seç" **prosedürünün** dürüst performansıdır. Bu, makalenin aslında raporlaması gereken sayıdır. [FROM PAPER: Varma & Simon 2006]

**Ek olarak öğrendiklerimiz:**
- Seçim **kararlı** mı? Her fold'da aynı model mi seçiliyor?
- "Sonradan bakarak en iyi sabit model" ve "her fold'da kâhin gibi en iyiyi seçmek" ne verir? Bu ikisi iyimser üst sınırlardır; iç içe CV ile aralarındaki fark iyimserliğin büyüklüğüdür.

**Kapsam sınırı:** CatBoost ve üç ensemble hesap maliyeti nedeniyle dışarıda kaldı: 10 model. Makalede bu dört model 7 görevden yalnız 1'inde (gelecek, Stacking) kazanandı.

**Sonuçlar** (makale özellikleri ve pipeline'ı, 10 model, 25 dış fold; görev başına ortalamalar 7 görevin ortalamasıdır):

| Sayı | Ne demek? | 7 görev ortalaması |
|---|---|---|
| Her fold'da en iyiyi test sonucuna bakarak seçmek ("kâhin") | En uç iyimserlik | 0.729 |
| Makale, Tablo 4 (14 model, tek bölünme, test sonucuna göre seçim) | Yayınlanan | 0.709 |
| EXP-010 (aynı prosedür, bizim veri, tohum 42) | Yeniden üretim | 0.702 |
| 25 fold'da sonradan bakarak en iyi sabit model | Hâlâ iyimser (seçim sonradan) | 0.683 |
| **İç içe CV: prosedürün dürüst performansı** | **Makalenin raporlaması gereken sayı** | **0.654** |
| 10 modelin medyanı | Rastgele bir model seçseydik | 0.642 |
| EXP-011 LR (önbellek, sınıf ağırlığı, önceden seçilmiş) | Bizim taban | 0.660 |

Görev başına iç içe sonuçlar:

| görev | iç içe AUC | makale kuralıyla EXP-010 | fark | 25 fold'da seçilen modeller |
|---|---|---|---|---|
| aaa | 0.693 ± 0.078 | 0.725 | −0.03 | LR 14, MLP 11 |
| araba | 0.598 ± 0.067 | 0.686 | −0.09 | 8 farklı model (LR 8, XGB 6, MLP 4, …) |
| ana | 0.630 ± 0.066 | 0.687 | −0.06 | 8 farklı model |
| ordu | 0.644 ± 0.082 | 0.677 | −0.03 | 8 farklı model (SVM 11) |
| gelecek | 0.681 ± 0.089 | 0.719* | −0.04 | XGB 9, MLP 8, GB 5, … |
| titiz | 0.667 ± 0.094 | 0.695 | −0.03 | SVM 10, MLP 8, … |
| ünlem | 0.664 ± 0.071 | 0.723 | −0.06 | XGB 8, MLP 7, AdaBoost 7, … |

\* gelecek'in EXP-010 kazananı Stacking'di; Stacking iç içe kapsamda değil.

Füzyon (her görevde iç CV'nin seçtiği model): 0.734 ± 0.087; havuzlanmış 0.759 [0.690–0.822]. EXP-011 LR füzyonu: 0.771; havuzlanmış 0.778 [0.713–0.836].

**Ne öğrendik?** [FACT + INFERENCE]
1. **Makalenin "en iyi" sayıları ortalama ~0.05 (görev başına 0.03–0.09) iyimser.** Prosedürün dürüst performansı (0.654), önceden seçilmiş basit LR'ninkiyle (0.660) aynı. Yani 14 model denemek bu veride bir şey kazandırmıyor; yalnızca daha yüksek görünen bir sayı üretiyor.
2. **Seçim kararsız.** Yedi görevin altısında 25 fold boyunca 5–8 farklı model seçildi (yalnız aaa'da LR ve MLP arasında gidip geldi). "Görev X için en iyi model Y" türünden cümleler bu örneklemde anlamlı değil.
3. **Doğru karşılaştırma şu:** EXP-011 (0.660) makalenin iç içe CV'deki karşılığıyla (0.654) uyumlu. Makalenin yayınlanan sayısıyla (0.709) olan fark, büyük ölçüde seçim iyimserliği ve tek bölünme şansıdır, "farklı bulgu" değil.

**Faz 2 için asıl ders:** 6 backbone × 7 görev arasından seçim yapacağız. Aynı iyimserlik orada da oluşur. Seçim ya önceden sabitlenmeli (D-032: birincil birim füzyon) ya da iç içe yapılmalı. Seçilmeyen bütün konfigürasyonlar da raporlanmalı (D-009). [INFERENCE]

---

## 3. SMOTE: kullandık mı, ne kazandırır, ne kaybettirir? — EXP-013

**Durum:**
- EXP-010'da makaleyi taklit etmek için SMOTE kullanıldı (yalnız eğitim fold'unda, makaledeki gibi).
- EXP-011'de kullanılmadı. Yerine **sınıf ağırlığı** kullanıldı: kayıp fonksiyonunda sağlıklı örneklere ~4.8 kat ağırlık. [FACT]

**SMOTE ne yapar?** Azınlık sınıfındaki her örnek için, özellik uzayında ona en yakın 5 azınlık komşusu bulunur. Aralarına rastgele noktalar konur. Yani yeni ("sentetik") sağlıklılar, gerçek sağlıklıların özellik vektörlerinin karışımlarıdır (Chawla ve ark. 2002).

**Kullanmanın sınırlılıkları:**

| # | Sınırlılık | Kaynak / kanıt |
|---|---|---|
| S1 | **Ayırmayı (AUC) iyileştirmez.** AUC sıralamaya bakar; sınıf oranı değişince sıralama pek değişmez. | van den Goorbergh ve ark. 2022 (LR, simülasyon + gerçek veri) [FROM PAPER]; EXP-013 [FACT] |
| S2 | **Kalibrasyonu bozar.** Model, azınlık sınıfının olasılığını sistematik olarak fazla tahmin eder. Klinik risk tahmini için zararlı. | van den Goorbergh ve ark. 2022 [FROM PAPER]; EXP-013 |
| S3 | Duyarlılık / özgüllükteki "iyileşme" aslında yalnızca **eşiğin kaymasıdır**. Aynı etki eşiği değiştirerek, sentetik veri üretmeden elde edilir. | van den Goorbergh ve ark. 2022 [FROM PAPER] |
| S4 | Yüksek boyutta sınırlı fayda; azınlık varyansını küçültür, sentetik örnekler kaynaklarıyla korelasyonludur, k-NN'i azınlığa yanlı yapar. | Blagus & Lusa 2013 [FROM PAPER]. Bizde 44 boyut ve fold başına ~47 sağlıklı: orta boyut. |
| S5 | Sentetik örnekler **bağımsız katılımcı değildir.** Eğitim kümesi içinde yapılan ikinci bir CV'de (Stacking'in iç CV'si, ızgara araması) aynı kişinin "türevleri" hem eğitimde hem doğrulamada olabilir. Bu iç seçimleri iyimser yapar (dış test geçerli kalır). | [INFERENCE] |
| S6 | **Confounding'i düzeltmez.** Azınlığı azınlık yapan her şeyi çoğaltır: hastalık bilgisi, kayıt saati / tarihinin izi, yaş. | [INFERENCE]; Bölüm 4 |
| S7 | Ham ses / dalga formu için anlamsız: iki kaydın arası "ortalama bir ses" değildir. Ancak özellik ya da gömme uzayında uygulanabilir. | [INFERENCE] |

**Kullanmamanın sınırlılıkları:**
- **Hiç düzeltme yoksa:** Olasılıklar gerçek oran olan %83 astıma göre kalibre olur, AUC aynı kalır. Ama sabit 0.5 eşiğinde neredeyse herkes "astım" denir (EXP-011'deki SVM sorunu).
- **Sınıf ağırlığı:** SMOTE gibi kalibrasyonu azınlık lehine kaydırır, ama sentetik veri üretmez ve belirlenimcidir.
- Her iki durumda da çözüm aynı: **eşiği iç CV'den seçmek** (D-032) ve gerekiyorsa kalibrasyonu düzeltmek.

**Kıyaslarken farkı nasıl yorumlamalı?**
- **AUC farkı:** Aynı bölünmelerde ±0.02'yi aşmıyorsa gürültüdür. EXP-012 C'ye göre tek bir 5-fold'un kendisi ±0.05 oynuyor.
- **Duyarlılık / özgüllük / dengeli doğruluk farkı:** Sabit 0.5 eşiğinde ölçülüyorsa, ayırma gücü farkı değil **çalışma noktası farkıdır**.
- **Kalibrasyon farkı:** Gerçektir, ama AUC hakkında bir şey söylemez.
- Makaledeki "yüksek duyarlılık, düşük özgüllük" örüntüsü (0.88–0.94 / 0.18–0.48) bu çalışma noktası etkisidir; EXP-010'da birebir yeniden çıktı.

**EXP-013 sonuçları** (füzyon, 25 fold; LR'nin C'si iç CV'de seçildi; MLP varsayılan):

| model | yöntem | AUC [%95 CI] | dengeli doğr. @0.5 | dengeli doğr. @iç-CV eşiği | Brier | ort. tahmin − gerçek oran |
|---|---|---|---|---|---|---|
| LR | hiçbiri | 0.776 [0.713–0.835] | **0.500** | 0.698 | **0.134** | **0.000** |
| LR | sınıf ağırlığı | 0.778 [0.713–0.836] | 0.667 | 0.696 | 0.185 | −0.244 |
| LR | SMOTE | 0.768 [0.702–0.826] | 0.669 | 0.678 | 0.188 | −0.247 |
| LR | rastgele aşırı örnekleme | 0.778 [0.714–0.834] | 0.662 | 0.687 | 0.189 | −0.250 |
| MLP | hiçbiri | 0.788 [0.724–0.843] | 0.516 | 0.695 | 0.125 | +0.042 |
| MLP | sınıf ağırlığı | 0.790 [0.729–0.846] | 0.560 | 0.686 | 0.122 | −0.009 |
| MLP | SMOTE | 0.788 [0.725–0.844] | 0.567 | 0.696 | 0.123 | +0.005 |
| MLP | rastgele aşırı örnekleme | 0.786 [0.723–0.844] | 0.550 | 0.681 | 0.123 | +0.012 |

Eşleştirilmiş farklar ("yöntem − hiçbiri", füzyon AUC): LR'de −0.008 ile +0.002, MLP'de −0.001 ile +0.002; hepsinin CI'ı 0'ı içeriyor.

**Okuma:** [FACT; literatürle tutarlı]
1. **Ayırma (AUC):** Dört yöntem arasında fark yok. van den Goorbergh ve ark. 2022'nin bulgusu bizim veride de geçerli.
2. **Kalibrasyon (LR):** Düzeltme yapılmayınca ortalama tahmin gerçek astım oranına eşit (fark 0.000) ve Brier en iyisi (0.134). SMOTE, ağırlık ve aşırı örnekleme astım olasılığını ortalama ~0.25 düşük tahmin ediyor, yani azınlık sınıfı olan sağlıklıyı fazla tahmin ediyor; Brier 0.185–0.189'a çıkıyor. Bu, literatürde tarif edilen zararın aynısı.
3. **Eşik:**
   - Düzeltme yoksa 0.5 eşiğinde herkes "astım" deniyor (dengeli doğruluk 0.500). Makalenin ve EXP-011 SVM'sinin yaşadığı sorun bu.
   - Eşik iç CV'den seçilince düzeltmesiz model de aynı dengeli doğruluğa ulaşıyor (0.698 vs 0.696 / 0.678).
   - Yani düzeltmenin tek "faydası" eşiği kaydırmak; bu, sentetik veri üretmeden yapılabilir.
4. **MLP:** AUC LR ile aynı düzeyde (füzyon 0.786–0.790 vs 0.776–0.778). MLP eğitim verisini neredeyse ezberlediği için yeniden örneklemeden pek etkilenmiyor; kalibrasyonu her durumda kabul edilebilir.
   - Görev başına yalnız titiz'de belirgin biçimde daha iyi (0.71–0.72 vs 0.67).
   - D-032'deki "güçlü ikinci taban" önerisinin ölçüsü şu: MLP füzyonu LR füzyonundan anlamlı biçimde yüksek değil. Yani çıta pratikte değişmiyor, ama MLP yine de ikinci taban olarak raporlanabilir.

---

## 4. SMOTE ile kayıt saati / tarihi dengesizliği düzeltilebilir mi? — EXP-014

**Önce kavram: iki ayrı "dengesizlik" var.**

| | Etiket dengesizliği | Bağlam dengesizliği (confounding) |
|---|---|---|
| Ne? | 283 astım / 59 sağlıklı | Astımın olasılığı kayıt bağlamına göre çok değişiyor |
| Sorun | Eşik ve kalibrasyon | Model, bağlamın sese bıraktığı izi "astım" diye öğrenebilir |
| SMOTE / ağırlık | İşe yarar (eşik / kalibrasyon için; AUC için değil) | Yalnız **örtüşme bölgesinde** dengeleyebilir |

**Veri yapısı (pozitiflik) — sorunun özü bu tablo:**

| hücre (dönem × seans) | astım | sağlıklı |
|---|---|---|
| erken – sabah | 76 | 12 |
| erken – öğleden sonra | 21 | 46 |
| geç – sabah | 120 | **0** |
| geç – öğleden sonra | 65 | **0** |

**SMOTE'un yapamayacağı şey** [FACT, SMOTE'un tanımı gereği]: SMOTE yalnızca **var olan** azınlık örneklerinin arasına nokta koyar. Geç dönemde hiç sağlıklı kayıt yok. Üretilecek "geç dönem sağlıklısı", erken dönem sağlıklılarının özelliklerinin karışımı olur; onlara "geç dönem" etiketi yapıştırmak sesteki dönem izini değiştirmez. Yani 185 geç dönem hastası için dengeleme **imkânsızdır**.

Sabah hücresinde 12 sağlıklıyı 76'ya çıkarmak ise o 12 kişinin özelliklerini (yaşları, sesleri) çoğaltır. Model "sabah sağlıklısı" diye 12 bireyi ezberleyebilir.

**Yapılabilecek en iyi şey:** Örtüşme bölgesinde (erken dönem), her hücrede etiketleri dengelemek. Bunun iki yolu var:
- SMOTE ile dengelemek;
- ağırlıkla dengelemek (ters eğilim ağırlığı; Idrissi ve ark. 2022'nin "basit grup dengeleme" yaklaşımı).

Bu, eğitimde modelin bağlamdan yararlanmasını zorlaştırır ama **kanıt üretmez**. İşe yarayıp yaramadığı yine bağlama koşullu bir değerlendirmeyle ölçülmeli. [INFERENCE + FROM PAPER]

**EXP-014'te denenenler** (LR, füzyon, 25 fold):
- V0: mevcut taban (sınıf ağırlığı, tüm veri).
- V2 / V3: tüm veride hücre-içi SMOTE / hücre ağırlığı (geç dönem değişmeden kalır).
- V4 / V5: **yalnız erken dönemde** eğitim + hücre ağırlığı / hücre-içi SMOTE. Eğitim verisinde etiket ve bağlam bağımsız hâle gelir.

Değerlendirme dört ölçütle yapıldı:
- bütün test;
- yalnız erken dönem;
- **hücre-içi AUC:** yalnız aynı hücredeki astım–sağlıklı çiftleri karşılaştırılır. Bu, kaba bağlam sabitken ayırma gücüdür.
- **bağlam bağımlılığı:** etiket sabitken skorun bağlamı ayırma gücü.

| | tüm test | yalnız erken dönem | hücre-içi | hastalarda skor: geç vs erken | hastalarda skor: sabah vs öğleden sonra |
|---|---|---|---|---|---|
| V0 sınıf ağırlığı, tüm veri | **0.777** | 0.739 | 0.706 | 0.556 | 0.524 |
| V2 hücre-içi SMOTE, tüm veri | 0.769 | 0.723 | 0.709 | 0.587 | 0.526 |
| V3 hücre ağırlığı, tüm veri | 0.760 | 0.721 | 0.718 | 0.562 | 0.521 |
| V4 yalnız erken + hücre ağırlığı | 0.707 | 0.673 | 0.649 | 0.548 | 0.516 |
| V5 yalnız erken + hücre-içi SMOTE | 0.713 | 0.679 | 0.650 | 0.555 | 0.514 |
| *ref: yalnız yaş* | 0.678 | 0.666 | 0.649 | — | — |
| *ref: yalnız bağlam* | 0.930 | 0.813 | 0.578 | — | — |

**Bulgular:** [FACT; yorum INFERENCE]
1. **SMOTE / ağırlıkla bağlam dengeleme daha iyi ya da daha dürüst bir taban üretmedi.**
   - Tüm veride (V2, V3) fark yok: −0.01 / −0.02, CI'lar 0'ı içeriyor.
   - Yalnız örtüşme bölgesinde eğitmek (V4, V5) AUC'yi her ölçütte ~0.06–0.07 düşürdü (CI'lar 0'ı dışlıyor). Sebep: eğitim verisinin ~%54'ü atılıyor (185 geç dönem hastası).
   - Bağlam bağımlılığı göstergeleri değişmedi.
2. **MFCC + LR modeli bağlama zaten zayıf biçimde bağımlı.** Hastalar içinde skorun geç / erken dönemi ayırma gücü 0.556, sabah / öğleden sonrayı ayırma gücü 0.524 (0.5 = bağımlılık yok). EXP-004'teki "kaba akustik bağlam izi yok" bulgusuyla tutarlı. Dengelenecek çok şey yoktu.
3. **Kaba bağlam sabitken bile füzyon ayırıyor.** Hücre-içi AUC 0.706; aynı ölçütle yaş 0.649, bağlam 0.578.
   - Bu, sesin kaba bağlamın ötesinde bilgi taşıyabileceğine işaret ediyor.
   - Ama yaş ayarlanmadı; dakika ve gün düzeyindeki bağlam ayarlanmadı; erken–sabah hücresinde yalnız 12 sağlıklı var.
   - Bu yüzden **kanıt değil, değerlendirme aşaması için bir hipotez** (D-023, D-028).

**Sonuç:** SMOTE ile "bağlamdan arınmış" yeni bir taban üretilemez. Dengeleme yalnız örtüşmede mümkün ve orada da veri kaybı kazancından büyük. Doğru yer eğitim değil **değerlendirme**: hücre-içi ya da bağlam-ayarlı ölçüm (D-023'teki T1, E2h). EXP-014'teki hücre-içi AUC bunun kaba bir ön izlemesi.

---

## 5. İleride (derin modeller): sınıf ağırlığı ve kayıp fonksiyonları

| Seçenek | Ne yapar | Bizim veride | Öneri |
|---|---|---|---|
| **Ağırlıklı BCE** (sınıf ağırlığı) | Sağlıklı örneklerin kaybı ~4.8 kat sayılır | Eğitimde gradyan dengesi; AUC'yi değiştirmez (EXP-013) | **Varsayılan** [DECISION önerisi] |
| Dengeli örnekleyici (batch başına eşit sınıf) | Azınlığı daha sık gösterir | Ağırlıklı BCE'ye denk; küçük veride aynı 59 kişiyi tekrar tekrar gösterir → aşırı uyum riski | Alternatif |
| Focal loss (Lin ve ark. 2017) | Kolay örneklerin kaybını kısar | Aşırı dengesizlik (1:100+) ve yoğun tespit için tasarlandı; 1:4.8'de gerekçesi zayıf | Hayır |
| Logit ayarlaması (Menon ve ark. 2021) | Sınıf önceliğini logit'ten çıkarır (kayıpta ya da sonradan) | Dengeli hatayı hedefler; ağırlıklı BCE + iç-CV eşiğiyle aynı amaca hizmet eder | Gerekmez |
| LDAM (Cao ve ark. 2019) | Azınlık sınıfı için daha büyük marj | Uzun kuyruklu çok sınıflı görevler için | Hayır |
| SMOTE (gömme uzayında) | Sentetik gömme | EXP-013: AUC'ye katkı yok, kalibrasyonu bozar; uçtan uca eğitimde uygulanamaz | Hayır |
| **Grup-dengeli ağırlık / Group DRO** (Sagawa ve ark. 2020) | Etiket × bağlam gruplarında en kötü grubun kaybını küçültür | Pozitiflik: geç dönemde sağlıklı grubu yok, sabahta 12 kişi → en kötü grup bu 12 kişi; aşırı uyum | Yalnız değerlendirme aşamasında, örtüşmede (D-026, D-028) |
| Son katmanı grup-dengeli veriyle yeniden eğitmek (Kirichenko ve ark.) | Gömmeyi sabit tutup yalnız sınıflandırıcıyı dengeli veride eğitir | Dondurulmuş gömme + lineer prob yapımıza doğal olarak uyar; ama EXP-014 V4'teki veri kaybı sorunu aynen geçerli | Değerlendirme aşamasında aday [ayrıntılar NEEDS VERIFICATION] |

**Ayrıca her derin model deneyinde:**
- Birincil metrik AUC.
- Eşik iç CV'den (D-032).
- Kalibrasyon raporlanır (Brier, eğim). Klinik olasılık gerekiyorsa iç CV'de yeniden kalibre edilir (Platt / izotonik).

---

## 6. Önerilen kararlar

D-033'te kayıtlı (**KABUL**, 2026-10-09, 12. tur):
1. Etiket dengesizliği için **SMOTE kullanılmaz** (EXP-010'daki makale taklidi hariç). Klasik modellerde sınıf ağırlığı, derin modellerde ağırlıklı BCE.
2. Eşik **iç 5-fold OOF** tahminlerinden seçilir. D-032'deki "iç doğrulama kümesi" ifadesinin daha az gürültülü hâli: bütün eğitim katılımcıları kullanılır, yalnız ~9 kişi değil.
3. Kalibrasyon her deneyde raporlanır.
4. Kayıt bağlamı için eğitim-düzeyi dengeleme (SMOTE / ağırlık / Group DRO) birincil hatta yok. Değerlendirme aşamasında yalnız örtüşmede ve yalnız duyarlılık analizi olarak (D-026, D-028).
5. Faz 2'de backbone / görev seçimi önceden sabitlenir ya da iç içe yapılır; seçilmeyenler de raporlanır.
