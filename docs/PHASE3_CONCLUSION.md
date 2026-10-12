# Faz 3 (EXP-030) ve proje — sonuç bildirimi ve yol haritası önerisi

**Durum:** taslak, 2026-10-12. Kaynak: `reports/phase3/EXP-030_finetune.md` (tanımlar PHASE3_DESIGN 5.2'de sonuçlardan önce), `reports/phase3/G1_frozen_daygrouped.md`, `reports/phase3/EXP-016DG_frozen.md`, `docs/EVAL016_CONCLUSION.md`. Bu belge **yeni analiz içermez**. Yol haritası (Bölüm 4) bir öneridir: D-044, ÖNERİLDİ.

---

## 1. Faz 3 sonuçları (gün-gruplu CV; 339 katılımcı, 282 astım / 57 sağlıklı)

| | cnn10 | cnn14 | ssl (12 beats / 2 wavlm_large / 1 wavlm_base_plus) | cnn10_scratch |
|---|---|---|---|---|
| E1 fine-tune (fold AUC ort ± SD) | 0.867 ± 0.063 | 0.865 ± 0.078 | 0.893 ± 0.077 | 0.686 ± 0.096 |
| E1 dondurulmuş eş | 0.823 ± 0.093 | 0.767 ± 0.084 | 0.882 ± 0.062 | — |
| ΔE1 fold-eşli (NB p; FT daha iyi fold) | +0.044 (0.30; 11/15) | +0.098 (0.22; 12/15) | +0.010 (0.73; 7/15) | — |
| T1 PLR p · OR/SD [boot %95] (FT) | <0.001 · 3.07 [1.80, 11.7] | <0.001 · 2.53 [1.48, 10.6] | <0.001 · 3.52 [2.13, 14.0] | 0.24 · 1.42 |
| Gün-kümeli CV3 p (FT; betimsel) | 0.083 | 0.072 | 0.050 | 0.31 |
| C3: LR farkı [%95] | +4.5 [−5.8, +17.6] **belirsiz** | +4.9 [−6.2, +24.2] **belirsiz** | +2.7 [−6.3, +12.9] **belirsiz** | — |
| C2 kısmi R² p (FT · dondurulmuş) | 0.52 · 0.44 | 0.40 · 0.12 | **0.024** · 0.13 | **0.003** |
| Sağlıklılarda sabah − öğleden sonra skoru (FT) [%95] | +0.18 [−0.24, +0.60] | **+0.65 [+0.10, +1.22]** | **+0.78 [+0.28, +1.29]** | −0.09 |
| O1 kalibrasyon eğimi (FT · dondurulmuş) | 1.03 · 2.17 | **0.56** · 2.83 | **0.61** · 1.95 | 0.77 |
| Ort. abs(ŝ) (FT · dondurulmuş) | 2.5 · 0.8 | 6.2 · 0.5 | 5.5 · 1.4 | 1.1 |

Referans çizgisi (D-028): **yalnız bağlam** (saat + tarih), aynı gün-gruplu fold'larda **0.942 ± 0.038**. MFCC 0.72–0.73.

### 1.1 Söyledikleri

1. **Fine-tune skorları da bağlamın ötesinde etiketle ilişkili** (T1 p < 0.001, OR/SD 2.5–3.5). Bu, gün-gruplu CV'de, yani eğitimde gün izi kapalıyken geçerli. Dondurulmuş bulgu fine-tune ailesinde de tekrarlanıyor [FACT].
2. **Fine-tuning'in bağlamdan bağımsız ek bilgi kattığı gösterilemedi (C3 üç kolda belirsiz).** Fark işaretleri pozitif ama aralıklar geniş ve 0'ı içeriyor [FACT].
3. **E1 artışları ölçülemeyecek kadar belirsiz.** PANNs kollarında yön pozitif (cnn14 +0.10, cnn10 +0.04); SSL'de pratikte sıfır. Hiçbiri fold-eşli testte ayırt edilemiyor [FACT].
4. **Aşırı uyum var, ama "sıralama bozuluyor" türünden değil.** SSL ve CNN14 fine-tune skorları aşırı güvenli: kalibrasyon eğimi 0.56–0.61, skorlar dondurulmuşa göre 4–10 kat daha uç. Buna karşın sıralama (E1) dondurulmuştan kötü değil. CNN10 fine-tune iyi kalibre (eğim 1.03). Klinik bir kullanımda olasılıklar yeniden kalibre edilmeden kullanılamaz [INFERENCE].
5. **Fine-tuning bağlam yoluna kayma işaretleri gösteriyor (SSL ve CNN14).**
   - Sağlıklı sabah kayıtları (n = 12) daha "astımlı" skorlanıyor. Bu kohortta astım kayıtlarının çoğu sabah, sağlıklıların çoğu öğleden sonra.
   - SSL fine-tune skoru, etiket içinde bağlamla ilişkili (kısmi test p = 0.024); dondurulmuş eşinde değil.
   - En çok aşırı uyan iki kol, bağlam işaretinin en güçlü olduğu iki kol. Ortak bir açıklama [HYPOTHESIS]: daha esnek model, eğitim verisini ayıran her şeyi, kayıt koşullarının akustiğini de öğreniyor.
   - Dikkat: n = 12, birkaç lens, çoklu karşılaştırma düzeltmesi yok → güçlü bir işaret, kanıt değil.
6. **Ön-eğitim önemli (RQ2):** cnn10 − sıfırdan ΔAUC +0.18 (p 0.025; 14/15 fold). Ama sıfırdan model bu tarif ve bütçeyle eğitim verisini bile zar zor öğrendi (eğitim kaybı 0.22, sabit tahminci 0.23). Sıfırdan kolun T1'i anlamsız, C2'si pozitif (p = 0.003): öğrendiği az şey bağlama yakın. Doğru cümle: "bu veri boyutunda ön-eğitimsiz model sesten bağlamın ötesinde bilgi öğrenemedi."

### 1.2 Söylemedikleri

- "Fine-tuning daha iyi" demiyor (C3 ve ΔE1 belirsiz).
- "Fine-tuning kötü" de demiyor: sıralama korunuyor, CNN10'da kalibrasyon iyi.
- Klinik ayırt edicilik hakkında bir şey söylemiyor: bağlam tek başına her ses modelinden iyi ayırıyor.
- Kollar arası sıralama yapmıyor: ssl > cnn10 ≈ cnn14 farkları test edilmedi ve belirsizlik içinde.

### 1.3 Olumsuz ve sapma kayıtları (korunur)

- CAL-001: kayıp tabanlı epoch kuralı 4 / 5 kolda 1. epoch'u seçti. D-043 ile post hoc değişti, dış eğitimden önce.
- DIAG-001: CNN14'te fp16 değerlendirme taşması (H2) → değerlendirme fp32.
- Gün-gruplu split ilk denemede sklearn sürümüne bağlı dengesizlik verdi; kendi gruplu atamamızla düzeltildi.
- O1 ve ΔE1 post hoc betimsel okumalardır.

---

## 2. Büyük resim — başından beri ne yapmaya çalıştık, nereye vardık?

**Başlangıç hedefi:** veri ekibinin MFCC + klasik ML astım / sağlıklı ses çalışmasını, önceden eğitilmiş ses modelleri ve uçtan uca fine-tuning ile geçmek (Boll ve ark. yöntemi).

**Yolda değişen soru:**
- Faz 0–2'de, bu kohortta kayıt bağlamının (saat, tarih / dönem, yaş) etiketi neredeyse belirlediği ortaya çıktı. Yalnız bağlamla AUC 0.92–0.94.
  - Sağlıklıların çoğu öğleden sonra ve erken dönemde kaydedilmiş; geç dönemde hiç sağlıklı yok.
- Bu yüzden "hangi model daha yüksek AUC veriyor?" sorusu bilimsel olarak cevapsız kalır. Bir ses modeli, ses yerine kayıt koşullarının izini öğrenerek de yüksek AUC alabilir.
- Soru şuna dönüştü: **ses, ölçülen bağlamın ötesinde ne kadar bilgi taşıyor?**

**Vardığımız sonuçlar (en sağlamdan en zayıfa):**

1. **Bu veri setinde AUC, sesle astım tanımanın ölçüsü değildir [FACT].** Bağlam tek başına her modelden iyi ayırıyor. Bu, projenin en sağlam ve en genellenebilir bulgusu. Rastgele bölünmüş CV ile bu veride raporlanan herhangi bir AUC, kayıt koşullarını da ölçüyor olabilir [INFERENCE].
   - Bu, veri ekibinin yayımlanmış MFCC çalışmasının yorumu için de geçerli olabilir. Onların split ve bağlam kontrolü kontrol edilmeli [NEEDS VERIFICATION].
2. **Önceden eğitilmiş derin temsiller, ölçülen bağlamın ötesinde etiketle ilişkili bir sinyal taşıyor [FACT, sınırlarıyla].**
   - Dondurulmuş ve fine-tune, katılımcı-gruplu ve gün-gruplu CV, yaş / cinsiyet / sigara ayarlamaları altında tekrarlanıyor.
   - Etki SD başına OR ≈ 2.5–3.5.
   - Ama: gün kümelenmesine dayanıklı çıkarımda sınırda (CV3 0.05–0.08); bağlama eklenince ayırt edicilikteki artış ölçülemiyor; ölçülmemiş yollar (ortam / arka plan, prosedür sırası G3, cihaz) dışlanmadı.
   - MFCC'de bu sinyal daha zayıf (T1 p 0.04–0.05, CV3 > 0.27). Bu, "derin temsiller MFCC'den fazla bilgi taşıyor" yönünde bir **işaret**; backbone karşılaştırması tasarlanmadığı için iddia değil.
3. **Uçtan uca fine-tuning bu veri boyutunda ölçülebilir bir kazanç getirmedi [FACT, belirsizlik geniş].** Daha esnek modellerde aşırı güven ve bağlama kayma işaretleri var [HYPOTHESIS düzeyinde güçlü işaret].
4. **Ön-eğitim gerekli** [FACT, bu tarif ve bütçeyle].

**Eleştirel değerlendirme:**
- **İyi yapılanlar:**
  - Sızıntı ve karıştırıcı kontrolü baştan kuruldu.
  - Analizler sonuçlardan önce yazıldı.
  - Olumsuz sonuçlar ve sapmalar korundu.
  - Sentetik "doğrusu bilinen" testler, gün-gruplu kontrolün gerçekten işe yaradığını gösterdi (F11: katılımcı-gruplu CV'de yanlış pozitif p = 3.6e-9, gün-gruplu CV'de p = 0.56).
  - Bunlar olmasaydı proje büyük olasılıkla "derin model MFCC'yi 0.89'a karşı 0.72 ile geçti" diye yanlış bir sonuca varırdı.
- **Bedeli / zayıf yanları:**
  - **Dar boğaz model değil, veri.** 57 sağlıklı, tek merkez, etiketle iç içe kayıt koşulları. Bu yapıda hiçbir model karmaşıklığı "ses mi, koşul mu?" sorusunu kesin cevaplayamaz. Faz 3 (≈ 7 T4 saati: smoke test, kalibrasyon, eğitim) buna rağmen değerliydi: fine-tuning'in kurtarıcı olmadığını ve yeni bir risk (bağlama kayma) getirdiğini gösterdi. Ama yeni bilgi miktarı sınırlıydı ve bu önceden kısmen öngörülebilirdi.
  - Karşılaştırmaların çoğu düşük güçlü (C3 aralıkları ±10–20 LR birimi). Daha fazla model / tarif denemek bu belirsizliği azaltmaz.
  - C1 (sinyal bağlamdan bağımsız) kurulamadı ve bu veriyle kurulamaz. G2 (ortam), G3 (prosedür) açık.
  - Süreç ağırlaştı: kapılar, belgeler ve harness'ler güvenilirlik sağladı ama ilerlemeyi yavaşlattı. Bundan sonrası için "sonucu değiştirebilecek mi?" filtresi daha sıkı uygulanmalı.

---

## 3. Bu veriyle yapılmaması önerilenler

- Daha fazla model / tarif araması (LP-FT, augmentation, daha uzun eğitim, ensemble): C3'ün belirsizliği veri boyutundan geliyor [INFERENCE].
- Kollar arası "en iyi model" iddiası.
- Klinik ayırt edicilik iddiası.

---

## 4. Yol haritası önerisi (D-044, ÖNERİLDİ)

**Aşama A — Kapanış ve yazım (öncelik; GPU yok; ~1–2 hafta):**
1. Bu belgenin kesinleşmesi; EVAL016_CONCLUSION ile birleşik tek bir proje raporu: yöntem, bulgular, olumsuzlar, sınırlılıklar, yeniden üretim kılavuzu.
2. Ana mesajın yazıya dökülmesi: **"Kayıt bağlamı, ses tabanlı astım sınıflandırmasında başarıyı nasıl şişirir — önceden eğitilmiş ses modelleriyle bir vaka çalışması."**
   - Katkılar: bağlam-referans çizgisi, T1 çerçevesi, gün-gruplu CV gerekliliği (sentetik kanıtla), fine-tuning'in bağlama kayma riski.
   - Biçim (makale / çalıştay / tez bölümü) Alper'in hedefine göre seçilir.
3. Veri ekibine geri bildirim: bağlam dengesizliği bulgusu ve yayımlanmış MFCC çalışmasının split / bağlam kontrolü sorusu. G3 (prosedür sırası) sorusu yinelenir.

**Aşama B — Veri düzeyinde çözüm (en yüksek bilimsel değer; veri ekibine bağlı):**
- Bağlamı dengeli yeni kayıt: sağlıklıların sabah ve geç dönemde, astımlıların öğleden sonra kaydı; aynı cihaz / oda.
- Ya da katılımcı içi tasarım: aynı astım hastasının farklı durumlarda kaydı (ör. bronkodilatör öncesi / sonrası, atak / stabil). Bu, gruplar arası karıştırıcıyı tasarımla ortadan kaldırır.
- Mevcut hattımız (split, T1, gün-gruplu CV, değerlendirme) yeni veriye doğrudan uygulanabilir. Modeller yeniden eğitilmeden dış doğrulama yapılabilir.

**Aşama C — Bu veriyle, daha az karışık bir soru (isteğe bağlı; önce fizibilite):**
- Yalnız astım hastaları içinde klinik değişkenlerle ilişki (ör. spirometri / astım kontrolü, `clinical_data.csv`'de varsa). Etiket-bağlam iç içeliği burada yok; ama bağlam bu değişkenlerle de ilişkili olabilir, önce betimsel kontrol gerekir [NEEDS VERIFICATION].
- Yeni bir araştırma sorusu sayılır: ayrı tasarım belgesi ve kararla açılır.

**Tez bağlantısı (bilgi notu):** "Önceden eğitilmiş ses temsilleri kayıt ortamını / koşullarını ne kadar kodluyor?" sorusu, bu projenin yan ürünü. Aynı zamanda akustik ortam temsili üzerine bir tez konusuyla doğrudan örtüşüyor. Aşama A'daki bulgular orada motivasyon olarak kullanılabilir.
