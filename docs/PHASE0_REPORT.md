# Faz 0 Raporu — Sesten Astım Sınıflandırması

| | |
|---|---|
| Tarih | 2026-10-08 |
| Veri | `clinical_data.csv` (sha256 `cab18a35…a3af9b`), XLSX ile hücre hücre aynı (Data Report 2026-03-14) |
| Ses verisi | Veri ekibi (düzeltilmiş bildirim): iPhone 14, ağızdan 10 cm, hep aynı yer, mono, **48 kHz**, `.m4a`, adlandırma `<ID>_<slot>.m4a`, son dosya `101344_7.m4a`. **Denetlendi (Colab, 2026-10-08):** 2 393 dosya, 342 katılımcı (283 astım / 59 sağlıklı); sonuçlar Bölüm 2.3 ve 6.7. |
| Durum | Faz 0 tamam (literatür, klinik ve ses denetimi, eşleme, girdi sözleşmeleri, protokol, kayıt bağlamı analizi). Faz 1 sürüyor: split dosyaları → ses önbelleği → MFCC baseline |
| Son güncelleme | 2026-10-08 (8. tur): split'ler Colab'da teyit edildi; ses önbelleği kodu ve testi (D-030). Önceki (7. tur): confounder ayrı araştırma başlığı, değerlendirme model geliştirme sonuna (D-028); D-015 ve D-016 kabul; Faz 1 başladı (split dosyaları). Önceki (6. tur): confounder kontrol yöntemlerinin değerlendirmesi ve deney tasarımı (`docs/CONFOUND_CONTROL_DESIGN.md`, D-025/026/027 önerildi); EXP-004/004b, tasarım referans çizgileri, SIM-001. Önceki (5. tur): kayıt bağlamı analizi (EXP-003) — günün saati en güçlü confounder; birincil test revize edildi (D-023); kesim 11.0 kHz; dosya tarihi önceliği (D-024). Önceki (4. tur): ses denetimi sonuçları, günün saati confounder'ı, iki kodlama zinciri, harmonizasyon kararı (D-021, D-022). Önceki (3. tur): eşleme düzeltildi (2 = araba, 3 = ana), 48 kHz ve `.m4a` teyit edildi; gün düzeyi kanıt, tarih alanının geçerliliği, EXP-002 ve negatif kontrol tasarımı (D-020) eklendi |

Etiketler: **[FACT]** veri/literatürle doğrudan destekli · **[FROM PAPER]** belirli makaleden · **[FROM OFFICIAL DOCS]** resmi kod/doküman · **[INFERENCE]** çıkarım · **[HYPOTHESIS]** test edilmemiş · **[DECISION]** bilinçli karar · **[NEEDS VERIFICATION]** kod/deneyden önce doğrulanmalı

---

## 0. Beş dakikada özet

1. **Zamansal confounder (en kritik bulgu).** Sağlıklı gönüllülerin tamamı 9 Ocak – 2 Nisan 2024 arasında kaydedilmiş; 284 hastanın **184'ü** bu tarihten *sonra* (Mayıs–Temmuz) kaydedilmiş. Sesi hiç kullanmadan, **yalnızca kayıt tarihi** etiketi AUC **0.797 ± 0.048** ile tahmin ediyor — yayınlanmış en iyi MFCC modelinden (AUC 0.769) yüksek. ID numaraları kayıt sırasıyla verildiği için (ρ = 0.985) **yalnızca katılımcı ID'si** de AUC 0.800 veriyor. [FACT] Bu, yayınlanmış modelin tarihi kullandığını *kanıtlamaz*; ama kayıt koşulları aylar içinde değiştiyse (oda, mevsimsel gürültü, klima, iOS güncellemesi, operatör) bir ses modelinin bu bilgiyi kısayol olarak öğrenmesi mümkündür. [INFERENCE] Zaman-örtüşen alt kohortta (99 hasta / 57 sağlıklı) tarihin tahmin gücü AUC 0.567 ± 0.116'ya düşüyor. [FACT] **Strateji (Bölüm 6, D-017):** ölç → tasarımla kontrol et (tam kohort / zaman penceresi / aynı gün) → negatif kontrollerle sına → gerekirse azalt → önceden yazılmış kurala göre yorumla.
2. **Yaş confounder'ı.** Sağlıklılar daha genç (medyan 40 vs 46, Mann–Whitney p = 1.5×10⁻⁵). **Yalnızca yaş** AUC 0.681 ± 0.077 veriyor ve zaman-örtüşen alt kohortta da sürüyor (0.674). [FACT] Referans makaledeki "gençlerde başarı çok düşük, yaşlılarda çok yüksek" deseni, tam da bir yaş kısayolunun üreteceği desendir. [INFERENCE]
3. **"%82 doğruluk" çoğunluk sınıfı seviyesinde.** 284/344 = **0.826**. Yayınlanmış en iyi model (gelecek, Voting) doğruluk 0.820, duyarlılık 0.933, özgüllük 0.283 → **dengeli doğruluk ≈ 0.61**. [FACT] Bu yüzden doğruluk tek başına raporlanmayacak. [DECISION]
4. **"STFT mi waveform mu?" pretrained modeller için serbest bir seçim değil.** PANNs, BEATs ve WavLM'nin üçü de API'de **ham dalga formu** alır ve temsili *kendi içinde* üretir (PANNs: STFT→64 bantlı log-Mel; BEATs: 128 bantlı Kaldi fbank; WavLM: öğrenilmiş CNN). Bizim işimiz doğru sampling rate'i ve modelin kendi frontend'ini kullanmak; kendi STFT'mizi hesaplayıp vermek ön-eğitimli ağırlıklarla uyumsuzluk yaratır. [FROM OFFICIAL DOCS] Doğrulanmış tuzaklar: CNN10 yalnızca **32 kHz** sürümüyle var; WavLM **Base+ normalizasyon yapmaz, Large yapar**; PANNs modeli `train()` modunda **SpecAugment'i kendiliğinden uygular**.
5. **Eşleme kuralı veri ekibinden geldi.** Dosyalar `<ID>_<slot>.m4a`; 1 = aaa, 2 = araba, 3 = ana, 4 = ordu, 5 = gelecek, 6 = titiz, 7 = ünlem (düzeltilmiş bildirim). Son dosya 101344_7 → 101345–101348'in sesi yok, birincil kohort 344 (D-002). Ses denetimi ve kısa bir dinleme teyidi bitmeden model eğitimine geçilmeyecek. [FROM DATA TEAM + DECISION]
6. **Ses denetimi (4. tur).** 2 393 dosya; hepsi `.m4a` AAC-LC mono, biri dışında 48 kHz; kopya yok, konum etiketi yok. **Tarih dengesizliği dosya zaman damgalarıyla doğrulandı.** **Yeni confounder: günün saati.** Hastalar sağlıklılardan belirgin biçimde daha erken saatte kaydedilmiş (AUC 0.13–0.17). Dosyaların %27'si ayrı bir FFmpeg zincirinden geçmiş. Ses işleme yalnız teknik farkları giderebilir; tarih, saat ve hasta profili sesin kendisinde (Bölüm 6.8, D-021). [FACT]
7. **Kayıt bağlamı (5. tur, EXP-003).** Sesi hiç kullanmadan **tarih + saat etiketi AUC 0.931 ± 0.032 ile ele veriyor**; saat tek başına 0.841. Sağlıklılar çoğunlukla öğleden sonra (medyan 15:10), hastalar sabah (medyan 11:05) kaydedilmiş. Zaman penceresi saati kontrol etmiyor (saat orada 0.873). Bu yüzden birincil soru, ses modeli sonuçları görülmeden önce değiştirildi: **"ses, kayıt bağlamı ve yaşın ötesinde astım bilgisi taşıyor mu?"** (D-023, Bölüm 6.9). [FACT + DECISION]
8. **Confounder kontrol yöntemleri (6. tur, Bölüm 6.10).** Yedi yöntem değerlendirildi (`docs/CONFOUND_CONTROL_DESIGN.md`). Kanıtı yalnız değerlendirme katmanı üretir: artımlı test T1, ayarlı E2h, Spisak testleri ve negatif kontroller. Harmonizasyon ve katılımcı split'i hijyendir, confounding kontrolü değildir. Marjinal adversarial eğitim burada astım sinyalini de siler. Üç yeni bulgu test spesifikasyonunda değişiklik önerdi (D-025; değerlendirme aşamasında kesinleşecek, D-028):
   - Ham E2h AUC'sinin şans çizgisi 0.5 değil: yalnız yaşı kodlayan bir model **0.70** alır.
   - Kuadratik bağlam modeliyle T1, ses yalnız bağlamı kodlarken bile **%22** yanlış pozitif verebilir; spline modelle %6.
   - Gerçek sinyal varken bile ΔAUC binde birler düzeyinde kalıyor.
   
   [FACT + DECISION]
9. **Confounder ayrı bir araştırma başlığı (7. tur, D-028).** Ana plan (D-012) değişmeden sürüyor. Kayıt bağlamı değerlendirmesi model geliştirme bitince, kaydedilmiş tahminler üzerinde yapılacak. O zamana kadar her ses modeli AUC'si **üst sınır** olarak, "yalnız bağlam 0.93" ve "yalnız yaş 0.68" referanslarıyla birlikte raporlanır. [DECISION]

---

## 1. Literatür denetimi

### 1.1 Yan yana karşılaştırma

| Konu | Alagöz ve ark. (BMC Pulm Med 2026, baskıda) | Boll ve ark. (PROPOR 2026) |
|---|---|---|
| Veri seti | **Bizim veri setimiz** (Yedikule, Ocak–Temmuz 2024) [FROM PAPER] | Brezilya Portekizcesi, mobil cihaz; hastane + kontrolsüz ev ortamı [FROM PAPER] |
| Katılımcı | 344: 284 astım, 60 sağlıklı | Konuşma 549 / ünlü 538 kayıt (tablolarda "unique patients"), %79 astım |
| Kayıt protokolü | iPhone 14 Pro, ağızdan 10 cm, sessiz oda, 48 kHz; /a/ 10 s + 6 kelime × 10 tekrar | Kısa sabit cümle okuma + /a/ "olabildiğince uzun"; 16 kHz; ort. 7.5 s; cihazlar farklı |
| Ön işleme | 22.05 kHz'e resample; enerji tabanlı VAD ile sessizlik kırpma | Tepeye göre enerji tabanlı kırpma, 16 kHz, tepe normalizasyonu |
| Özellik | 12 MFCC + Δ + ΔΔ (çerçeve 2048, hop 512, Hamming) = 36 + ZCR/centroid/bandwidth/rolloff ort.±SD = 8 → **kayıt başına 44 boyutlu vektör** | Her pencere için log-Mel spektrogram (parametreler **raporlanmamış**) |
| Segmentasyon | Yok (kayıt başına tek vektör) | 4.0 s pencere, 2.0 s hop, kısa kayıtlar sıfırla doldurulur; **split'ten sonra** |
| Split | StratifiedKFold 5, katılımcı düzeyinde, tek tohum (42), tekrar yok, ayrı validation yok | Sabit 60/20/20, katılımcı düzeyinde; **yaş grubu ve cinsiyete göre** tabakalı |
| Sızıntı önlemi | Fold başına train/test katılımcı kesişimi = 0 doğrulanmış; StandardScaler ve SMOTE yalnız train'de fit | Pencereleme split'ten sonra; oversampling yalnız train'de; gürültü enjeksiyonu |
| Dengesizlik | SMOTE (yalnız train) | Azınlığı rastgele çoğaltma (yalnız train) |
| Augmentation | Yok | Çoğaltılan örneklerde biri: Gauss gürültü / kazanç / pitch shift / time stretch; + her pencereye hastane gürültü havuzundan karışım |
| Model | NB, kNN, DT, LR, SVM, sığ MLP, RF, AdaBoost, GB, CatBoost, XGBoost + Voting / Stacking / Weighted Voting | PANNs **CNN10, CNN14** (AudioSet), **tüm katmanlar fine-tune**, MLP başlık (2 lineer + ReLU + dropout 0.3); sıfırdan CNN10 |
| Hiperparametre | Model bazında raporlanmamış; karar eşiği 0.5 | Adam, lr 1e-4, weight decay 1e-4, batch 16, ≤50 epoch, early stopping: val dengeli doğruluk, patience 10 |
| Değerlendirme | Fonetik birim başına; acc, F1, sens, spec, AUC (5 fold ort.±SD); her birim için "en iyi model" = en yüksek ort. AUC | Segment ve hasta düzeyi; acc, dengeli acc, sens, spec, MCC, ROC-AUC, PR-AUC; 10 tohum ort.±SD; bootstrap CI; McNemar |
| Hasta düzeyi birleştirme | Gerekmiyor (her birimde kişi başı 1 kayıt); birimler arası füzyon yok | Segment logit'lerinin ortalaması → argmax |
| Ana sonuç | En iyi AUC 0.769 ± 0.046 (gelecek, Voting), acc 0.820, sens 0.933, spec 0.283 | CNN14 + konuşma: hasta düzeyi dengeli acc 0.85 ± 0.03, AUC 0.93 ± 0.01; CNN10 benzer; sıfırdan ve ünlü daha kötü |
| Yazarların kısıt beyanı | Dış doğrulama yok; yaş dengesizliği; ayarlama yapılmamış | Küçük ve dengesiz veri; yalnız ikili sınıflama |

### 1.2 Alagöz ve ark. — eleştirel değerlendirme

| # | Gözlem | Etiket | Bizim için sonucu |
|---|---|---|---|
| A1 | "%82 doğruluk" ≈ çoğunluk sınıfı (0.826); en iyi modelin dengeli doğruluğu ≈ 0.61 | [FACT] (tablolardan hesap) | Birincil metrik AUC + dengeli doğruluk |
| A2 | Her birim için ~14 sınıflandırıcı arasından **aynı CV sonuçlarına bakarak** "en iyi" seçilmiş; nested CV yok → iyimser seçim yanlılığı (yazarların kendi atıf yaptığı Varma & Simon 2006'nın uyardığı durum) | [INFERENCE] | Model seçimi iç döngüde ya da tüm adaylar raporlanacak |
| A3 | İki ayrı tablo aynı başlıkla farklı "en iyi" model ve sayılar veriyor (ör. gelecek AUC 0.717 vs 0.769; aaa 0.649 vs 0.700); metin araba için 0.656'yı "en düşük" diyor ama diğer tabloda araba 0.704 | [FACT] | Yeniden üretim hedefi belirsiz → aralık olarak hedefleyeceğiz (AUC ≈ 0.65–0.77) |
| A4 | Δ ve ΔΔ'nin zaman ortalaması neredeyse sıfırdır; 36 boyutun bir kısmı pratikte bilgi taşımıyor olabilir. Özetin yalnız ortalama mı olduğu yazılmamış | [INFERENCE] + [NEEDS VERIFICATION] | Yeniden üretimde varsayım açıkça kaydedilecek |
| A5 | "Enerji tabanlı VAD" deniyor ama atıf istatistiksel-model tabanlı VAD (Sohn 1999); eşik verilmemiş | [FACT] | Yeniden üretimde parametre seçimi belgelenecek |
| A6 | Weighted Voting ağırlıkları "validation-set AUC"ye orantılı; hangi validation setinin kullanıldığı yazılmamış → iç sızıntı ihtimali | [NEEDS VERIFICATION] | Bu ensemble'ı yeniden üretimde ayrı işaretleyeceğiz |
| A7 | Tek tohum, tek 5-fold bölünmesi → split varyansı ölçülmemiş | [FACT] | Tekrarlı CV |
| A8 | Yaş farkı kabul edilmiş, ayarlanmamış | [FROM PAPER] | Yaş-only baseline zorunlu karşılaştırıcı |
| A9 | **Kayıt tarihi dengesizliği makalede hiç geçmiyor** (bizim bulgumuz) | [FACT] | Zaman-kontrollü analiz zorunlu |
| A10 | Sağlıklılarda SFT yok → "sağlıklı" etiketi spirometriyle doğrulanmamış | [FACT] (CSV'de sağlıklılarda SFT alanları %0 dolu) | Etiket gürültüsü ihtimali kısıtlar arasında yazılacak |
| A11 | Kontrol durumu sayıları makalede 47/237, CSV'de 41/242 (+1 eksik) | [FACT] | İkincil analizlerde CSV kullanılacak, fark not edilecek |

### 1.3 Boll ve ark. — eleştirel değerlendirme

| # | Gözlem | Etiket | Bizim için sonucu |
|---|---|---|---|
| B1 | Tek sabit 60/20/20 bölünme; 10 tohum yalnızca başlatmayı/augmentation'ı değiştiriyor → raporlanan SD **split varyansını içermiyor** | [FROM PAPER] + [INFERENCE] | Bizde 60 sağlıklı var: %20 test = 12 kişi. Tek split yerine tekrarlı CV |
| B2 | Tabakalama yaş grubu ve cinsiyete göre; etikete göre tabakalama açıkça yazılmamış | [FROM PAPER] | Bizde etiket × yaş grubu tabakalaması |
| B3 | McNemar p-değerleri 10 tohum üzerinden Fisher yöntemiyle birleştirilmiş; tohumlar **aynı test hastalarını** paylaştığı için testler bağımsız değil → p-değerleri fazla iyimser | [INFERENCE] | Düzeltilmiş tekrarlı-CV t-testi + DeLong |
| B4 | 16 kHz'de işlediklerini söylüyorlar ama **CNN10'un resmi 16 kHz ağırlığı yok** (yalnız 32 kHz). Ya 32 kHz'e yükseltip vermişler ya da frontend uyumsuz çalışmış; log-Mel parametreleri raporlanmamış | [FROM OFFICIAL DOCS] + [NEEDS VERIFICATION] (bilinemez) | Biz her modeli resmi SR'si ile kullanacağız |
| B5 | Pitch shift ve time stretch, hastalığa dair olabilecek ipuçlarını (perde, konuşma hızı) bozabilir; ablasyonu yapılmamış | [HYPOTHESIS] | Augmentation başlangıçta kapalı, sonra ablasyon |
| B6 | Alt grup tablosundaki N'ler (35–37) ana testteki N ile (107–111) uyuşmuyor | [FACT] | Kaynak olarak alt grup sayılarına temkinli yaklaşılacak |
| B7 | Konuşmada ≤30 yaş doğruluğu 0.33, >60 yaş 0.99 — sınıflara göre yaş dağılımı verilmemiş; yaş kısayoluyla tutarlı | [INFERENCE] | Yaş kontrollü değerlendirme |
| B8 | PR-AUC 0.98'in taban çizgisi %79 prevalanstır; azınlık (astımsız) sınıf için PR-AUC yok | [INFERENCE] | Her iki sınıf için PR-AUC |
| B9 | Ortam (hastane/ev) ve cihazın sınıflara göre dağılımı raporlanmamış | [FACT] | Bizde cihaz sabit (iPhone 14 Pro) ama zaman değişkeni var |

### 1.4 Boll ve ark.'ndan ne alıyoruz, neyi değiştiriyoruz?

**Alıyoruz:** split'i katılımcı düzeyinde yapıp pencerelemeyi **sonra** yapmak; 4 s / 2 s pencere; dengelemeyi yalnız train'de yapmak; segment + hasta düzeyi değerlendirme; logit ortalamasıyla birleştirme; tam fine-tune + 2 katmanlı MLP başlık; hiperparametreleri **önceden sabit** varsayılanlar olarak (lr 1e-4, wd 1e-4, batch 16, ≤50 epoch, patience 10); çoklu tohum; bootstrap CI; sıfırdan eğitilmiş karşılaştırma modeli; alt grup analizi.

**Değiştiriyoruz (gerekçeler DECISIONS.md'de):** tek split → tekrarlı katılımcı-düzeyi CV (D-009); her model kendi resmi SR'si (D-008); augmentation başlangıçta kapalı (D-016); McNemar+Fisher → eşleştirilmiş fold farkları + düzeltilmiş t-test + DeLong (D-011); zorunlu zaman ve yaş kontrolleri (D-004, D-005).

**Veri boyutu karşılaştırması:** Boll'da ~115 astımsız katılımcı var; bizde 60. Azınlık sınıfı yaklaşık yarısı kadar. Tüm 344 katılımcının havuzlanmış out-of-fold tahminlerinde bile AUC = 0.80 için beklenen %95 CI yarı genişliği **±0.052**; zaman-örtüşen alt kohortta **±0.068**; tek bir test fold'unda (≈57/12) **±0.117** (Hanley–McNeil yaklaşımı). [INFERENCE] Yani 0.02–0.03'lük model farkları tek başına anlamlı sayılamaz.

---

## 2. Veri denetimi

### 2.1 Klinik veri (tamamlandı — `scripts/audit_clinical.py`)

| Soru | Sonuç | Etiket |
|---|---|---|
| CSV ve XLSX aynı mı? | Evet: 348 × 140, aynı sütunlar, aynı ID'ler, **0 farklı hücre** (XLSX'te 3 satırlık başlık var) | [FACT] |
| Katılımcı sayısı | 348 (ID 101001–101348, hepsi tekil): **284 astım, 64 sağlıklı** | [FACT] |
| Makaledeki 344 nereden? | ID ≤ 101344 alınınca tam olarak **284 / 60**; hasta demografisi makale tablosuyla birebir (197 K / 87 E; sigara 150 hiç / 94 içiyor / 40 bırakmış; yaş medyan 46 (39–54)) | [FACT] → makale kohortu = 101001–101344 [INFERENCE, güçlü] |
| Fazla 4 kişi | 101345–101348: sağlıklı, **hiçbir demografik/tarih bilgisi yok** | [FACT] |
| Demografisi tamamen boş | 101149 + 101345–101348 (5 sağlıklı) | [FACT] |
| Tarih ve yaş eksik | 101018, 101019 (cinsiyet ve sigara var) | [FACT] |
| Tarama kriterleri boş | 9 sağlıklı: 101018, 101019, 101020, 101047, 101149, 101345–101348 | [FACT] → [NEEDS VERIFICATION] veri ekibiyle |
| 18–65 dışı hasta (protokol: 18–65) | 101290 | [FACT] |
| Her iki grupta da dolu değişkenler | **Yalnızca** yaş, doğum tarihi, cinsiyet, sigara (ve kayıt tarihi) | [FACT] |
| Yalnız hastalarda dolu | 47 değişken: GINA, AKT, SFT (FEV1, FVC…), IgE, eozinofil, ilaçlar, komorbiditeler… | [FACT] |
| Kayıt alanları | 7 sütunun her birinde **tek bir UUID** + `_k` soneki; k dağılımı slot1'de 0–6, diğerlerinde 0–2 | [FACT] |
| "Tekrar deneme" oranı | Herhangi bir slotta k>0: astım %48.9, sağlıklı %34.4 (Fisher p = 0.038) | [FACT]; k'nin anlamı [HYPOTHESIS] |
| Spirometri sütunları (FEV1%, FVC%, FEV1/FVC) | Değerlerin ~%98'i ondalık virgüllü (`91,8`); düz sayısal okuma bunları **sessizce boş** yapıyordu → düzeltildi. Düzeltilmiş hasta medyanları makaleyle birebir: FEV1% 78.15, FVC% 86.0, FEV1/FVC 75.89 | [FACT] — kohort eşleşmesinin bağımsız bir doğrulaması |

**Demografi (makale kohortu, 344):**

| | Astım (n=284) | Sağlıklı (n=60) |
|---|---|---|
| Yaş medyan (IQR) | 46 (39–54) | 40 (27–46), n=57 |
| Kadın / Erkek | 197 / 87 | 38 / 21 (+1 eksik) |
| Sigara: hiç / içiyor / bırakmış | 150 / 94 / 40 | 26 / 29 / 4 (+1 eksik) |
| Yaş grubu ≤30 / 31–45 / 46–60 / >60 | 36 / 98 / 123 / 27 | 20 / 20 / 16 / 1 |

**Ay × grup (tüm kohort):**

| Ay (2024) | Oca | Şub | Mar | Nis | May | Haz | Tem |
|---|---|---|---|---|---|---|---|
| Astım | 19 | 41 | 38 | 15 | 72 | 42 | 57 |
| Sağlıklı | 16 | 8 | 23 | 10 | 0 | 0 | 0 |

(7 katılımcının tarihi eksik.)

**Hastalarda alt grup değişkenleri (ikincil analizler için):** GINA kontrol: kontrol altında değil 242 / kontrol altında 41 / eksik 1 · Tedavi basamağı 1/2/3/4/5: 3/33/79/147/21 · SFT tanısı: normal 101 / kombine 95 / obstrüktif 86 / restriktif 2. Hastaların %36'sının spirometrisi normal — "spirometrisi normal astımlıları da yakalayabiliyor muyuz?" anlamlı bir alt grup sorusu. [INFERENCE]

### 2.2 EXP-001 — Sesi kullanmayan baseline'lar (tamamlandı)

Lojistik regresyon (sınıf ağırlıklı), katılımcı düzeyinde stratified 5-fold × 20 tekrar = 100 fold; hiperparametre yok, ölçekleme fold içinde.

| Kohort | Özellik | n (astım/sağlıklı) | AUC ort ± SD | Fold AUC %2.5–97.5 | Dengeli doğr. ± SD |
|---|---|---|---|---|---|
| 344 | yaş | 341 (284/57) | 0.681 ± 0.077 | 0.51–0.83 | 0.624 ± 0.070 |
| 344 | cinsiyet | 343 (284/59) | 0.516 ± 0.072 | 0.35–0.62 | 0.516 ± 0.072 |
| 344 | sigara | 343 (284/59) | 0.593 ± 0.066 | 0.45–0.69 | 0.580 ± 0.061 |
| 344 | yaş+cinsiyet+sigara | 341 (284/57) | 0.677 ± 0.083 | 0.51–0.82 | 0.634 ± 0.075 |
| 344 | **kayıt tarihi** | 341 (284/57) | **0.797 ± 0.048** | 0.71–0.88 | 0.734 ± 0.069 |
| 344 | **katılımcı ID numarası** | 344 (284/60) | **0.800 ± 0.053** | 0.70–0.90 | 0.694 ± 0.080 |
| zaman-örtüşen | yaş | 156 (99/57) | 0.674 ± 0.081 | 0.52–0.80 | 0.615 ± 0.080 |
| zaman-örtüşen | cinsiyet | 156 (99/57) | 0.489 ± 0.070 | 0.34–0.59 | 0.489 ± 0.070 |
| zaman-örtüşen | sigara | 156 (99/57) | 0.653 ± 0.089 | 0.46–0.79 | 0.605 ± 0.084 |
| zaman-örtüşen | yaş+cinsiyet+sigara | 156 (99/57) | 0.687 ± 0.090 | 0.51–0.84 | 0.652 ± 0.081 |
| zaman-örtüşen | kayıt tarihi | 156 (99/57) | 0.567 ± 0.116 | 0.34–0.79 | 0.552 ± 0.079 |
| zaman-örtüşen | katılımcı ID numarası | 156 (99/57) | 0.578 ± 0.116 | 0.35–0.81 | 0.570 ± 0.075 |

ID numarası kayıt sırasıyla artıyor (Spearman ρ = 0.985 ile tarih) — yani ID, tarihin vekili. [FACT]

**Yorum:** Zaman-örtüşen alt kohortta sigara daha güçlü bir ayırıcı (0.653; tam kohortta 0.593) — çünkü erken dönemde kaydedilen hastalarda aktif içici oranı düşük (%25; geç dönem hastalarında %37, sağlıklılarda %49). Alt kohort zamanı kontrol eder ama başka confounder'ları büyütebilir. [FACT/INFERENCE] Bir ses modelinin "işe yaradığını" söyleyebilmek için en az (a) yaş-only baseline'ı aşması, (b) zaman-örtüşen alt kohortta da çalışması ve (c) yaşın ötesinde bilgi taşıdığını göstermesi gerekir. [DECISION] Fold'lar arası AUC yayılımı (0.51–0.83) tek bir fold sonucuna neden güvenilemeyeceğini somut olarak gösteriyor. [FACT]

### 2.3 Ses verisi — denetim sonuçları (Colab, 2026-10-08)

| Kontrol | Sonuç | Etiket |
|---|---|---|
| Dosya sayısı | **2 393** (beklenen 344 × 7 = 2 408; makale de 2 408 diyor) | [FACT] |
| Sesi hiç olmayan | **101043** (astım), **101149** (sağlıklı, demografisi de yok), 101345–101348 (sağlıklı) | [FACT] |
| Eksik tek kayıt | **101244** (astım): slot 4 (ordu) yok | [FACT] |
| Sesli katılımcı | **342 = 283 astım / 59 sağlıklı**; 341'inde 7/7 kayıt | [FACT] |
| Format | Hepsi `.m4a`, AAC-LC, mono; 2 392 dosya 48 kHz, **1 dosya 44.1 kHz** (astım, geç dönem) | [FACT] |
| Bitrate | 141–196 kbps (medyan 184) | [FACT] |
| Etkin bant genişliği | İki tepeli: medyan 13.5 kHz, ≥%5'i 24 kHz, en düşük 11.3 kHz | [FACT] |
| Süre | Medyan 10.5 s (%1: 8.4 s, en kısa 4.4 s, en uzun 13.6 s) | [FACT] |
| Kodlama zinciri | 1 759 Apple (`iso5`, Core Media, zaman damgalı) / **634 FFmpeg** (`Lavf59.16.100`, zaman damgasız) | [FACT] |
| Dosya tarihi ↔ CSV tarihi | Apple dosyalarında **%98.7 aynı gün** (1 754 dosya); birkaç aykırı (−30 / +182 gün) | [FACT] |
| Kopyalar | Birebir kopya yok; yakın kopya yok (katılımcılar arası en yüksek parmak izi korelasyonu 0.51) | [FACT] |
| Gizlilik | Konum etiketi yok | [FACT] |
| Dinleme teyidi | 21 kayıt (her slottan 3) eşlemeyle uyumlu; 101042 ve 101050 farklı kişiler | [FROM USER] |

Makaledeki 2 408 kayıtla aradaki 15 dosyalık fark (101043 ve 101149'un 7'şer kaydı + 101244'ün 1 kaydı) veri ekibine sorulacak. Kayıt koşullarının etiket ve dönemle ilişkisinin yorumu Bölüm 6.7'de; ölçüm yöntemi `scripts/audit_audio.py`'nin başındaki açıklamada ve `tests/test_audit_audio.py`'de.

---

## 3. Hasta ↔ kayıt ↔ görev eşlemesi

**Kural (veri ekibi, düzeltilmiş bildirim) [FROM DATA TEAM]:** dosya adı `<katılımcı ID>_<slot>.m4a`. Eşleme dosya adından yapılır; CSV'deki `UUID_k` alanlarına gerek yoktur.

| Slot | Görev | Örnek dosya |
|---|---|---|
| 1 | aaa (sürdürülmüş /a/) | `101001_1.m4a` |
| 2 | araba | `101001_2.m4a` |
| 3 | ana | `101001_3.m4a` |
| 4 | ordu | `101001_4.m4a` |
| 5 | gelecek | `101001_5.m4a` |
| 6 | titiz | `101001_6.m4a` |
| 7 | ünlem | `101001_7.m4a` |

İlk bildirimdeki slot 2/3 çelişkisi veri ekibinin düzeltmesiyle giderildi. Her slottan 3 kayıt dinlendi ve eşlemeyle uyumlu bulundu → `configs/slot_task_map.yaml`: `confirmed_by_listening`. [FROM USER, D-019]

**CSV'deki `UUID_k` alanı ne olacak?** UUID = veri toplama sistemindeki form alanı; `k` = muhtemelen kabul edilen denemenin sırası. [HYPOTHESIS] Dışa aktarılan dosyanın hangi denemeye karşılık geldiği dosya adından anlaşılmıyor. `k > 0` oranı gruba göre farklı (astım %48.9, sağlıklı %34.4; p = 0.038) → yalnız ikincil analizde değişken; modele girdi değil.

**Doğrulama planı (eğitimden önce zorunlu):**
1. `audit_audio.py` → makale kohortunun 344 × 7 = 2 408 hücresinin hepsi OK olmalı ya da her istisna açıklanmalı; dosyası hiç olmayanlar listelenir (beklenti: 101345–101348).
2. Dinleme teyidi (her slottan 3 kayıt) → `configs/slot_task_map.yaml`.
3. Belirsizlik kalırsa veri ekibine sor (Bölüm 11).

---

## 4. Sızıntı (leakage) ve confounder denetimi

| Tür | Risk | Durum / Önlem |
|---|---|---|
| Katılımcı sızıntısı | Aynı kişinin kayıtları/pencereleri farklı split'lerde | Split **katılımcı tablosunda** yapılır, kayıtlar sonra eşlenir; her split dosyasında kesişim = 0 assert'i [DECISION] |
| Kimlik sızıntısı | Aynı kişi iki ID ile | 6 aynı doğum tarihi+cinsiyet grubu; 2'si 1 Ocak yer tutucusu. Şans eseri beklenen ≈7, gözlenen 4 → kanıt yok [INFERENCE]. Yalnız farklı etiketli çift (101042 astım / 101050 sağlıklı, 4 gün arayla) dinlenecek |
| Birebir/yakın kopya | Aynı dosya iki kişide | Ses denetimi (hash + parmak izi) |
| Ön işleme sızıntısı | Veri seti geneli istatistik (normalizasyon, imputation, feature selection) | Dosya-başı deterministik işlemler (resample, kırpma, tepe normalizasyonu) split'ten önce güvenli; **istatistik öğrenen her şey** (StandardScaler, PCA…) fold içinde fit |
| Augmentation sızıntısı | Split'ten önce augmentation; test'e augmentation | Augmentation yalnız train DataLoader'ında; gürültü havuzu kurulursa yalnız train fold'larının sessizliklerinden |
| Yeniden örnekleme sızıntısı | SMOTE / oversampling split'ten önce | Yalnız train fold'unda |
| Meta veri sızıntısı | Dosya adı, klasör, ID, kayıt sırası etiketi taşıyor mu | **ID numarası tek başına AUC 0.800** (ID↔tarih ρ = 0.985) [FACT]. Model girdisine asla ID/dosya yolu girmez; DataLoader her epoch karıştırır (ID sırasıyla batch oluşmaz); BatchNorm istatistikleri ID-sıralı veriyle hesaplanmaz [DECISION]. `_k` soneki gruba göre farklı → girdi değil |
| Eksiklik deseni | Hastaya özgü alanların sağlıklılarda boş olması | Bu alanlar astım-vs-sağlıklı modelinde **kullanılamaz** (etiketi birebir ele verir) [FACT] |
| Zamansal confounder | Kayıt dönemi ↔ etiket | **Bölüm 6 stratejisi (D-017)**: üç düzeyde değerlendirme (tam / zaman penceresi / aynı gün), negatif kontroller, önceden yazılmış karar kuralı |
| Yaş confounder'ı | Sağlıklılar daha genç | Yaş-only baseline; "yaşın ötesinde bilgi" testi; yaş gruplarına göre performans |
| Sigara | Sağlıklılarda daha çok aktif içici (%48 vs %33) | Alt grup analizi; yaş+cinsiyet+sigara baseline |
| Görev / süre | Görev zorluğu ve kayıt uzunluğu gruba göre değişebilir | Görev başına ayrı model; süre ve sessizlik confounder tablosunda |
| Deneme sayısı | k>0 oranı gruba göre farklı (p = 0.038) | Dışa aktarılan dosyanın hangi denemeye karşılık geldiği bilinmiyor; k yalnız ikincil analizde, modele girdi değil |
| Cihaz / ortam / kodlama | Veri ekibi: tek cihaz (iPhone 14), aynı yer, 10 cm | Uzantı, codec, bitrate, bant genişliği, iOS sürümü, cihaz modeli; her birinin etiketle ve hastalarda dönemle ilişkisi ses denetiminde [NEEDS VERIFICATION] |
| Gizlilik | iPhone dosyaları GPS konumu taşıyabilir | Denetim yalnız var/yok yazar; dinleme ve önbellek dosyalarında meta veri silinir; ham dosyalar paylaşılmaz |

---

## 5. Model girdi sözleşmeleri (resmi kaynaklardan doğrulandı)

| | PANNs CNN10 | PANNs CNN14 | PANNs CNN14_16k | BEATs | WavLM Base+ | WavLM Large |
|---|---|---|---|---|---|---|
| Ön-eğitim | AudioSet, gözetimli etiketleme | ← | ← | AudioSet, öz-gözetimli (iter3+ AS2M önerilir) | 94k saat İngilizce konuşma, öz-gözetimli | ← |
| API girdisi | waveform `(B, T)`, float | ← | ← | waveform `(B, T)` + padding mask | waveform `(B, T)` | ← |
| Sampling rate | **32 kHz** | **32 kHz** | 16 kHz | 16 kHz | 16 kHz | 16 kHz |
| İç frontend | STFT n_fft 1024, hop 320, Hann, center/reflect → 64 log-Mel, 50–14 000 Hz → mel bantları üzerinde BatchNorm (`bn0`, 64'e sabit) | ← | n_fft 512, hop 160, 64 Mel, 50–8 000 Hz | waveform × 2¹⁵ → Kaldi fbank 128 Mel, 25 ms / 10 ms → (fbank − 15.41663) / (2 × 6.55582) | Öğrenilmiş 7 katmanlı CNN, toplam adım 320 örnek (20 ms) | ← |
| Girdi normalizasyonu | Yok (bn0 öğrenilmiş) | ← | ← | Sabit fbank ort./SD (koda gömülü) | **`do_normalize = False`** | **`do_normalize = True`** |
| Model içi augmentation | **`train()` modunda SpecAugment** (zaman 64, frekans 8, ikişer şerit) | ← | ← | — | config: `mask_time_prob 0.05`, `layerdrop 0.05` | [NEEDS VERIFICATION] |
| Gömme boyutu | 512 | 2048 | 2048 | 768 / patch | 768 / 20 ms, 13 gizli durum | 1024, 25 gizli durum |
| Ağırlık | Zenodo 3987831 `Cnn10_mAP=0.380.pth` | `Cnn14_mAP=0.431.pth` | `Cnn14_16k_mAP=0.438.pth` | Resmi README (OneDrive) | HF `microsoft/wavlm-base-plus` | HF `microsoft/wavlm-large` |
| Parametre (yaklaşık) | ~5–6 M | ~80 M | ~80 M | ~90 M | ~95 M | ~316 M |

Kaynaklar: PANNs `pytorch/models.py` ve README (qiuqiangkong/audioset_tagging_cnn), Zenodo kaydı 3987831; BEATs `BEATs.py` ve README (microsoft/unilm); WavLM README (microsoft/unilm) ve HF `preprocessor_config.json` / `config.json`. Parametre sayıları checkpoint boyutlarından çıkarım [INFERENCE] — smoke test'te `sum(p.numel())` ile doğrulanacak. CNN10 için 16 kHz checkpoint **yok** [FROM OFFICIAL DOCS].

### 5.1 "STFT mi waveform mu?" sorusunun cevabı

- **KARAR:** Pretrained modellere 48 kHz kaynaktan tek seferde resample edilmiş **ham dalga formu** verilir; temsil, her modelin **kendi** frontend'iyle üretilir. Kendi STFT/log-Mel'imizi hesaplayıp vermeyiz.
- **NEDEN:** Ön-eğitimli ağırlıklar belirli bir girdi dağılımına göre öğrenildi (PANNs: 64 bantlı log-Mel ve bn0; BEATs: 128 bantlı fbank ve sabit ort./SD; WavLM: ham dalga formu). Farklı bir STFT ilk katmanı tanımadığı bir girdiyle besler. [FROM OFFICIAL DOCS] Ayrıca "STFT" bir *adımdır*; PANNs ve BEATs'in gördüğü şey lineer STFT değil, log-Mel'dir.
- **ALTERNATİFLER:** (a) Ortak bir log-Mel'i tüm modellere zorlamak → WavLM için anlamsız, PANNs/BEATs için dağılım kayması. (b) Modelin ilk katmanını yeniden eğitmek → küçük veride ön-eğitimin faydasını azaltır.
- **RİSK:** Modeller farklı SR ve bant genişliği görecek (PANNs-32k 14 kHz'e kadar, diğerleri 8 kHz'e kadar). Fark modelden değil bant genişliğinden gelebilir.
- **BİLİMSEL SONUÇ:** "Waveform vs spektrogram" sorusu yalnızca **sıfırdan eğitilen** model için serbest bir tasarım sorusudur ve şimdilik düşük önceliklidir. Bant genişliği etkisini izole etmek için **CNN14 (32 kHz) vs CNN14_16k** karşılaştırması temiz bir ablasyondur — mimari aynı, yalnız SR/frontend farklı. [DECISION]
- **UYGULAMA:** `configs/model_input_contracts.yaml`; her model için smoke test'te checkpoint'i `strict=True` ile yükle (frontend tensör boyutları uyuşmazsa yükleme hata verir — bu kendiliğinden bir format kontrolüdür), tek pencere için girdi/çıktı şekillerini assert et.

### 5.2 Gizli tuzaklar

1. PANNs `model.train()` çağrıldığında SpecAugment **otomatik açılır**; "augmentation kapalı" deneyde bilinçli olarak kapatılmalı. Gömme çıkarımında `model.eval()` şart.
2. PANNs `bn0` BatchNorm'u mel bantları üzerinde çalışır; küçük batch'li fine-tune'da running istatistikleri kayabilir → dondurma seçeneği bir karar noktası. [NEEDS VERIFICATION]
3. WavLM Base+ ve Large farklı normalizasyon bekler; aynı ön işleme kodu iki modele körlemesine uygulanamaz.
4. BEATs ağırlıkları OneDrive'da; Colab'a otomatik indirme sorunlu olabilir → bir kez indirip Drive'da sabit tutmak + sha256 kaydetmek. [NEEDS VERIFICATION]
5. WavLM İngilizce konuşmayla ön-eğitildi; Türkçe kelimelerde fonetik içerikten çok sesin kalitesine (paralinguistik) dair temsil işimize yarar. Üst katmanlar dile/foneme özgüleşir; katman ağırlıklı toplam (SUPERB tarzı) bu yüzden mantıklı. [INFERENCE]

### 5.3 Ses formatımız ve "ham dalga formu" ne demek?

**Evet, pretrained modellere ham dalga formu vereceğiz** — ama "ham", dosyayı olduğu gibi vermek demek değil:

```
101001_1.m4a (kapsayıcı + AAC)
   └─ ffmpeg ile BİR KEZ çöz ─────────────▶ float32, mono, 48 kHz dalga formu
        └─ resample_poly (tam oranlar) ────▶ 32 kHz (PANNs CNN10/CNN14)  |  16 kHz (CNN14_16k, BEATs, WavLM)
             └─ kenar sessizliği kırp, tepe normalizasyonu (D-015) ─▶ 4 s pencereler
                  └─ model: STFT / log-Mel / fbank / CNN kodlayıcıyı KENDİSİ yapar
```

| Özellik | Bizdeki (veri ekibi) | Uygun mu? | Not |
|---|---|---|---|
| Kapsayıcı | `.m4a` | Evet | Kapsayıcı yalnızca "zarf"; önemli olan içindeki codec. ffmpeg okur |
| Codec | Büyük olasılıkla AAC (kayıplı) [NEEDS VERIFICATION] | Evet, koşullu | PANNs ve BEATs'in ön-eğitim verisi AudioSet, YouTube videolarından derlenmiştir; modeller sıkıştırılmış sese yabancı değil [FACT]. Kritik olan mutlak kalite değil **tekdüzelik** |
| Sampling rate | 48 kHz (veri ekibi teyit etti) | Evet | 48→16 kHz tam 1/3, 48→32 kHz tam 2/3 oranı → kesirli/yaklaşık resample gerekmez |
| Kanal | Mono | Evet | Modeller mono bekler |
| Bant genişliği | AAC, bitrate'e bağlı olarak yüksek frekansları keser [NEEDS VERIFICATION] | Ölçülecek | PANNs-32k 14 kHz'e kadar bakar. Testte ffmpeg kodlayıcısıyla 96 kbps → ~21 kHz, 48 kbps → ~18 kHz; Apple'ın kodlayıcısı farklı olabilir |
| Sabit cihaz, yer, mesafe | Evet | Çok iyi | Cihaz/ortam confounder'ının en büyük kaynakları ortadan kalkıyor; geriye zamanla değişebilecekler kalıyor (iOS, kodlayıcı ayarı, oda gürültüsü, operatör) |

**Üç uyarı:**
1. *Tekdüzelik kaliteden önemli.* Tüm dosyalar aynı codec/bitrate/iOS sürümüyle kodlandıysa sıkıştırma izleri etiketle ilişkili olamaz. Bunlardan biri Mayıs'ta değiştiyse, zamansal confounder'ın akustik yolu tam olarak budur → ses denetimi bunu ölçüyor.
2. *Telefonun ses işlemesi.* Kayıt uygulamasında gürültü azaltma ("Kaydı Geliştir" vb.) açılıp kapandıysa gürültü tabanı döneme göre değişir → denetim yakalar; veri ekibine sorulacak.
3. *Gizlilik.* iPhone dosyaları GPS konumu taşıyabilir → ham dosyalar paylaşılmaz; dinleme ve önbellek dosyalarına meta veri taşınmaz.

**Neden bir kez çözüp kayıpsız önbellek (D-018)?** Yeniden kayıplı kodlamak yeni artefakt ekler; int16'ya çevirmek, AAC çözümünde ±1'i hafifçe aşabilen tepeleri kırpabilir → float32. Resample `scipy.signal.resample_poly` ile (deterministik, tam oranlı polifaz filtre).

---

## 6. Zamansal confounder stratejisi (D-017)

### 6.1 Sorun tam olarak ne?

```
İşe alım takvimi ──▶ etiket (sağlıklılar yalnız Ocak–Nisan'da toplandı)
        │
        └────────▶ kayıt tarihi ──▶ kayıt koşulları? ──▶ ses ◀── astım
                         └──────▶ hasta profili? (basamak, SFT, sigara)
```

- Tarih astıma neden olmaz; tarih ile etiket arasındaki ilişki **işe alım takviminden** geliyor. [FACT]
- Bu ilişki bir ses modeli için ancak tarih **sesi** de etkiliyorsa tehlikelidir. İki olası yol var:
  - **Kayıt koşulları (artefakt yolu):** oda gürültüsü, klima/mevsim, iOS veya uygulama güncellemesi (kodlayıcı, ses işleme), operatör ve telefon tutuşu alışkanlığı. [HYPOTHESIS — ses denetimi ölçecek]
  - **Hasta profili (gerçek ama karışık yol):** geç dönem hastalarında tedavi basamağı (p ≈ 3×10⁻¹²), SFT kategorisi (p = 0.001) ve aktif içicilik (p = 0.004) farklı; yaş, cinsiyet, GINA kontrolü ve ölçülen FEV1 değerleri benzer. [FACT] Model "geç dönem hastası" profilini öğrenirse bu bir artefakt değil, bir hastalık özelliği olabilir. Yani dönem sinyalinin hepsi kötü değildir — ama ayırt edilmesi gerekir.
- Cihaz ve yer sabit olduğu için en büyük risk kaynakları zaten yok; geriye zamanla değişebilenler kalıyor. [INFERENCE]

**"Her tarihte hem sağlıklı hem hasta yok mu?" — CSV'ye göre hayır:** [FACT]

| Gün türü | Gün | Kaydedilen hasta | Kaydedilen sağlıklı |
|---|---|---|---|
| Yalnız hasta kaydedilen günler | 52 | **245** | 0 |
| İki grubun birlikte kaydedildiği günler | 12 | 39 | 51 |
| Yalnız sağlıklı kaydedilen günler | 3 | 0 | 6 |

Hafta düzeyinde de aynı: 23 haftanın 14'ünde yalnız hasta kaydedilmiş. Hastaların %86'sı, o gün hiçbir sağlıklının kaydedilmediği bir günde kaydedilmiş.

**Tarih alanı gerçek bir ziyaret tarihi mi?** Hastaların %97.5'inde "veri toplama tarihi" SFT tarihiyle aynı gün; yaş 341/341 kişide bu tarihe göre hesaplanmış. [FACT] Sağlıklılarda SFT olmadığı için bu kontrol onlar için yapılamıyor; sağlıklıların gerçek kayıt günlerini ses dosyalarının iç zaman damgası doğrulayacak. Dosyalar sağlıklıların da Mayıs–Temmuz'da kaydedildiğini gösterirse, CSV tarihi sağlıklılar için kayıt günü değil demektir. O durumda sorun büyük ölçüde ortadan kalkar ve bu strateji güncellenir.

**Neden sorun? (benzetme).** Hasta fotoğraflarının çoğunun yazın güneşli bir odada, sağlıklılarınkinin kışın çekildiğini düşün. Model "güneş ışığı = hasta" demeyi öğrenebilir. Bu kısayol test setinde de çalışır, çünkü test setindeki hastalar da yazın çekilmiştir. Cross-validation bunu yakalamaz: katılımcıları rastgele böler, yazın çekilen hastalar hem eğitimde hem testte bulunur ve kısayol her fold'da "işe yarar". Sonuç: yanlış nedenle doğru cevap, şişirilmiş AUC ve yeni bir klinikte ya da yeni bir dönemde çöken bir model. Birkaç ortak gün bunu engellemez, çünkü 245 hasta karşılaştırılabilecek hiçbir sağlıklının olmadığı koşullarda kaydedildi.

Bu riskin gerçekleşmesi için tarihin **sesi** etkilemesi gerekir: ya kayıt koşulları (oda, iOS, operatör) ya da hasta profili zamanla değişmiş olmalı. İkisi de ölçülebilir; bu bölümün geri kalanı nasıl ölçüleceğini anlatıyor.

### 6.2 Neyi yapamayız (dürüst sınır)

- Nisan sonrası hiç sağlıklı kayıt yok → modelin **geç dönem koşullarındaki özgüllüğü** hiçbir analizle ölçülemez. [FACT]
- Tarihi modele kovaryat olarak eklemek işe yaramaz: sağlıklılarda Nisan sonrası değer yok, ayarlama ekstrapolasyon olur.
- Bu bir **veri toplama tasarımı** sorunudur. Analiz, etkisini ancak *ölçebilir ve sınırlayabilir*; kalıcı çözüm yeni veridir (6.6).

### 6.3 Beş katmanlı strateji

**Katman 1 — Ölç (eğitimden önce, ses denetiminde).** Meta veri (uzantı, codec, bitrate, bant genişliği, iOS sürümü, kodlayıcı, cihaz, dosya içi kayıt saati) ve kayıt koşulu ölçümleri (gürültü tabanı, sessizlik, seviye, süre) **yalnız hastalarda** erken vs geç dönemle karşılaştırılır. Etiket sabitken bir ölçüm dönemi ayırıyorsa kayıt koşulları değişmiştir.

**Katman 2 — Tasarımla kontrol: her model üç sayıyla raporlanır.** Hepsi aynı dış-CV out-of-fold tahminlerinden hesaplanır; ek eğitim gerekmez.

| Kod | Değerlendirme | Kimler | Neyi kontrol eder | Beklenen kesinlik (gerçek AUC = 0.80'de, %95) |
|---|---|---|---|---|
| E1 | Tam kohort | 284 / 60 | Hiçbirini — makaleyle karşılaştırma ve üst sınır | ±0.05 |
| **E2** | **Zaman penceresi** (9 Oca – 2 Nis 2024) | 99 / 57 | Aylar arası kayıt değişimi ve hasta profili kayması | ±0.07 |
| E3 | Aynı-gün tabakalı AUC (yalnız aynı gün kaydedilmiş hasta–sağlıklı çiftleri karşılaştırılır) | 12 gün, 39 / 51, 173 çift | Gün düzeyindeki **tüm** koşullar (oda, telefon, iOS, operatör) | ±0.11 |
| E4 | (duyarlılık) Yalnız zaman penceresinde yeniden eğit + değerlendir | 99 / 57 | Geç dönem hastalarının eğitime etkisi | — |

Kesinlik değerleri simülasyonla hesaplandı. [INFERENCE] E3 kesin bir tahmin vermez ama güçlü bir **yanlışlama testi**dir: gerçek AUC 0.65 bile olsa %98.5 olasılıkla 0.5'in üstünde çıkar.

**Neden E2'de modeli tam kohortla eğitiyoruz?** Daha çok veriyle eğitmek meşrudur. Sapma eğitimde değil, *değerlendirmede* oluşur. Zaman penceresinde bütün katılımcılar erken dönemdendir, bu yüzden "geç dönem → astım" kısayolu orada hiçbir avantaj sağlamaz. E4 bunun duyarlılık kontrolüdür.

**Okuma tablosu:**

| Desen | Yorum |
|---|---|
| E1 ≈ E2 ≈ E3 | Ses sinyali dönemden bağımsız; tam kohort sonucu da güvenilir |
| E1 > E2 (ΔAUC CI 0'ı dışlıyor), E2 ≈ E3 | Tam kohort sonucunun bir kısmı dönem bilgisinden; başlık sayısı E2 |
| E2 > E3 ya da E3 ≈ 0.5 | Gün düzeyi koşullar şüpheli (E3 küçük örneklemli → temkinli yorum) |
| E2 ≈ yaş-only baseline | Ses, yaşın ötesinde bilgi taşımıyor |

**Katman 3 — Negatif kontroller (yanlışlama testleri).** Dondurulmuş gömmelerle ucuzdur.

- **N1 Dönem probu (D-020).** Yalnız hastalarda çalışılır, çünkü orada etiket sabittir; sesten dönem ayırt edilebiliyorsa bunun nedeni astım olamaz. Referans çizgisini **EXP-002** koydu: klinik profil tek başına dönemi **AUC 0.855 ± 0.048** ile ayırıyor (yalnız tedavi basamağı 0.744; ölçülen spirometri 0.536). Yani "ses dönemi ayırıyor" bulgusu tek başına bir şey kanıtlamaz; ses hastalık ağırlığını kodluyorsa profil üzerinden de dönemi ayırabilir. İki tasarım kullanılır:
  - (a) **Artımlı (birincil):** "klinik profil" ile "klinik profil + ses dönem skoru" aynı fold'larda karşılaştırılır. Ses skoru iç CV'de üretilir ki sızmasın. ΔAUC ≈ 0 ise ses profilin ötesinde dönem bilgisi taşımıyor demektir (iyi haber). ΔAUC belirgin şekilde > 0 ise kayıt koşulları değişmiş olabilir.
  - (b) **Tabakalı (okunması kolay):** yalnız tedavi basamağı 4 olan hastalarda (81 erken / 66 geç) ses → dönem. Aynı tabakada klinik-only AUC da raporlanır.
  - Okuma: ses AUC ≈ 0.5 → kayıt koşulları sabit görünüyor; ses AUC yüksek ama profil açıklıyor → profil kayması (dikkat, ama artefakt değil); ses profilin ötesinde → kayıt koşulları değişmiş → Katman 4 azaltmaları devreye girer.
  - Kesinlik (Hanley–McNeil, AUC 0.6–0.7): tüm hastalarda ±0.06–0.07, basamak 4'te ±0.09. [INFERENCE]
- **N2 Skor–tarih ilişkisi:** yalnız hastalarda, OOF astım skoru ile kayıt tarihi arasındaki ilişki; klinik profile göre ayarlı.
- **N3 Arka plan kontrolü (D-020).** Her kayıttan konuşma içermeyen **en düşük enerjili** kareler seçilir; bunlardan basit bir özet çıkarılır (bant başına ortalama log-enerji + gürültü düzeyi), lojistik regresyonla iki hedef denenir: (i) zaman penceresinde etiket, (ii) hastalarda dönem. Odadaki uğultunun etiketi tahmin etmesi doğrudan ortam sızıntısıdır. **Önemli nüans:** kelime aralarındaki "sessizlik" nefes sesi içerebilir, ve duyulabilir soluma ya da hırıltı **gerçek bir astım işareti** olabilir. Bu yüzden yalnız nefes enerjisinin altındaki kareler kullanılır; sonuç pozitif çıkarsa bu kareler dinlenip "oda mı, nefes mi?" ayrılır. Ses denetimindeki baş/son sessizlik süreleri yeterli malzeme olup olmadığını gösterecek; yetersizse N3 yapılmaz ve bu kısıt yazılır.
- **N4 Referans çizgileri:** tarih-only (0.797) ve ID-only (0.800) AUC'leri her tabloda.

**Katman 4 — Azaltma (koşullu).** Yalnız Katman 1 veya 3 bir sürüklenme gösterirse uygulanır:

- A1. Kenar sessizliklerini kırpmak (zaten planlı, D-015).
- A2. Eğitim fold'unun sessizliklerinden oluşturulan, etikete ve döneme göre dengelenmiş bir gürültü havuzuyla karıştırma (Boll'un yaklaşımı; havuz yalnız train'den).
- A3. Birincil modeli yalnız zaman penceresinde eğitmek (E4'ü birincil yapmak).
- A4. Dönem sınıflandırıcısına karşı domain-adversarial eğitim. Yalnız araştırma uzantısı olarak; varsayılan değil, çünkü karmaşıklığı artırır ve dönemle ilişkili gerçek hastalık sinyalini de silebilir.

**Katman 5 — Önceden yazılmış karar kuralı.** "Ses astıma dair bilgi taşıyor" iddiası ancak şu durumda yapılır:

- (a) E2 AUC, **aynı katılımcılarda** yaş-only baseline'ından yüksek (eşleştirilmiş bootstrap ΔAUC %95 CI 0'ı dışlıyor);
- (b) E3 nokta tahmini > 0.5 ve E2 ile aynı yönde;
- (c) N3 arka plan kontrolü E2'de şans düzeyinden ayrılmıyor (veri izin verirse).

E1 her zaman raporlanır ama **başlık sayısı E2'dir**; E1 − E2 farkı ayrıca raporlanır. Bu kural, sonuçları görmeden önce yazıldı ve değiştirilirse DECISIONS.md'ye yeni bir karar olarak işlenir.

### 6.4 Uygulama ayrıntıları

- Fold tabakalaması etiket × dönem × yaş grubu: her test fold'una zaman penceresinden orantılı katılımcı düşer.
- `participants.csv` sütunları: `period`, `in_time_overlap`, `collection_day`, `same_day_as_other_group`.
- E3, gün içi hasta–sağlıklı çiftleri üzerinden hesaplanan tabakalı AUC'dir; CI, günler ve katılımcılar üzerinde bootstrap ile.
- Tarihi eksik 3 sağlıklı (101018, 101019, 101149) E2/E3'e girmez; dosya içi kayıt tarihi CSV ile uyumluysa tarihleri dosyadan tamamlanabilir (ayrı bir karar olarak).

### 6.5 Alternatifler ve neden seçilmedi

| Alternatif | Neden birincil değil |
|---|---|
| Yalnız zaman penceresiyle çalışmak | Örneklem yarıya iner, makaleyle karşılaştırma kaybolur; E2 + E4 aynı bilgiyi verir |
| Tarihi kovaryat yapmak | Sağlıklılarda Nisan sonrası değer yok → ekstrapolasyon |
| Geç dönem hastalarını tamamen atmak | E4 bunu duyarlılık analizi olarak zaten yapıyor; birincil yapmak eldeki bilgiyi gereksiz yere atar |
| Domain-adversarial eğitim | Karmaşık; gerçek hastalık sinyalini silme riski; ancak Katman 1/3 gerektirirse |

### 6.6 Kalıcı çözüm (öneri, engelleyici değil)

- Veri ekibine: Nisan sonrası kaydedilmiş sağlıklı var mı?
- Aynı protokolle (aynı telefon, oda, uygulama) yeni 20–30 sağlıklı ve birkaç hasta kaydı → **zamansal dış doğrulama seti**. Modelin özgüllüğünü yeni koşullarda ölçmenin tek yolu budur.

### 6.7 Ses denetiminin stratejiye kattıkları (2026-10-08, 4. tur)

1. **Tarih dengesizliği gerçek, CSV yanılsaması değil.** Apple zincirindeki 1 754 dosyada dosya içi kayıt tarihi, CSV tarihiyle %98.7 oranında aynı gün. Bu, sağlıklıların Apple dosyalarının **en az %92'sinin (271/293)** CSV tarihiyle aynı gün kaydedildiği anlamına geliyor. Yani sağlıklılar gerçekten Ocak–Nisan 2024'te kaydedilmiş. [FACT]
2. **Yeni confounder: günün saati.** Zaman damgası olan dosyalarda hastalar, sağlıklılardan belirgin biçimde daha erken saatte kaydedilmiş. Saat → etiket AUC'si görev başına 0.13–0.17 (n = 233–281 dosya); yani hasta–sağlıklı çiftlerinin yaklaşık %85'inde hasta daha erken saatte kaydedilmiş. Çoklu karşılaştırma sınırı yaklaşık 0.33, bu değerler onun çok dışında. [FACT] Günün saati sesi iki yoldan etkileyebilir [INFERENCE]:
   - *fizyolojik:* sabah sesi, gün içinde ses yorgunluğu, astım semptomlarının sabaha karşı ağırlaşması;
   - *ortam:* klinik yoğunluğu, gürültü.
   Tarih gibi bu da **tasarımdan gelen** bir confounder'dır.
3. **İki kodlama zinciri var.**
   - 1 759 dosya doğrudan Apple kaydı: marka `iso5`, "Core Media Audio", zaman damgalı.
   - 634 dosya FFmpeg'den geçmiş: `Lavf59.16.100`, marka `M4A `, zaman damgasız.
   Zincir etiketle ilişkili değil (FFmpeg oranı sağlıklı %29, hasta %26; p = 0.22), ama hastalarda dönemle ilişkili (erken %34, geç %21; p = 6×10⁻¹⁰). [FACT] FFmpeg'in yalnız kapsayıcıyı mı değiştirdiği yoksa sesi yeniden mi kodladığı henüz bilinmiyor. [NEEDS VERIFICATION — bitrate ve bant genişliğinin zincire göre dağılımı gösterecek]
4. **Bant genişliği iki tepeli.** Medyan 13.5 kHz; dosyaların en az %5'i 24 kHz (hiç kesim yok); en düşük 11.3 kHz. Muhtemelen zincire bağlı. [HYPOTHESIS]
5. **Prosedür farkları (orta düzey).** Çoklu karşılaştırma düzeltmesinden sonra:
   - *etikete göre:* yalnız süre (slot 4, AUC 0.34) sınırı aşıyor; sağlıklıların kayıtları biraz daha uzun.
   - *hastalarda döneme göre:* süre, baş/son sessizlik ve saat (AUC 0.35–0.36 / 0.63–0.64) sınırı aşıyor; gürültü tabanı sınırda (0.37); bitrate ve bant genişliği aşmıyor.
   Şans sınırları (Bonferroni, 84 test) etiket için ≈ 0.36/0.64, dönem için ≈ 0.38/0.62. [INFERENCE] Yani zamanla değişen şeyler daha çok **kayıt prosedürü** (ne zaman başlatıp durdurulduğu, günün hangi saati), kodlayıcı değil.

**Strateji güncellemesi:** Günün saati E1–E3'e ek bir kontrol gerektirir (D-022): saat-only baseline (EXP-003), aynı gün tasarımının "aynı gün + aynı yarım gün" sürümü (sayılar yeterliyse) ve saate göre alt grup performansı. Saat yalnız Apple zincirindeki dosyalarda var (~%73); FFmpeg zincirindeki katılımcılar için bilinmiyor.

### 6.8 "Sesi temizleyerek ya da değiştirerek bu sorunu çözebilir miyiz?"

**Kısa cevap: kısmen.** Ses işleme, yalnızca **teknik ve bilinen** farkları giderebilir; işe yarayıp yaramadığı da **ölçülerek** gösterilmelidir. Elimizdeki en güçlü confounder'lar (tarih, günün saati, hasta profili) teknik değil, sesin kendisinde yaşıyor; hiçbir filtre onları gideremez.

| Fark kanalı | Nerede yaşıyor | Ses işleme giderir mi? |
|---|---|---|
| Dosya meta verisi (iOS/kodlayıcı adı, marka, zaman damgası) | Dosya başlığında | **Zaten sorun değil:** modele yalnız çözülmüş ses örnekleri gider, etiketler atılır (D-018) |
| Kodlayıcının bant genişliği kesimi (11–24 kHz) | Seste, ~11 kHz üstünde | **Evet:** tüm dosyaları en düşük kesimin altından alçak geçiren filtreyle eşitlemek. 16 kHz modellerde resample zaten 8 kHz'te keser; sorun yalnız 32 kHz PANNs yolunda |
| Sampling rate farkı (1 dosya 44.1 kHz) | Seste | **Evet:** ortak SR'ye resample |
| Baş/son sessizlik, kaydı başlatma/durdurma alışkanlığı | Seste, konuşma dışında | **Evet:** kenar kırpma |
| Durağan oda gürültüsü (uğultu, klima) | Seste, konuşmanın altında da | **Kısmen:** eğitimde gürültü karıştırma (değişmezlik öğretir); gürültü giderme önerilmez |
| AAC sıkıştırma izleri (konuşma bandında) | Seste | **Hayır:** geri alınamaz; yeniden kodlama üstüne yeni iz ekler |
| Oda yankısı, mikrofon açısı/mesafesi | Sesin "rengi" | **Güvenilir biçimde hayır** |
| Günün saati (sabah sesi, yorgunluk, astım ritmi) | Sesin kendisinde (fizyoloji) | **Hayır** |
| Hasta profili kayması (tedavi basamağı, SFT) | Sesin kendisinde | **Hayır** |

**Önerilen üç fikrin değerlendirmesi:**

1. *"Hepsini aynı DAW'dan export etmek, iOS izi kalmasın."* iOS izi dosyanın **meta verisinde**; model onu hiç görmüyor. Modelin gördüğü örneklerdeki izler (bant genişliği, sıkıştırma artefaktı) ise export ile silinmez. Kayıplı export üstüne yeni bir sıkıştırma katmanı ekler; kayıpsız export (WAV) ise bizim çözme zincirimizin aynısıdır. Fikir doğru yöne bakıyor, ama iz meta veride değil örneklerde, ve onun doğru karşılığı **bant sınırlama + ortak SR**.
2. *"Gürültüyü temizleyip hepsine aynı arka planı eklemek."* Varsayılan olarak önerilmez, üç nedenle:
   - (a) Gürültü gidericiler "temiz konuşma" için eğitilmiştir. Nefesli ses, hava türbülansı ve hırıltı gibi **astımla ilişkili olabilecek** bileşenleri de gürültü sayıp silebilirler. [INFERENCE]
   - (b) Gidericinin çıktısı girişteki gürültünün türüne ve SNR'ye bağlıdır. Farklı koşullarda farklı artefakt ("musical noise") üretir, yani confounder'ı silmez, şeklini değiştirir. [INFERENCE]
   - (c) Geri dönüşsüz bir işlemdir. Sonra performans düşük çıkarsa "sinyal mi yoktu, yoksa biz mi sildik?" ayrılamaz.
3. *"Hepsine aynı arka plan gürültüsünü eklemek."* Bu bir **maskeleme**dir. Ancak eklenen gürültü her kaydın kendi gürültüsünden yüksekse farkları örter; bunun bedeli herkes için SNR kaybıdır. Konuşma bandındaki farklara (yankı, mesafe, saat) da dokunmaz. Daha iyi bir sürüm, gürültüyü **yalnız eğitimde ve rastgele** karıştırmaktır (Boll ve ark.): model arka plana güvenmemeyi öğrenir, test verisi ise değişmez. Bu D-016 kapsamında bir ablasyon olarak denenecek.

**Bunun yerine yapacağımız (D-021):** kayıpsız çözme (meta veri atılır) → ortak SR → 32 kHz yolu için ~11 kHz alçak geçiren filtre (kesin değer envanterden seçilecek) → kenar kırpma → tepe normalizasyonu. Gürültü giderme yok, DAW/kayıplı export yok.

**İşe yaradığını nasıl ölçeceğiz?** Elimizde mükemmel bir ölçü çubuğu var. Kodlama zinciri (Apple / FFmpeg) **bilinen, teknik** ve iki grupta da bulunan bir etiket. Harmonizasyondan önce ve sonra bir **"zincir probu"** çalıştırılacak: ses gömmelerinden zincir tahmin edilebiliyor mu? Önce tahmin edilebiliyor, sonra ≈ 0.5 ise harmonizasyon teknik kanalı kapatmış demektir. Aynı ölçüm hastalarda dönem probu (D-020) için de yapılacak. Kanıt yoksa "temizledik" denmeyecek.

### 6.9 Kayıt bağlamı analizi (EXP-003) ve revize birincil test (D-023) — 5. tur

**Kodlama zincirleri.** Zincir katılımcı düzeyinde değil, **dosya düzeyinde** karışık: 285 katılımcının kayıtlarında iki zincir birlikte var, 55'inde yalnız Apple, 2'sinde yalnız FFmpeg. [FACT] FFmpeg dosyaları yalnızca yeniden paketlenmemiş, **yeniden kodlanmış**: bitrate medyanı 159 kbps (Apple 185), bant genişliği neredeyse hep 24 kHz (Apple'da çoğunlukla 12–14), süre biraz daha kısa (10.1 / 10.5 s). [FACT] Zincir yeniden deneme numarasıyla ilişkili değil (p = 0.20); slot 1 ve 3'te biraz daha az. Mekanizma bilinmiyor; muhtemelen iki ayrı kayıt/yükleme yolu var. [HYPOTHESIS → veri ekibine soru] En düşük kodlayıcı kesimi 11.27 kHz → **alçak geçiren kesim 11.0 kHz** (D-021).

**Günün saati — veri setinin en güçlü confounder'ı.** Seans başlangıç saati 340/342 katılımcıda biliniyor; zincir dosya düzeyinde karıştığı için neredeyse herkesin en az bir zaman damgalı dosyası var.

| | Seans başlangıcı (medyan, IQR) | Öğleden önce |
|---|---|---|
| Astım (n = 282) | 11:05 (10:13–12:33) | %70 |
| Sağlıklı (n = 58) | 15:10 (13:52–16:27) | %21 |

**EXP-003 — sesi kullanmayan bağlam baseline'ları** (lojistik regresyon, 5-fold × 20, ort. ± SD):

| Özellik | Ses kohortu (282/58) | Zaman penceresi (97/57) |
|---|---|---|
| Saat | 0.841 ± 0.061 | **0.873 ± 0.053** |
| Tarih | 0.806 ± 0.055 | 0.558 ± 0.118 |
| **Saat + tarih** | **0.931 ± 0.032** | 0.881 ± 0.055 |
| Yaş | 0.681 ± 0.076 | 0.671 ± 0.080 |
| Yaş + saat + tarih | 0.933 ± 0.034 | 0.908 ± 0.044 |

Yalnız öğleden sonra kayıtlarına bakmak da yetmiyor: orada saat 0.767, tarih 0.862 ile ayırıyor. Sağlıklılar öğleden sonranın daha geç saatlerinde kaydedilmiş. [FACT]

**Ne demek?** Sesi hiç kullanmadan, yalnızca "ne zaman kaydedildi" bilgisi etiketi AUC 0.93 ile ele veriyor. Kayıt bağlamının akustik izini (oda, gün ışığı/klima, klinik yoğunluğu, sabah sesi) öğrenen bir model, ses biliminden hiçbir şey öğrenmeden çok yüksek sayılar üretebilir. Yayınlanmış AUC 0.77 bu tavanın altında kalıyor. [FACT + INFERENCE]

**Saat kısmen gerçek fizyoloji olabilir:** sabah sesi, gün içinde ses yorgunluğu, astımın günlük ritmi; ayrıca hastaların kaydı spirometri ve bronkodilatör testi gününe denk geliyor. Bu durumda saati kontrol etmek gerçek sinyalin bir kısmını da siler. Saat-kontrollü sonuçlar bu yüzden **muhafazakâr** okunacak. [INFERENCE]

**Değerlendirme tasarımlarının dengesi.** Tabakalar içinde saat, tarih ve yaşın etiketi hâlâ ne kadar ayırdığı gösteriliyor; 0.5 = tam denge. Kesinlik: gerçek AUC 0.75 varsayımıyla %95 aralığın yarı genişliği (simülasyon).

| Tasarım | Astım / sağlıklı | Saat | Tarih | Yaş | ± kesinlik |
|---|---|---|---|---|---|
| E1 tam kohort | 283 / 59 | 0.158 | 0.808 | 0.681 | 0.069 |
| E1h + aynı 1 saat dilimi | 208 / 51 | 0.422 | 0.823 | 0.708 | 0.094 |
| E2 zaman penceresi | 98 / 57 | **0.125** | 0.436 | 0.671 | 0.079 |
| **E2h zaman penceresi + aynı 1 saat dilimi** | 65 / 50 | **0.378** | **0.375** | 0.704 | 0.125 |
| E3 aynı gün | 39 / 52 | **0.103** | 0.500 | 0.711 | 0.121 |
| E3h aynı gün + aynı 1 saat dilimi | 9 / 14 | 0.214 | 0.500 | 0.929 | 0.250 |

**Sonuçlar:**
- Hiçbir tasarım tarih, saat ve yaşı aynı anda dengelemiyor.
- E2 ve E3, saat açısından tam kohorttan bile kötü: aynı gün kaydedilenlerde de hastalar sabah, sağlıklılar öğleden sonra.
- En dengeli seçenek E2h, ama belirsizliği yüksek (±0.125).

**Bu yüzden birincil test değişti (D-023).** Bu karar ses modeli sonuçları görülmeden önce alındı. Birincil soru artık "ses astımı ne kadar iyi ayırıyor?" değil, **"ses, kayıt bağlamı (tarih + saat) ve yaşın ötesinde astım bilgisi taşıyor mu?"**

- **T1 (birincil, artımlı test):** ikinci düzey tekrarlı CV'de "yaş + saat + tarih" modeline OOF ses skoru eklenir. ΔAUC ve eşleştirilmiş katılımcı-bootstrap CI raporlanır; ses skorunun bağlam-ayarlı odds oranı ve CI'ı da verilir (tavana yakın AUC'de ΔAUC küçük kalacağı için). Kanıt: ΔAUC CI 0'ı dışlıyor **ve** odds oranı CI'ı 1'i dışlıyor.
- **Destekleyici tasarım:** E2h tabakalı AUC (> 0.5, CI ile), denge sütunlarıyla birlikte.
- **Her zaman raporlanır:** E1 (makaleyle karşılaştırma), E1h, E2, E3, her biri denge tablosuyla. E2 artık başlık sayısı değil.
- **Yorum kuralı:** T1 ve E2h aynı yönü göstermiyorsa "bağlamdan bağımsız bir ses sinyali gösterilemedi" denir. Bu da yayınlanabilir, değerli bir sonuçtur.

### 6.10 Confounder kontrol yöntemleri: hangisi gerçek, hangisi kozmetik? (6. tur)

Tam değerlendirme, gerekçeler ve deney tasarımı: **`docs/CONFOUND_CONTROL_DESIGN.md`**. Kararlar: D-025 (test spesifikasyonu), D-026 (kontrol yığını), D-027 (fine-tuning protokolü), üçü de ÖNERİLDİ.

> **7. tur (D-028):** Bu bölümdeki kontrol deneyleri (EXP-020, EXP-022) ve D-025–027 önerileri **değerlendirme aşamasına ertelendi**. Model geliştirme kabul edilmiş plana göre sürüyor; değerlendirme, geliştirme bitince kaydedilmiş tahminler üzerinde yapılacak.

**Yöntemlerin rolü:**

| Yöntem | Rolü |
|---|---|
| Artımlı test T1 | **kanıt (birincil)** |
| E2h, ayarlı koşullu lojistik regresyonla | **kanıt (destekleyici)** |
| Spisak tam / kısmi testleri | **kanıt** |
| Harmonizasyon | hijyen; saat / tarih için **kozmetik** |
| Katılımcı split'i | hijyen; confounding kontrolü **değil** |
| Gün-gruplu CV | duyarlılık |
| Codec / cihaz kontrolü | gerekli, ana confounder'a dokunmuyor |
| Residualization | yalnız dondurulmuş gömmede, fold içinde, hasta-içi; duyarlılık |
| Adversarial | marjinal hâli **yanlış hedef**; etiket-koşullu hâli yalnız uzantı |

**Yeni veri kontrolleri (hepsi sesi model olarak kullanmadan):**
- **EXP-004:** Hastalarda sabah / öğleden sonra kayıtları kaba ses ölçümleriyle ayrılamıyor (0.549 ± 0.067). → Kaldırılacak kaba bir oda izi yok.
- **EXP-004b:** Sabah ve öğleden sonra hastaları klinik olarak benzer (0.528 ± 0.071). → Etiket içi saat kontrastı temiz bir negatif kontrol.
- **Referans çizgileri:** Bağlamı mükemmel kodlayan ama astım bilgisi taşımayan bir model E2h'de **0.35–0.70** arası bir değer alır (yaş: 0.703). → Ham E2h AUC'si 0.5'e karşı yorumlanamaz.
- **SIM-001:** Gerçek bağlamla, ses skoru yapay. Sonuçlar:
  - Kuadratik bağlam modeliyle yanlış pozitif oranı **0.216**, spline modelle 0.060.
  - Ölçülmemiş güne özgü etkide ~0.12; gün-kümeli SE bunu düzeltmiyor.
  - %80 güç ancak aynı-bağlam AUC'si ≈ 0.69'da (α = 0.05), Holm düzeyinde ≈ 0.74'te; bu sayılar iyimser.
  - Tam kohort T1'in gücü E2h'ninkine yakın: bilgi örtüşme bölgesinde.

**Pozitiflik sınırı:** Sabah kaydedilmiş sağlıklı 12, öğleden sonra kaydedilmiş hasta 86; geç dönemde hiç sağlıklı yok. Bu bölgelerin dışında sesin astım bilgisi bağlamdan **hiçbir yöntemle** ayrılamaz. [FACT + INFERENCE]

**Önerilen sıra:**
1. **EXP-020 (G1):** Gömmeler etiket içinde saati kodluyor mu?
2. **EXP-021:** 6 backbone, dondurulmuş, tüm lens seti (L1–L8); onaylayıcı aile, Holm düzeltmeli.
3. **EXP-022:** Koşullu azaltma, yalnız G1 pozitifse.
4. **EXP-030:** Fine-tune. Backbone T1'e göre seçilir (E1'e göre değil); kısmi FT, sabit epoch, confounded metrikle early stopping yok.

---

## 7. Ortak değerlendirme protokolü (tüm modeller)

**İstatistiksel birim:** katılımcı. Her deneyde ayrı ayrı raporlanır: N katılımcı (astım/sağlıklı), N kayıt, N segment.

**Kohortlar:**
- Birincil: analiz edilen görev(ler) için geçerli ses kaydı olan herkes → **342 katılımcı (283 astım / 59 sağlıklı)**; görev 4 (ordu) için 282 / 59. Sesi olmayanlar: 101043, 101149, 101345–101348. [FACT + DECISION, D-002 — KABUL]
- Zorunlu zaman kontrolleri: zaman penceresi 155 (98/57) ve aynı gün tasarımı 12 gün (39/51) — Bölüm 6; günün saati için ek kontroller — Bölüm 6.7, D-022.

**Split (D-009):**
- Dış döngü: katılımcı düzeyinde, **etiket × kayıt dönemi (erken / geç / bilinmiyor; D-024 kayıt tarihiyle) × yaş grubu (≤40 / >40 / bilinmiyor)** ile tabakalı 5-fold (D-009; D-026'daki sabah/öğleden sonra eklemesi önerisi D-028 ile ertelendi) (her test fold'una zaman penceresinden orantılı katılımcı düşsün diye); ucuz modellerde 5 tekrar (25 fold), fine-tune'da ölçülen maliyete göre 1–3 tekrar.
- Fold atamaları **bir kez** üretilir, `splits/` altında sha256 ile saklanır; **her deney aynı dosyayı okur** → modeller aynı katılımcılarda eşleştirilmiş olarak karşılaştırılabilir.
- İç döngü: her dış train kümesinde tabakalı 80/20 katılımcı bölünmesi → yalnız early stopping ve eşik için. Lineer problarda düzenlileştirme (C) için iç 5-fold.
- Hiperparametreler fine-tune için **önceden sabit** (Boll değerleri); değiştirmek ayrı, kayıtlı bir ablasyondur.

**Kilitli test seti — KARAR: Seçenek A (D-009, KABUL):**
- *Seçenek A (seçildi):* Ayrı holdout yok; tüm tahminler tekrarlı dış-CV'den. Gerekçe: %20 holdout = 12 sağlıklı → AUC CI ±0.12, sonuç neredeyse bilgi taşımaz; dış fold'lar zaten modeli hiç görmemiş test verisidir. Araştırmacı serbestlik derecesine karşı koruma: karşılaştırılacak konfigürasyonları önceden yazmak ve **hepsini** raporlamak (yalnız "en iyiyi" değil).
- *Seçenek B (reddedildi):* %20 kilitli holdout (≈12 sağlıklı / 57 astım), geliştirme kalan %80'de. Artı: dış CV sonuçlarına bakarak verdiğimiz kararlara karşı bağımsız kontrol. Eksi: geliştirme verisi azalır, final tahmin çok gürültülü.

**Toplama hiyerarşisi:** segment → kayıt (pencere logit'lerinin ortalaması) → katılımcı (görev başına = kayıt; çok görevli füzyonda görev logit'lerinin ortalaması).

**Metrikler (D-010):** Birincil: katılımcı düzeyi ROC-AUC + dengeli doğruluk. İkincil: duyarlılık, özgüllük, MCC, PR-AUC (**her iki sınıf için**). Doğruluk yalnızca çoğunluk tabanıyla (0.826) yan yana.

**Belirsizlik:** fold başına metrik → ort. ± SD ve aralık (fold-fold değişkenliği); her tekrar için havuzlanmış OOF tahminlerinde katılımcı bootstrap %95 CI (2 000 tekrar); tekrarlar arası özet.

**Model karşılaştırması (D-011):** aynı fold'larda eşleştirilmiş ΔAUC; Nadeau–Bengio düzeltilmiş tekrarlı-CV t-testi; tekrar başına havuzlanmış OOF'ta DeLong. Bir iyileşme ancak (i) ΔAUC CI'ı sıfırı dışlıyor ve (ii) yön zaman-örtüşen alt kohortta da korunuyorsa "destekleniyor" sayılır.

**Model geliştirme sırasında her sonuçla birlikte:** üst sınır uyarısı ve iki referans satırı (yalnız bağlam 0.93, yalnız yaş 0.68); dış-test tahminleri katılımcı / kayıt / segment düzeyinde saklanır (D-028).

**Değerlendirme aşamasında (D-028; model geliştirme bitince, kaydedilmiş tahminler üzerinde) başlık sonuçları için kontroller:** lens seti L1–L8 (`docs/CONFOUND_CONTROL_DESIGN.md` Bölüm 4; ayrıntılar D-025 önerisi üzerinden kesinleşecek): birincil artımlı test T1 (spline bağlam modeli, olabilirlik oranı testi + OR; D-025), Spisak tam / kısmi testleri, E2h koşullu lojistik regresyonu (ham E2h AUC'si referans çizgileriyle), karşıt-hücre tablosu, N1–N5, zincir duyarlılığı; E1/E1h/E2/E3 her zaman raporlanır; yaş-only ve yaş+cinsiyet+sigara baseline'ı; "yaşın ötesinde bilgi" (OOF ses skoru + yaş vs yalnız yaş, ikinci düzey CV); zaman-örtüşen alt kohort; dönem probu (yalnız hastalarda erken vs geç kayıt, ses gömmelerinden); kayıt-koşulu baseline'ı (gürültü tabanı, süre, sessizlik); alt gruplar (yaş grubu, cinsiyet, sigara; hastalarda SFT normal/anormal, GINA kontrol, basamak).

---

## 8. Baseline yeniden üretimi (MFCC) — plan

Kod, eşleme doğrulandıktan sonra yazılacak (ses verisi olmadan test edilemez).

- **EXP-010 — Sadık yeniden üretim:** görev başına; 22.05 kHz; enerji tabanlı kırpma (parametre makalede yok → belgelenmiş seçim); 12 MFCC + Δ + ΔΔ (2048/512, Hamming) zaman ortalaması + 8 spektral özet = 44; StratifiedKFold(5, shuffle, seed 42) katılımcı düzeyinde; fold içinde StandardScaler + SMOTE; makaledeki sınıflandırıcılar varsayılan ayarlarla. Hedef: makaledeki AUC aralığına (≈0.65–0.77) fold SD'si içinde ulaşmak. Ulaşılamazsa nedenleri (bilinmeyen detaylar) rapora yazılır — bu da bir sonuçtur.
- **EXP-011 — Aynı özellikler, bizim protokolümüz:** ortak split dosyaları, tekrarlı CV, sınırlı ve önceden belirlenmiş model ailesi (LR, SVM-RBF, gradient boosting) iç döngüde ayarlanmış, katılımcı düzeyinde çok görevli füzyon ve Bölüm 6–7'deki tüm kontroller. RQ1'in gerçek karşılaştırma noktası budur.

---

## 9. Araştırma zinciri (deney merdiveni)

| Basamak | Deney | Cevapladığı soru | Maliyet [INFERENCE] |
|---|---|---|---|
| 0 | Confounder baseline'ları ✅ EXP-001 | RQ7 | saniyeler |
| 1 | MFCC: sadık + düzeltilmiş protokol (EXP-010/011) | RQ1 tabanı, RQ4, RQ5 | dakikalar (CPU) |
| 2 | **Dondurulmuş gömme + lineer prob**: CNN10, CNN14, CNN14_16k, BEATs, WavLM Base+, WavLM Large; görev başına | RQ3, RQ4, RQ5 | ~1 saat T4 (çıkarım) + dakikalar (prob) |
| 3 | Sıfırdan CNN10 (aynı frontend, rastgele başlatma) | RQ2 | fine-tune ile benzer |
| 4 | Uçtan uca fine-tune: Basamak 2'de öne çıkan 1–2 aile + PANNs (referans makaleyle karşılaştırma için) | RQ2, RQ3; dondurulmuş vs fine-tune ablasyonu | CNN14: fold başına ~10–20 dk tahmini → görev × tekrar ile çarpılır |
| 5 | Çok görevli füzyon; yaş/zaman kontrolleri; alt gruplar; ses + yaş/cinsiyet/sigara | RQ6, RQ7, RQ8, RQ9 | ucuz (OOF tahminleri üzerinde) |

**Neden önce dondurulmuş prob?** Ucuz, deterministik, düşük varyanslı; altı backbone'u aynı protokolde karşılaştırmanın en dürüst yolu; GPU saatlerini yalnız gerekçesi olan fine-tune'a harcatır; "dondurulmuş vs fine-tune" ablasyonunu kendiliğinden sağlar. [DECISION, D-012]

**RQ9'un yeniden tanımı:** Klinik değişkenlerin neredeyse tamamı yalnız hastalarda var; bunları modele eklemek etiketi eksiklik deseninden sızdırır. Astım-vs-sağlıklı için kullanılabilecek tek klinik bilgi yaş, cinsiyet, sigara. Soru şöyle olmalı: *"Ses, yaş/cinsiyet/sigaranın ötesinde bilgi taşıyor mu?"* [FACT + DECISION, D-006]

**Fine-tune bütçesi uyarısı:** CNN14 için kaba tahmin: dış train ≈ 275 katılımcı × ~4 pencere ≈ 1 100 segment (dengelemeyle ~1 500) → batch 16'da ~95 adım/epoch; T4 + mixed precision'da epoch başına 10–20 s → fold başına 10–20 dk. 7 görev × 5 fold × 3 tekrar × 2 model ≈ 40+ saat. Bu yüzden fine-tune yalnız seçilmiş görevlerde; gerçek süre smoke test'te ölçülecek. [INFERENCE]

---

## 10. Altyapı (özet — ayrıntı `docs/COLAB_WORKFLOW.md`)

- **GitHub (private):** kod, config'ler, agrega raporlar, DECISIONS.md, EXPERIMENTS.md, `results/registry.csv`. Hiçbir ses veya katılımcı düzeyi tablo girmez.
- **Google Drive:** `data_raw/` (salt okunur, sha256'lı), `data_derived/` (participants.csv, recording_map.csv, resample önbellekleri), `models/` (indirilmiş ağırlıklar), `experiments/EXP-XXX/` (config, log, checkpoint, OOF tahminleri).
- **Colab:** repo her oturumda `/content`'e klonlanır (Drive içinde git kırılgan ve yavaştır); ses zip'i Drive'dan yerel diske kopyalanıp açılır (Drive üzerinden binlerce küçük dosya okumak yavaştır).
- **Kayıt düzeni:** `results/registry.csv` tek doğruluk kaynağı (kod otomatik satır ekler); Google Sheets bunun **görünümü** (iki kaynak = tutarsızlık).
- **Checkpoint kuralı (eğitim fazları için):** her epoch `latest.pt` (atomik yazım), en iyi val'de `best.pt`, epoch CSV logu her satırda flush, config + seed + git commit + `pip freeze` deney klasörüne; kaldığı yerden devam. Drive kotası (ücretsiz 15 GB) CNN14'te fold başına ~1 GB tam durum demek → biten fold'larda yalnız tahminler + fp16 en iyi ağırlık tutulur. Fine-tune'da hem en iyi val checkpoint'inin hem son epoch'un dış-test tahminleri saklanır (D-028).

---

## 11. Açık sorular ve belirsizlikler

**Ses denetimiyle çözülecek [NEEDS VERIFICATION]:** tüm dosyalar `.m4a` ve 48 kHz mono mu, codec ve bitrate, etkin bant genişliği; **sağlıklıların gerçek kayıt günleri (dosya içi zaman damgası)**; 344 × 7 kaydın varlığı; 101345–101348'in sesi var mı; kopyalar; meta verinin (iOS sürümü, kodlayıcı) ve kayıt koşullarının döneme göre değişip değişmediği; dosya içi tarih ↔ CSV tarihi.

**Dinleyerek teyit edilecek:** ~~slot → görev eşlemesi; 101042 / 101050 aynı kişi mi~~ → ikisi de çözüldü: eşleme uyumlu, 101042 ve 101050 farklı kişiler (Bölüm 3).

**Veri ekibine sorulacaklar:**
1. ~~Slot 2 ve 3~~ → çözüldü: 2 = araba, 3 = ana.
2. Dışa aktarılan dosya, CSV'deki `UUID_k` alanındaki k. denemeye mi karşılık geliyor? Diğer denemeler saklanıyor mu?
3. ~~101345–101348~~ → ses kaydı yok, birincil analiz dışında (D-002).
4. 9 sağlıklı gönüllüde tarama kriterleri neden boş?
5. Mayıs–Temmuz 2024'te değişen bir şey oldu mu: oda, operatör, telefon (aynı cihaz mı?), iOS sürümü, **kayıt uygulaması** (Sesli Notlar mı, Ascleb uygulaması mı?), gürültü azaltma / "Kaydı Geliştir" gibi ayarlar?
6. Sağlıklılar nasıl seçildi (personel, refakatçi, gönüllü)? Spirometri yapıldı mı?
7. GINA kontrol sayıları neden makalede 47/237, dışa aktarımda 41/242?
8. Geç dönem hastalarında tedavi basamağı 4 oranı %82'den %36'ya, SFT "kombine" oranı %19'dan %41'e değişmiş (ölçülen FEV1 değerleri ise benzer): işe alım kaynağı ya da kodlama pratiği değişti mi?
9. Nisan 2024 sonrası kaydedilmiş sağlıklı gönüllü var mı? Aynı protokolle yeni kayıt toplamak mümkün mü? (Bölüm 6.6)
10. ~~Sağlıklıların tarihi gerçek kayıt günü mü?~~ → Evet: dosya zaman damgaları doğruladı (Bölüm 6.7).
11. Makalede 2 408 kayıt var, dışa aktarımda 2 393: 101043 ve 101149'un kayıtları ve 101244'ün slot 4 kaydı nerede?
12. Dosyaların %27'si FFmpeg'den geçmiş ve yeniden kodlanmış (`Lavf59.16.100`, zaman damgasız, farklı bitrate ve bant genişliği). Aynı katılımcının kayıtlarında iki yol karışık. Hangi durumda dosyalar bu yoldan geçiyor (yükleme hatası sonrası yeniden deneme, web arayüzü, sunucuda dönüştürme)?
13. Sağlıklılar neden çoğunlukla öğleden sonra (medyan 15:10), hastalar sabah (medyan 11:05) kaydedildi (klinik iş akışı, personel gönüllüler)? **Hastaların ses kaydı spirometri ve bronkodilatör (reversibilite) testinden önce mi sonra mı alındı?** Sonra ise ilaç sesi etkileyebilir.
14. 4 hastada dosya tarihi CSV tarihinden farklı (−30, +1, +5, +182 gün): CSV mi dosya mı doğru?

**Karar bekleyen yok.** D-015 (ön işleme) ve D-016 (augmentation) ses denetiminden sonra Faz 1'de kesinleşecek.

---

## 12. Durum raporu (8. tur — Faz 1)

- **Ne yaptık?**
  - Split dosyaları Colab'da yeniden üretildi; 5 dosyanın sha256'ı manifestle aynı (D-029 teyit).
  - Ses önbelleği tasarlandı ve yazıldı (D-030):
    - `scripts/build_audio_cache.py`;
    - sahte kayıtlarla test `tests/test_build_audio_cache.py` (9 kontrol geçti);
    - Colab notebook'u `notebooks/03_audio_cache.ipynb`.
- **Ne öğrendik?**
  - Split'ler farklı kütüphane sürümleriyle (scikit-learn 1.6.1 / 1.9.1) birebir aynı çıkıyor.
  - Denetim envanterinden tahmin, kırpma sonrası kayıtlar için:
    - kalan süre medyan 10.3 s, en kısa 2.3 s;
    - 17 kayıt 4 s'den kısa;
    - önbellek ≈ 4.6 GB.
  - Baştaki sessizliğin süresi etiketle ilişkili (AUC 0.369). Kırpma bu sesle ilgisiz ipucunu girdiden çıkarıyor.
- **Hangi kararları aldık?** D-030, uygulama ayrıntısı: kırpma payı 0.10 s, −1 dBFS, 511 taplık FIR, float32 tek dosya + memmap.
- **Hangi belirsizlikler kaldı?**
  - Drive'da ~5 GB boş alan olup olmadığı.
  - Dinleme kontrolü: kırpma konuşmayı kesiyor mu?
  - Bant genişliğinden zincir ayrımının filtreden sonra ne kadar kaldığı (rapor gösterecek).
- **Bir sonraki minimum gerekli adım:** Colab'da `03_audio_cache.ipynb` (CPU) → rapor PR'ı → dinleme kontrolü. Ardından MFCC baseline tasarımı (EXP-010 sadık yeniden üretim, EXP-011 ortak protokol).

*Önceki (7. tur):* confounder ayrı başlık (D-028), D-015/D-016 kabul, split dosyaları (D-029).
*Daha önce (6. tur):* confounder kontrol yöntemlerinin değerlendirmesi (`docs/CONFOUND_CONTROL_DESIGN.md`), EXP-004/004b, tasarım referans çizgileri, SIM-001, D-025/026/027 önerildi.
*Önceki (5. tur):* kayıt bağlamı analizi (EXP-003), D-021 kesimi 11.0 kHz, D-023, D-024.
