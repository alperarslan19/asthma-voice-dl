# Faz 0 Raporu — Sesten Astım Sınıflandırması

| | |
|---|---|
| Tarih | 2026-10-08 |
| Veri | `clinical_data.csv` (sha256 `cab18a35…a3af9b`), XLSX ile hücre hücre aynı (Data Report 2026-03-14) |
| Ses verisi | Veri ekibi (düzeltilmiş bildirim): iPhone 14, ağızdan 10 cm, hep aynı yer, mono, **48 kHz**, `.m4a`, adlandırma `<ID>_<slot>.m4a`, son dosya `101344_7.m4a`. **Henüz denetlenmedi** — `soundData` bu oturumda yok; `scripts/audit_audio.py` hazır ve testli. |
| Durum | Literatür, klinik denetim, girdi sözleşmeleri, protokol, zaman stratejisi ve altyapı tamam; ses denetimi, eşleme doğrulaması ve MFCC baseline bekliyor |
| Son güncelleme | 2026-10-08 (3. tur): eşleme düzeltildi (2 = araba, 3 = ana), 48 kHz ve `.m4a` teyit edildi; gün düzeyi kanıt, tarih alanının geçerliliği, EXP-002 ve negatif kontrol tasarımı (D-020) eklendi |

Etiketler: **[FACT]** veri/literatürle doğrudan destekli · **[FROM PAPER]** belirli makaleden · **[FROM OFFICIAL DOCS]** resmi kod/doküman · **[INFERENCE]** çıkarım · **[HYPOTHESIS]** test edilmemiş · **[DECISION]** bilinçli karar · **[NEEDS VERIFICATION]** kod/deneyden önce doğrulanmalı

---

## 0. Beş dakikada özet

1. **Zamansal confounder (en kritik bulgu).** Sağlıklı gönüllülerin tamamı 9 Ocak – 2 Nisan 2024 arasında kaydedilmiş; 284 hastanın **184'ü** bu tarihten *sonra* (Mayıs–Temmuz) kaydedilmiş. Sesi hiç kullanmadan, **yalnızca kayıt tarihi** etiketi AUC **0.797 ± 0.048** ile tahmin ediyor — yayınlanmış en iyi MFCC modelinden (AUC 0.769) yüksek. ID numaraları kayıt sırasıyla verildiği için (ρ = 0.985) **yalnızca katılımcı ID'si** de AUC 0.800 veriyor. [FACT] Bu, yayınlanmış modelin tarihi kullandığını *kanıtlamaz*; ama kayıt koşulları aylar içinde değiştiyse (oda, mevsimsel gürültü, klima, iOS güncellemesi, operatör) bir ses modelinin bu bilgiyi kısayol olarak öğrenmesi mümkündür. [INFERENCE] Zaman-örtüşen alt kohortta (99 hasta / 57 sağlıklı) tarihin tahmin gücü AUC 0.567 ± 0.116'ya düşüyor. [FACT] **Strateji (Bölüm 6, D-017):** ölç → tasarımla kontrol et (tam kohort / zaman penceresi / aynı gün) → negatif kontrollerle sına → gerekirse azalt → önceden yazılmış kurala göre yorumla.
2. **Yaş confounder'ı.** Sağlıklılar daha genç (medyan 40 vs 46, Mann–Whitney p = 1.5×10⁻⁵). **Yalnızca yaş** AUC 0.681 ± 0.077 veriyor ve zaman-örtüşen alt kohortta da sürüyor (0.674). [FACT] Referans makaledeki "gençlerde başarı çok düşük, yaşlılarda çok yüksek" deseni, tam da bir yaş kısayolunun üreteceği desendir. [INFERENCE]
3. **"%82 doğruluk" çoğunluk sınıfı seviyesinde.** 284/344 = **0.826**. Yayınlanmış en iyi model (gelecek, Voting) doğruluk 0.820, duyarlılık 0.933, özgüllük 0.283 → **dengeli doğruluk ≈ 0.61**. [FACT] Bu yüzden doğruluk tek başına raporlanmayacak. [DECISION]
4. **"STFT mi waveform mu?" pretrained modeller için serbest bir seçim değil.** PANNs, BEATs ve WavLM'nin üçü de API'de **ham dalga formu** alır ve temsili *kendi içinde* üretir (PANNs: STFT→64 bantlı log-Mel; BEATs: 128 bantlı Kaldi fbank; WavLM: öğrenilmiş CNN). Bizim işimiz doğru sampling rate'i ve modelin kendi frontend'ini kullanmak; kendi STFT'mizi hesaplayıp vermek ön-eğitimli ağırlıklarla uyumsuzluk yaratır. [FROM OFFICIAL DOCS] Doğrulanmış tuzaklar: CNN10 yalnızca **32 kHz** sürümüyle var; WavLM **Base+ normalizasyon yapmaz, Large yapar**; PANNs modeli `train()` modunda **SpecAugment'i kendiliğinden uygular**.
5. **Eşleme kuralı veri ekibinden geldi.** Dosyalar `<ID>_<slot>.m4a`; 1 = aaa, 2 = araba, 3 = ana, 4 = ordu, 5 = gelecek, 6 = titiz, 7 = ünlem (düzeltilmiş bildirim). Son dosya 101344_7 → 101345–101348'in sesi yok, birincil kohort 344 (D-002). Ses denetimi ve kısa bir dinleme teyidi bitmeden model eğitimine geçilmeyecek. [FROM DATA TEAM + DECISION]

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

### 2.3 Ses verisi (bekliyor — `scripts/audit_audio.py`)

**Veri ekibinin bildirdikleri [FROM DATA TEAM]:** iPhone 14, ağızdan 10 cm, tüm kayıtlar aynı yerde; mono; 48 kHz (teyit edildi); `.m4a` (düzeltilmiş bildirim); adlandırma `<ID>_<slot>.m4a`; son dosya `101344_7.m4a`; toplam 344 katılımcı. Makaledeki toplam (2 408 kayıt, 401 dk) 344 × 7 ve ~10 s/kayıt ile tutarlı. [FACT] Hepsi denetimle doğrulanacak. [NEEDS VERIFICATION]

Script, Colab'da çalıştığında şunları ölçecek:

| Ölçüm | Neden |
|---|---|
| Dosya sayısı (beklenen 2 408), uzantı, kapsayıcı, codec, bitrate, SR, kanal, süre | Format tekdüzeliği; "48 kHz mono `.m4a`" bildiriminin doğrulanması |
| Etkin bant genişliği (kayıplı kodlayıcının alçak geçiren kesimi) | PANNs-32k 14 kHz'e kadar bakar; kesimin yeri ve dönemler arası sabitliği |
| Meta veri etiketleri: iOS sürümü, kodlayıcı, cihaz modeli, marka | Kayıt zinciri zamanla değişti mi? (zamansal confounder'ın en olası akustik yolu) |
| Dosya içi kayıt zamanı ↔ CSV tarihi | Tarihin bağımsız doğrulanması; CSV'de tarihi eksik 3 sağlıklının tarihi |
| Konum etiketi var/yok (değer **yazılmaz**) | Gizlilik |
| Katılımcı × slot eşlemesi: OK / EKSİK / ÇOKLU; dosyası hiç olmayanlar | Eşleme tablosu |
| Dosya hash'i, çözülmüş PCM hash'i, spektro-zamansal parmak izi | Birebir ve yakın kopya (duplicate leakage) |
| Baş/son sessizlik, gürültü tabanı, konuşma seviyesi, SNR vekili, tepe, kırpılma | Kayıt koşulu ölçümleri |
| Tüm bunların etiketle ve **yalnız hastalarda** kayıt dönemiyle ilişkisi | Etiket sabitken dönemi ayıran bir ölçüm = kayıt koşulları değişmiş |

Script, kendi kendine yeten bir testle (`tests/test_audit_audio.py`) doğrulandı: sahte katılımcılar ve bilerek hatalar konmuş sahte ses dosyaları üretir. Yakalanan tuzaklar: eksik dosya, aynı slot için iki uzantı, video akışlı mp4, kalıba uymayan adlar, dosyası olmayan katılımcı, tarihi kaymış dosya, CSV'de tarihi olmayan ama dosyada olan katılımcı, konum etiketi (değeri hiçbir çıktıda yok), geç dönemde farklı iOS sürümü + yüksek gürültü (dönemle p ≈ 10⁻⁵¹), geç dönemde düşük bitrate (bant genişliği 21.3 → 18.1 kHz). Testler script'teki iki gerçek hatayı da yakaladı (meta veri sütunlarının eşleme tablosuna taşınmaması; pandas'ta `df.take`'in sütun değil metot olması). [FACT]

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

İlk bildirimdeki slot 2/3 çelişkisi veri ekibinin düzeltmesiyle giderildi. Yine de her slottan 3 kayıt dinlenerek kısa bir teyit yapılır (~5 dk): yanlış bir eşleme RQ4/RQ5'i ve makaleyle görev bazlı karşılaştırmayı sessizce bozar. `configs/slot_task_map.yaml` teyitten sonra `confirmed_by_listening` olur. [DECISION, D-019]

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

---

## 7. Ortak değerlendirme protokolü (tüm modeller)

**İstatistiksel birim:** katılımcı. Her deneyde ayrı ayrı raporlanır: N katılımcı (astım/sağlıklı), N kayıt, N segment.

**Kohortlar:**
- Birincil: analiz edilen görev(ler) için geçerli ses kaydı olan herkes; veri ekibine göre 101001–101344 = **284/60**. Demografik eksiklik yalnız yaş/zaman analizlerinden çıkarır (101149 birincil analizde). [DECISION, D-002 — KABUL]
- Zorunlu zaman kontrolleri: zaman penceresi 156 (99/57) ve aynı gün tasarımı 12 gün (39/51) — Bölüm 6.

**Split (D-009):**
- Dış döngü: katılımcı düzeyinde, **etiket × kayıt dönemi (erken / geç / bilinmiyor) × yaş grubu (≤40 / >40 / bilinmiyor)** ile tabakalı 5-fold (her test fold'una zaman penceresinden orantılı katılımcı düşsün diye); ucuz modellerde 5 tekrar (25 fold), fine-tune'da ölçülen maliyete göre 1–3 tekrar.
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

**Her başlık sonucu için zorunlu kontroller:** E1/E2/E3 üç düzeyli değerlendirme ve Bölüm 6.3'teki karar kuralı (Katman 5); yaş-only ve yaş+cinsiyet+sigara baseline'ı; "yaşın ötesinde bilgi" (OOF ses skoru + yaş vs yalnız yaş, ikinci düzey CV); zaman-örtüşen alt kohort; dönem probu (yalnız hastalarda erken vs geç kayıt, ses gömmelerinden); kayıt-koşulu baseline'ı (gürültü tabanı, süre, sessizlik); alt gruplar (yaş grubu, cinsiyet, sigara; hastalarda SFT normal/anormal, GINA kontrol, basamak).

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
- **Checkpoint kuralı (eğitim fazları için):** her epoch `latest.pt` (atomik yazım), en iyi val'de `best.pt`, epoch CSV logu her satırda flush, config + seed + git commit + `pip freeze` deney klasörüne; kaldığı yerden devam. Drive kotası (ücretsiz 15 GB) CNN14'te fold başına ~1 GB tam durum demek → biten fold'larda yalnız tahminler + fp16 en iyi ağırlık tutulur.

---

## 11. Açık sorular ve belirsizlikler

**Ses denetimiyle çözülecek [NEEDS VERIFICATION]:** tüm dosyalar `.m4a` ve 48 kHz mono mu, codec ve bitrate, etkin bant genişliği; **sağlıklıların gerçek kayıt günleri (dosya içi zaman damgası)**; 344 × 7 kaydın varlığı; 101345–101348'in sesi var mı; kopyalar; meta verinin (iOS sürümü, kodlayıcı) ve kayıt koşullarının döneme göre değişip değişmediği; dosya içi tarih ↔ CSV tarihi.

**Dinleyerek teyit edilecek:** slot → görev eşlemesi (her slottan 3 kayıt); 101042 / 101050 aynı kişi mi.

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
10. CSV'ye göre sağlıklılar yalnız 9 Ocak – 2 Nisan 2024 arasında, 15 günde kaydedilmiş (hastalar 67 günde). Sağlıklılar için "veri toplama tarihi" gerçek kayıt günü mü, yoksa sisteme giriş tarihi mi?

**Karar bekleyen yok.** D-015 (ön işleme) ve D-016 (augmentation) ses denetiminden sonra Faz 1'de kesinleşecek.

---

## 12. Faz 0 durum raporu (3. tur)

- **Ne yaptık?** Düzeltilmiş eşlemeyi (2 = araba, 3 = ana), `.m4a` ve 48 kHz bilgisini işledik. "Her tarihte iki grup da var mı?" sorusunu CSV'den gün düzeyinde test ettik. Tarih alanının geçerliliğini SFT tarihi ve yaşla doğruladık. Klinik profilin kayıt dönemini ne kadar ayırdığını ölçtük (EXP-002). Negatif kontrollerin uygulanış biçimini yazdık (D-020).
- **Ne öğrendik?** 67 kayıt gününün 52'sinde yalnız hasta var; 245 hasta (%86) sağlıklısız günlerde kaydedilmiş. Tarih alanı hastalar için gerçek ziyaret günü (%97.5 SFT ile aynı gün). Klinik profil hastalar içinde dönemi AUC 0.855 ile ayırıyor; bunun büyük kısmı tedavi basamağından geliyor, ölçülen spirometri ise ayırmıyor (0.536). Yani sesten dönem tahmini, klinik profil kontrol edilmeden yorumlanamaz.
- **Hangi kararları aldık?** D-002 onaylandı (4 ek sağlıklı dışarıda), D-019 güncellendi (eşleme veri ekibinden, dinleme = teyit), D-018 (48 kHz teyit), D-020 (negatif kontrol tasarımı).
- **Hangi belirsizlikler kaldı?** Bölüm 11; en önemlisi sağlıklıların gerçek kayıt günleri (dosya zaman damgası çözecek) ve geç dönemde kayıt koşullarının değişip değişmediği.
- **Bir sonraki minimum gerekli adım:** repo'yu GitHub'a yükle → `soundData`'yı zip'leyip Drive'a koy → `notebooks/01_data_audit.ipynb`'i CPU runtime'da çalıştır → `audio_audit.md`'yi (özellikle "dosya kayıt zamanı vs CSV tarihi" ve "meta veri ↔ dönem" satırlarını), dinleme teyidini ve kimlik kontrolünü paylaş. Eşleme doğrulanana kadar model eğitimi yok.
