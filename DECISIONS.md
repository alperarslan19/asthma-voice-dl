# DECISIONS.md — Karar günlüğü

**Kurallar**
- Her karar bir ID alır (`D-NNN`) ve **silinmez**. Bir karar değişirse yeni bir karar yazılır ve eskisi `DURUM: değiştirildi → D-0XX` olarak işaretlenir.
- Durumlar: `ÖNERİLDİ` (onay bekliyor) · `KABUL` · `DEĞİŞTİRİLDİ` · `İPTAL`.
- Etiketler: [FACT] [FROM PAPER] [FROM OFFICIAL DOCS] [INFERENCE] [HYPOTHESIS] [DECISION] [NEEDS VERIFICATION]
- Ayrıntılı gerekçeler ve sayılar: `docs/PHASE0_REPORT.md`.

---

## D-001 — Klinik verinin tek doğruluk kaynağı `clinical_data.csv`
DURUM: KABUL · Tarih: 2026-10-08
- **KARAR:** Tüm kod CSV'yi okur; XLSX yalnızca köken (provenance) olarak saklanır.
- **NEDEN:** CSV ve XLSX hücre hücre aynı (348 × 140, 0 farklı hücre). [FACT] İki dosyadan okumak, gelecekte biri güncellenirse sessiz tutarsızlık yaratır.
- **ALTERNATİFLER:** XLSX'i okumak (3 satırlık başlık, tarih tipleri, Excel bağımlılığı → daha kırılgan).
- **RİSK:** Veri ekibi yeni bir dışa aktarım gönderirse hangisinin güncel olduğu karışabilir.
- **BİLİMSEL SONUÇ:** Veri sürümü = CSV'nin sha256'sı (`cab18a35…a3af9b`); her deney bu hash'i kaydeder.
- **UYGULAMA:** `scripts/audit_clinical.py` hash'i ve CSV–XLSX karşılaştırmasını raporlar.

## D-002 — Birincil kohort kuralı: geçerli ses kaydı olan herkes
DURUM: **KABUL** (2026-10-08, 2. tur — ilk öneri "makale kohortu 344" idi, kural genelleştirildi; 3. turda Alper onayladı: 101345–101348 ses kaydı olmadığı için dışarıda)
- **KARAR:** Birincil analize, analiz edilen görev(ler) için geçerli ses kaydı olan **her** katılımcı girer. Demografik/tarih eksikliği yalnız o bilgiyi gerektiren analizlerden (yaş baseline'ı, zaman penceresi, aynı-gün) çıkarır. Veri ekibine göre ses kayıtları 101001–101344 → birincil kohort **284 astım / 60 sağlıklı**; 101345–101348'in ses kaydı yok (ses denetimiyle doğrulanacak). 101149 (demografisi yok) birincil analizde.
- **NEDEN:** Alper'in itirazı: "Demografiyi modelde kullanmayacağız, ek sağlıklıları dışlamak gereksiz." Bu, **model girdisi** açısından doğru. Ama demografi bir ikinci rolde de gerekli: confounder kontrolünde (yaş baseline'ı, zaman analizleri). İki rolü ayırınca kural netleşiyor: girdi rolü ses ister, kontrol rolü demografi ister. Sonuçta kohortu belirleyen ses kaydının varlığı; o da veri ekibine göre 344'te bitiyor. [FROM DATA TEAM + DECISION]
- **ALTERNATİFLER:** (a) Yalnız tam demografili katılımcılar → gereksiz veri kaybı. (b) Makale kohortunu ID aralığıyla sabitlemek → ses varlığından bağımsız, kırılgan bir kural.
- **RİSK:** 101345–101348'in sesi çıkarsa birincil kohort 348 olur ve makaleyle birebir karşılaştırma için ayrıca 344 alt kümesi raporlanır.
- **BİLİMSEL SONUÇ:** Kohort tanımı veri kalitesine bağlı, keyfi değil.
- **UYGULAMA:** `recording_map.csv` → görev bazında `status == OK` olan katılımcılar; `participants.csv` bayrakları yalnız ilgili kontrol analizlerinde.
- **SES DENETİMİ SONUCU (4. tur):** 342 sesli katılımcı = **283 astım / 59 sağlıklı**. Sesi olmayan: 101043 (astım), 101149 (sağlıklı), 101345–101348. 101244'ün slot 4 kaydı yok → görev 4 için 282 / 59. [FACT]

## D-003 — İstatistiksel birim katılımcıdır; split katılımcı tablosunda yapılır
DURUM: KABUL
- **KARAR:** Fold atamaları `participants.csv` üzerinde üretilir; kayıtlar ve segmentler sonra bu atamaya göre eşlenir. Her deney N katılımcı / N kayıt / N segment'i ayrı raporlar. Klinik yorum katılımcı düzeyindedir.
- **NEDEN:** Segment veya kayıt düzeyinde bölmek aynı kişinin sesini hem train hem test'e koyar (katılımcı sızıntısı). Binlerce segment binlerce bağımsız hasta değildir.
- **ALTERNATİFLER:** `GroupKFold` ile kayıt tablosu üzerinden bölmek — eşdeğer ama tabakalama ve denetim katılımcı tablosunda daha şeffaf.
- **RİSK:** Yok (standart doğru uygulama).
- **BİLİMSEL SONUÇ:** Her split dosyası için `train ∩ val ∩ test = ∅` assert'i.
- **UYGULAMA:** `scripts/make_splits.py` (eşleme doğrulandıktan sonra yazılacak).

## D-004 — Zamansal confounder'a karşı zorunlu zaman-kontrollü analiz
DURUM: DEĞİŞTİRİLDİ → D-017 (genişletildi: aynı-gün tasarımı, negatif kontroller, karar kuralı)
- **KARAR:** Her başlık sonucu (a) birincil kohortta ve (b) **zaman-örtüşen alt kohortta** (kayıt tarihi 2024-01-09 … 2024-04-02; 99 astım / 57 sağlıklı) raporlanır. Ek olarak "dönem probu": yalnız hastalarda erken vs geç kayıt, ses gömmelerinden tahmin edilebiliyor mu?
- **NEDEN:** 184/284 hasta son sağlıklı kayıttan sonra kaydedilmiş; tarih tek başına AUC 0.797 ± 0.048; zaman-örtüşen alt kohortta 0.567 ± 0.116. [FACT]
- **ALTERNATİFLER:** (a) Yalnız alt kohortta çalışmak → güç kaybı, makaleyle karşılaştırılamaz. (b) Tarihi kovaryat olarak modele eklemek → sağlıklılarda Nisan sonrası yok; ayarlama ekstrapolasyon olur.
- **RİSK:** Alt kohort küçük (AUC CI ±0.07); iki sonuç çelişirse yorum zorlaşır — ama bu zaten öğrenmek istediğimiz şey.
- **BİLİMSEL SONUÇ:** Yalnız birincil kohortta görülen bir ses başarısı "astım sinyali" olarak yorumlanmaz.
- **UYGULAMA:** `participants.csv` → `in_time_overlap`, `period` sütunları.

## D-005 — Sesi kullanmayan baseline'lar zorunlu karşılaştırıcıdır
DURUM: KABUL
- **KARAR:** Her ses modeli; yaş-only, yaş+cinsiyet+sigara ve kayıt-koşulu (gürültü tabanı, süre, sessizlik) baseline'larıyla **aynı fold'larda** karşılaştırılır; ek olarak "OOF ses skoru + yaş" vs "yalnız yaş" testi.
- **NEDEN:** Yaş-only AUC 0.681; sağlıklılar belirgin olarak genç (p = 1.5×10⁻⁵). [FACT] Referans makaledeki yaş grubu deseni yaş kısayoluyla tutarlı. [INFERENCE]
- **ALTERNATİFLER:** Yaş eşleştirilmiş alt örneklem → sağlıklı sayısı daha da azalır; ikincil analiz olarak tutulabilir.
- **RİSK:** Yaş astımla gerçekten ilişkili olabilir (yaşla artan prevalans); "yaşın ötesinde bilgi" testi bu yüzden "yaşı aş" testinden daha doğru soru.
- **BİLİMSEL SONUÇ:** EXP-001 sonuçları her karşılaştırma tablosunda ilk satırlardır.
- **UYGULAMA:** `scripts/audit_clinical.py` → `EXP-001_confounder_baselines.json`.

## D-006 — RQ9 yeniden tanımlandı: "Ses, yaş/cinsiyet/sigaranın ötesinde bilgi taşıyor mu?"
DURUM: KABUL
- **KARAR:** Astım-vs-sağlıklı modellerinde klinik değişken olarak yalnız yaş, cinsiyet, sigara kullanılabilir. GINA, SFT, ilaç, IgE vb. yalnız hastalara özgü alt grup analizlerinde kullanılır.
- **NEDEN:** Bu değişkenler yalnız hastalarda dolu (sağlıklılarda %0). Modele eklemek etiketi eksiklik deseninden sızdırır. [FACT]
- **ALTERNATİFLER:** Eksikleri "0" ya da ortalamayla doldurmak → eksiklik göstergesi etiketle %100 ilişkili kalır; yanlış.
- **RİSK:** Yok.
- **BİLİMSEL SONUÇ:** RQ9 confounder-ayarlı ve anlamlı bir soruya dönüşür.
- **UYGULAMA:** Füzyon deneylerinde izinli sütun listesi kodda sabit.

## D-007 — Pretrained modellere ham dalga formu + modelin kendi frontend'i
DURUM: KABUL
- **KARAR:** Kendi STFT/log-Mel'imizi hesaplayıp pretrained modellere vermeyiz. Her model resmi SR'sinde ham dalga formu alır.
- **NEDEN:** PANNs (64 bantlı log-Mel + bn0), BEATs (128 bantlı fbank + sabit ort./SD), WavLM (ham dalga formu) temsili kendi içinde üretir. [FROM OFFICIAL DOCS] "STFT" bir adımdır; PANNs/BEATs log-Mel görür.
- **ALTERNATİFLER:** Ortak log-Mel'i tüm modellere zorlamak; ilk katmanı yeniden eğitmek (bkz. rapor 5.1). Ayrıntılı çözme zinciri: D-018.
- **RİSK:** Modeller farklı bant genişliği görür → bkz. D-008.
- **BİLİMSEL SONUÇ:** "Waveform vs STFT" yalnız sıfırdan model için serbest bir sorudur; düşük öncelikli.
- **UYGULAMA:** `configs/model_input_contracts.yaml`; smoke test'te `load_state_dict(strict=True)` + şekil assert'leri.

## D-008 — Sampling rate'ler ve bant genişliği ablasyonu
DURUM: KABUL
- **KARAR:** PANNs CNN10 / CNN14 → 32 kHz; CNN14_16k, BEATs, WavLM → 16 kHz. 48 kHz kaynaktan **tek seferde**, yüksek kaliteli resampler ile, deterministik önbellek. CNN14 (32k) vs CNN14_16k ablasyonu planlanır.
- **NEDEN:** Resmi checkpoint'ler bu SR'lerle eğitildi; CNN10'un 16 kHz sürümü yok. [FROM OFFICIAL DOCS] 32k frontend 14 kHz'e, 16k frontend'ler 8 kHz'e kadar görür → model farkı bant genişliğinden gelebilir. [INFERENCE]
- **ALTERNATİFLER:** Her şeyi 16 kHz'e indirmek (Boll) → CNN10/CNN14'ü yanlış frontend'le kullanmak.
- **RİSK:** Veri ekibi 48 kHz mono bildirdi (2026-10-08); ses denetiminde doğrulanacak. [NEEDS VERIFICATION] AAC kesimi 14 kHz'in altındaysa 32k/16k ablasyonunun anlamı azalır → bant genişliği ölçülüyor.
- **BİLİMSEL SONUÇ:** Aynı mimari, farklı SR karşılaştırması bant genişliği etkisini izole eder.
- **UYGULAMA:** Faz 1'de `scripts/build_audio_cache.py`.

## D-009 — Değerlendirme protokolü: tekrarlı katılımcı-düzeyi CV; ayrı kilitli test seti yok
DURUM: **KABUL** (2026-10-08, Seçenek A onaylandı)
- **KARAR:** Dış döngü: etiket × yaş grubu tabakalı 5-fold, tekrarlı (ucuz modellerde 5×, fine-tune'da 1–3×). İç döngü: yalnız early stopping / eşik / düzenlileştirme. Fine-tune hiperparametreleri önceden sabit (Boll). Fold dosyaları bir kez üretilip tüm deneylerce paylaşılır. **Ayrı kilitli holdout yok (Seçenek A).** Tabakalama: etiket × kayıt dönemi × yaş grubu (D-017).
- **NEDEN:** 60 sağlıklı ile %20 holdout = 12 kişi → AUC CI ±0.12; tek split'in fold'dan fold'a AUC yayılımı EXP-001'de bile 0.51–0.83. [FACT/INFERENCE]
- **ALTERNATİFLER:** Seçenek B: %20 kilitli holdout + kalan %80'de CV (bağımsız ama çok gürültülü son kontrol). Boll'un tek 60/20/20 split'i (split varyansını gizler). Tam nested CV ile her deneyde mimari seçimi (çok pahalı).
- **RİSK:** Seçenek A'da, dış CV sonuçlarına bakarak verdiğimiz kararlar iyimser yanlılık yaratabilir → karşılaştırılacak konfigürasyonlar önceden yazılır ve **hepsi** raporlanır.
- **BİLİMSEL SONUÇ:** Validation (iç) ve test (dış) rolleri hiçbir deneyde karışmaz.
- **UYGULAMA:** `scripts/make_splits.py` → `splits/outer_r{r}.csv` + sha256.

## D-010 — Metrikler
DURUM: KABUL
- **KARAR:** Birincil: katılımcı düzeyi ROC-AUC + dengeli doğruluk. İkincil: duyarlılık, özgüllük, MCC, PR-AUC (her iki sınıf). Doğruluk yalnız çoğunluk tabanıyla (0.826) yan yana.
- **NEDEN:** Yayınlanmış "%82 doğruluk" çoğunluk tabanının altında; dengeli doğruluğu ≈ 0.61. [FACT]
- **ALTERNATİFLER:** F1 (azınlık sınıfa duyarsız olabilir), yalnız AUC (eşik davranışını göstermez).
- **RİSK:** Yok.
- **BİLİMSEL SONUÇ:** Hem Alagöz (AUC) hem Boll (dengeli doğruluk) ile karşılaştırılabilir.
- **UYGULAMA:** `src/.../metrics.py` (Faz 1).

## D-011 — Model karşılaştırma istatistiği
DURUM: KABUL
- **KARAR:** Aynı fold'larda eşleştirilmiş ΔAUC; Nadeau–Bengio düzeltilmiş tekrarlı-CV t-testi; tekrar başına havuzlanmış OOF tahminlerinde DeLong; katılımcı bootstrap CI. Tohumlar arası Fisher birleştirmesi kullanılmaz.
- **NEDEN:** CV fold'ları birbirinden bağımsız değildir (train kümeleri örtüşür) → düz t-testi fazla iyimser; Boll'un aynı test hastalarını paylaşan tohumları Fisher ile birleştirmesi aynı sorunu taşır. [INFERENCE]
- **ALTERNATİFLER:** McNemar (eşikli tahminlerde; tek split için uygun), permütasyon testi (pahalı ama geçerli).
- **RİSK:** Küçük örneklemde gerçek ama küçük farklar "anlamsız" çıkabilir — bu dürüst bir sonuçtur.
- **BİLİMSEL SONUÇ:** Bir iyileşme ancak ΔAUC CI'ı 0'ı dışlıyor ve yön zaman-örtüşen alt kohortta korunuyorsa "destekleniyor" denir.
- **UYGULAMA:** Faz 1 değerlendirme modülü.

## D-012 — Deney merdiveni: önce dondurulmuş gömme + lineer prob, sonra fine-tune
DURUM: KABUL
- **KARAR:** Basamaklar: 0 confounder (✅) → 1 MFCC (sadık + bizim protokol) → 2 dondurulmuş gömme + lineer prob (CNN10, CNN14, CNN14_16k, BEATs, WavLM Base+, Large) → 3 sıfırdan CNN10 → 4 fine-tune (öne çıkan 1–2 aile + PANNs referans) → 5 füzyon, alt grup ve confounder analizleri.
- **NEDEN:** Prob ucuz ve düşük varyanslı; altı backbone'u aynı protokolde karşılaştırır; GPU saatini gerekçeli deneylere ayırır; dondurulmuş-vs-fine-tune ablasyonunu kendiliğinden verir.
- **ALTERNATİFLER:** Doğrudan tüm modelleri fine-tune etmek (40+ GPU saati tahmini, yüksek varyans, Colab oturum riskleri).
- **RİSK:** Fine-tune'u Basamak 2'ye göre seçmek, dış fold sonuçlarına bakarak seçim demektir → bu seçim açıkça yazılır ve seçilmeyenler de raporlanır.
- **BİLİMSEL SONUÇ:** Her deney bir RQ'ya bağlanır (rapor Bölüm 9).
- **UYGULAMA:** EXPERIMENTS.md sırası.

## D-013 — Altyapı: GitHub (kod) + Drive (veri/çıktı) + registry.csv (tek kaynak)
DURUM: KABUL
- **KARAR:** Kod, config, agrega raporlar ve günlükler private GitHub repo'da; ham veri, katılımcı düzeyi tablolar, önbellekler, ağırlıklar ve deney çıktıları Drive'da. `results/registry.csv` tek doğruluk kaynağı; Google Sheets onun görünümü.
- **NEDEN:** Sürüm kontrolü kod için; veri büyük ve hassas. İki elle tutulan tablo zamanla ayrışır.
- **ALTERNATİFLER:** W&B / MLflow (dış hesap, öğrenme maliyeti; ileride eklenebilir).
- **RİSK:** Colab oturum kopmaları → checkpoint kuralları (`docs/COLAB_WORKFLOW.md`).
- **BİLİMSEL SONUÇ:** Her sonuç: git commit + veri hash'i + config + seed ile yeniden üretilebilir.
- **UYGULAMA:** `docs/COLAB_WORKFLOW.md`.

## D-014 — Veri yönetişimi: katılımcı düzeyi hiçbir veri git'e girmez
DURUM: KABUL
- **KARAR:** Repo private; `.gitignore` ses, CSV/XLSX ve `data_*` klasörlerini dışlar. Git'teki raporlar yalnız agrega sayılar ve veri-kalitesi bayrağı taşıyan katılımcı ID'leri içerir; hiçbir katılımcıya özgü klinik değer (yaş, doğum tarihi, ölçüm) yazılmaz.
- **NEDEN:** Ses kayıtları ve klinik veriler kişisel sağlık verisi; yayınlanmış makale verinin KVKK nedeniyle paylaşılmadığını belirtiyor. [FROM PAPER]
- **ALTERNATİFLER:** Yok.
- **RİSK:** Bir notebook çıktısında katılımcı satırları kalırsa git'e sızabilir → notebook çıktıları commit'ten önce temizlenir.
- **BİLİMSEL SONUÇ:** —
- **UYGULAMA:** `.gitignore`, `docs/COLAB_WORKFLOW.md` commit kontrol listesi.

## D-015 — Ön işleme (geçici): Boll'u izle, ses denetiminden sonra kesinleştir
DURUM: **KABUL** (2026-10-08, 7. tur — Alper onayladı; kırpma eşiği `build_audio_cache.py` tasarımında gürültü tabanına göre sabitlenecek) · Harmonizasyon ayrıntısı: D-021
- **KARAR:** Mono → model SR'sine resample → enerji tabanlı kenar kırpma (tepeye göre) → tepe normalizasyonu → 4.0 s pencere / 2.0 s hop, kısa kayıtlar sıfırla doldurulur. Hepsi dosya başına deterministik (split'ten önce güvenli).
- **NEDEN:** Referans makaleyle karşılaştırılabilirlik [FROM PAPER]; kırpma, baş/son sessizlikteki oda gürültüsünü (ortam confounder'ı) azaltır. [INFERENCE]
- **ALTERNATİFLER:** Kırpmasız; tüm kaydı tek girdi olarak vermek (10 s, PANNs ön-eğitimiyle uyumlu); RMS normalizasyonu.
- **RİSK:** Tepe normalizasyonu mutlak ses yüksekliği bilgisini siler (hem olası hastalık sinyali hem mikrofon mesafesi confounder'ı). Kırpma eşiği ses denetimindeki gürültü tabanına göre seçilmeli. [NEEDS VERIFICATION]
- **BİLİMSEL SONUÇ:** —
- **UYGULAMA:** Faz 1.

## D-016 — Augmentation başlangıçta kapalı
DURUM: **KABUL** (2026-10-08, 7. tur — Alper onayladı)
- **KARAR:** İlk deneylerde augmentation yok (PANNs'in model içi SpecAugment'i dahil, bilinçli olarak kapatılır). Augmentation sonradan tek değişkenli ablasyon olarak; pitch shift ve time stretch ayrı ayrı.
- **NEDEN:** Pitch ve konuşma hızı hastalığa dair bilgi taşıyabilir. [HYPOTHESIS] Boll bunu ablasyonsuz uyguladı. [FROM PAPER]
- **ALTERNATİFLER:** Boll'un tam augmentation paketi.
- **RİSK:** Augmentation'sız fine-tune'da aşırı uyum artabilir → early stopping ve dondurulmuş-prob karşılaştırması bunu görünür kılar.
- **BİLİMSEL SONUÇ:** —
- **UYGULAMA:** Faz 1 eğitim config'i: `augment: none`, `panns_specaugment: false`.

## D-017 — Zamansal confounder stratejisi: ölç → tasarımla kontrol et → yanlışla → (gerekirse) azalt → önceden yazılmış kurala göre yorumla
DURUM: KABUL, **kısmen DEĞİŞTİRİLDİ → D-023** (5. tur: "başlık sayısı E2" kuralı geçersiz; E2 saat açısından dengesiz) · Tarih: 2026-10-08 · D-004'ün yerine geçer · Ayrıntı: rapor Bölüm 6
- **KARAR:**
  1. *Ölç:* ses denetimi meta veriyi ve kayıt koşullarını yalnız hastalarda erken vs geç dönemle karşılaştırır.
  2. *Tasarımla kontrol:* her model aynı OOF tahminlerinden üç sayıyla raporlanır — E1 tam kohort (284/60), **E2 zaman penceresi (99/57, başlık sayısı)**, E3 aynı-gün tabakalı AUC (12 gün, 39/51, 173 çift); duyarlılık: E4 pencerede yeniden eğitim.
  3. *Yanlışla:* N1 dönem probu (ses vs klinik profil), N2 skor–tarih ilişkisi, N3 arka plan (sessizlik) kontrolü, N4 tarih/ID referans çizgileri.
  4. *Azalt (koşullu):* kırpma, train-fold sessizlik havuzuyla karıştırma, pencerede eğitim; domain-adversarial yalnız uzantı olarak.
  5. *Karar kuralı:* "Ses astım bilgisi taşıyor" iddiası ancak E2 > yaş-only (aynı katılımcılar, eşleştirilmiş ΔAUC CI 0'ı dışlıyor) VE E3 > 0.5 aynı yönde VE (veri izin verirse) arka plan kontrolü şans düzeyinde ise yapılır.
- **NEDEN:** Tarih tek başına AUC 0.797, ID tek başına 0.800; zaman penceresinde tarih 0.567. Geç dönemde hiç sağlıklı yok. Geç dönem hastalarının klinik profili de farklı (tedavi basamağı p ≈ 3×10⁻¹², SFT p = 0.001, sigara p = 0.004) → dönem sinyali hem artefakt hem gerçek hastalık farkı olabilir; ayırmak için klinik-profil-kontrollü prob gerekli. [FACT]
- **ALTERNATİFLER:** Yalnız zaman penceresiyle çalışmak (güç kaybı, makaleyle karşılaştırılamaz); tarihi kovaryat yapmak (sağlıklılarda Nisan sonrası değer yok → ekstrapolasyon); domain-adversarial eğitim (karmaşık, gerçek sinyali silebilir).
- **RİSK:** E2 kesinliği ±0.07, E3 ±0.11 (simülasyon, gerçek AUC 0.80'de) → küçük farklar çözülemez. Geç dönem koşullarında özgüllük **hiçbir analizle** ölçülemez.
- **BİLİMSEL SONUÇ:** Başlık iddiası yalnız zaman-kontrollü kanıta dayanır; tam kohort sonucu karşılaştırma ve üst sınır olarak raporlanır. Kalıcı çözüm için veri ekibine yeni sağlıklı kayıt önerildi (rapor 6.6).
- **UYGULAMA:** `participants.csv` → `period`, `in_time_overlap`, `collection_day`, `same_day_as_other_group`; ses denetimi meta-veri ↔ dönem tabloları; Faz 1 değerlendirme modülü E1–E3'ü her deney için otomatik üretir.

## D-018 — Ses formatı ve çözme zinciri
DURUM: KABUL (ses denetimi sonrası doğrulanacak) · Tarih: 2026-10-08
- **KARAR:** Her `.m4a` dosyası ffmpeg ile **bir kez** çözülür (yalnız ilk ses akışı, meta veri atılır) → float32 mono 48 kHz → `scipy.signal.resample_poly` ile 32 kHz (2/3) ve 16 kHz (1/3) → float32 önbellek (Drive). Pretrained modellere bu dalga formu verilir; STFT/log-Mel/fbank'i model yapar (D-007).
- **NEDEN:** Kapsayıcı yalnızca zarftır; içindeki AAC kayıplıdır ama ön-eğitim verileri (AudioSet ← YouTube) de sıkıştırılmıştır [FACT]. Kritik olan tekdüzeliktir. 48 kHz'den hedef SR'lere tam oranlar var. Yeniden kayıplı kodlama yeni artefakt ekler; int16'ya çevirmek AAC çözümünün ±1'i aşan tepelerini kırpabilir.
- **ALTERNATİFLER:** `torchaudio.load` / `librosa.load` ile her epoch'ta yeniden çözmek (arka uç bağımlı, yavaş, sürüm farkları); 16-bit WAV önbellek (kırpma riski).
- **RİSK:** 48 kHz veri ekibince teyit edildi; tüm dosyalar `.m4a` bildirildi (denetim doğrulayacak). Codec/bitrate/bant genişliği/iOS sürümü dönemle değiştiyse zamansal confounder'ın akustik yolu olur (D-017 Katman 1). Dosyalar GPS konumu taşıyabilir.
- **BİLİMSEL SONUÇ:** Format uygun; koşul, tekdüzeliğin ses denetimiyle gösterilmesi.
- **UYGULAMA:** `scripts/audit_audio.py` (format, bant genişliği, meta veri, konum var/yok); Faz 1 `scripts/build_audio_cache.py`.

## D-019 — Slot → görev eşlemesi veri ekibinden alınır, kısa dinlemeyle teyit edilir
DURUM: KABUL — **teyit edildi** (4. tur: 21 kayıt dinlendi, uyumlu) · Güncellendi: 2026-10-08
- **KARAR:** Dosya adı `<ID>_<slot>.m4a`; **1 = aaa, 2 = araba, 3 = ana, 4 = ordu, 5 = gelecek, 6 = titiz, 7 = ünlem**. İlk bildirimdeki slot 2/3 çelişkisi veri ekibinin düzeltmesiyle giderildi. Her slottan 3 kayıt dinlenerek teyit edilir; teyit yapılmadan `configs/slot_task_map.yaml` `confirmed_by_listening` olmaz ve görev bazlı analiz başlamaz.
- **NEDEN:** Yanlış eşleme RQ4/RQ5'i ve makaleyle görev bazlı karşılaştırmayı sessizce bozar; teyit ~5 dk. Kaynağın kendisi bir kez çelişkili bilgi verdiği için ikinci, bağımsız bir kontrol değerli. [DECISION]
- **ALTERNATİFLER:** Teyitsiz kabul (hızlı ama tek kaynağa dayanır); Whisper ile otomatik doğrulama (ek bağımlılık; dinleme yeterli).
- **RİSK:** Dinlenen kayıtlar istisnaysa yanlış genelleme → uyuşmazlık görülürse daha fazla dinlenir.
- **BİLİMSEL SONUÇ:** —
- **UYGULAMA:** `configs/slot_task_map.yaml` (durum: `FROM_DATA_TEAM_PENDING_LISTENING_CHECK`), `notebooks/01_data_audit.ipynb` Bölüm 3.

## D-020 — Negatif kontrollerin uygulanış biçimi (D-017 N1 ve N3)
DURUM: KABUL · Tarih: 2026-10-08 · Ayrıntı: rapor Bölüm 6.3, EXP-002
- **KARAR:**
  - **N1 (dönem probu)** yalnız hastalarda yapılır ve iki tasarımla yorumlanır: (a) *artımlı* — "klinik profil" vs "klinik profil + iç-CV'de üretilmiş ses dönem skoru", aynı fold'larda eşleştirilmiş ΔAUC; (b) *tabakalı* — yalnız tedavi basamağı 4 hastaları (81 erken / 66 geç), ses-only ve klinik-only AUC yan yana. "Kayıt koşulları değişmiş" sonucu yalnız (a)'da ΔAUC'nin CI'ı 0'ı dışlıyorsa çıkarılır.
  - **N3 (arka plan)** yalnız nefes enerjisinin altındaki en düşük enerjili karelerden çıkarılan basit spektral özetle yapılır (lojistik regresyon; hedefler: zaman penceresinde etiket, hastalarda dönem). Pozitif sonuç dinlenerek "oda mı, nefes mi?" diye ayrılmadan "sızıntı" olarak yorumlanmaz.
- **NEDEN:** EXP-002'ye göre klinik profil tek başına dönemi AUC 0.855 ile ayırıyor (tedavi basamağı 0.744, ölçülen spirometri 0.536) → sesten dönem tahmini, profil kontrol edilmeden "kayıt artefaktı" diye okunamaz. [FACT] Kelime aralarındaki duyulabilir soluma ya da hırıltı gerçek bir hastalık işareti olabilir → arka plan kontrolü onu "sızıntı" sanmamalı. [INFERENCE]
- **ALTERNATİFLER:** Profil kontrolsüz dönem probu (yanlış pozitif "artefakt" alarmı); klinik profile göre eşleştirme (örneklem daha da küçülür); arka plan için derin model (gereksiz karmaşık; basit özet yeterli).
- **RİSK:** Kesinlik sınırlı (tüm hastalarda ±0.06–0.07, basamak 4'te ±0.09); kategorik kodlama dönemler arasında değişmişse "profil" kısmen kodlama farkıdır (veri ekibine soruldu, rapor Bölüm 11 S8).
- **BİLİMSEL SONUÇ:** Dönem sinyalinin iki kaynağı (kayıt koşulları, hasta profili) ayrıştırılabilir hale gelir.
- **UYGULAMA:** Faz 2'de dondurulmuş gömmelerle (`21_frozen_embeddings.ipynb`); referans çizgisi `reports/clinical_audit/EXP-002_clinical_period_probe.json`.

## D-021 — Harmonizasyon politikası: teknik farkları eşitle, sesi "temizleme", etkisini ölç
DURUM: KABUL — **kesim 11.0 kHz** (5. tur: en düşük kodlayıcı kesimi Apple 11.27 / FFmpeg 11.58 kHz; `scripts/analyze_recording_context.py`) · Tarih: 2026-10-08 · Ayrıntı: rapor Bölüm 6.8
- **KARAR:** Tüm dosyalara (train ve test, split'ten önce, deterministik) şu işlemler uygulanır:
  1. kayıpsız çözme (meta veri atılır);
  2. ortak SR (44.1 kHz'lik tek dosya dahil);
  3. 32 kHz yolu için en düşük kodlayıcı kesiminin altında alçak geçiren filtre (~11 kHz; kesin değer `audio_inventory.csv`'deki zincir başına en düşük kesimden);
  4. kenar sessizliği kırpma;
  5. tepe normalizasyonu.
  
  **Yapılmayacaklar:** gürültü giderme, DAW üzerinden yeniden export, kayıplı yeniden kodlama, tüm veriye sabit gürültü ekleme. Gürültü yalnız eğitimde, rastgele ve train-fold havuzundan karıştırılabilir; bu bir ablasyondur (D-016).
- **NEDEN:**
  - Meta veri (iOS sürümü, kodlayıcı adı) modele hiç ulaşmaz. Örneklerdeki teknik izler (bant genişliği 11–24 kHz, SR) ise bant sınırlama ve resample ile eşitlenebilir. [FACT/INFERENCE]
  - Gürültü giderme nefesli ses, türbülans ve hırıltı gibi olası hastalık işaretlerini silebilir. Ayrıca girişe bağlı yeni artefaktlar üretir ve geri dönüşsüzdür. [INFERENCE]
  - Sabit gürültü ekleme yalnız maskeler: herkes için SNR kaybettirir ve konuşma bandındaki farklara dokunmaz.
  - Tarih, günün saati ve hasta profili sesin kendisindedir; hiçbir ses işleme onları gideremez (bunlar D-017/D-022 ile tasarım düzeyinde ele alınır).
- **ALTERNATİFLER:**
  - Alper'in önerisi: gürültü giderme + ortak arka plan + ortak DAW export.
  - Yalnız ortak arka plan ekleme.
  - Hiç harmonizasyon yapmama.
  - Yalnız 16 kHz modeller kullanma (bant farkını kendiliğinden gizler ama PANNs-32k karşılaştırmasını kaybettirir).
- **RİSK:**
  - ~11 kHz kesim, 32 kHz modelin 11–14 kHz bandını herkes için boşaltır. Bu, ön-eğitim dağılımından bir sapmadır ama tüm dosyalarda aynıdır.
  - Harmonizasyon, ölçmediğimiz teknik farkları kaçırabilir → doğrulama şart.
- **BİLİMSEL SONUÇ:** "Temizledik" iddiası ancak ölçümle desteklenirse yapılır. **Zincir probu** kullanılacak: Apple / FFmpeg zinciri (bilinen, teknik, iki grupta da var) ses gömmelerinden tahmin edilebiliyor mu? Harmonizasyon öncesi ve sonrası ölçülür, aynı ölçüm hastalarda dönem probu (D-020) için de yapılır. Harmonize ve ham ses karşılaştırması RQ7 kapsamında bir ablasyondur.
- **UYGULAMA:** Faz 1 `scripts/build_audio_cache.py` (önbellekte harmonizasyon parametreleri + sha256); Faz 2 zincir probu.

## D-022 — Günün saati confounder'ı
DURUM: KABUL · Güncellendi (5. tur): saat 340/342 katılımcıda biliniyor (seans başlangıcı); astım medyan 11:05, sağlıklı 15:10; saat-only AUC 0.841, saat + tarih 0.931 (EXP-003). Ölçüm ve test D-023'e taşındı. · Tarih: 2026-10-08
- **KARAR:** Günün saati tasarım kaynaklı bir confounder olarak ele alınır:
  1. **EXP-003:** saat-only ve tarih+saat baseline'ları (yalnız zaman damgalı Apple dosyaları olan katılımcılarda).
  2. Sayılar yeterliyse E3'ün "aynı gün + aynı yarım gün (sabah/öğleden sonra)" sürümü.
  3. Saate göre alt grup performansı.
  4. Saat bilgisi olmayan (FFmpeg zinciri) katılımcılar ayrı raporlanır.
- **NEDEN:** Zaman damgalı dosyalarda saat → etiket AUC'si 0.13–0.17 (Bonferroni şans sınırı ≈ 0.33). Hastalar belirgin biçimde daha erken saatte kaydedilmiş. Saat sesi hem fizyolojik olarak (sabah sesi, ses yorgunluğu, astımın günlük ritmi) hem ortam olarak etkileyebilir. [FACT + INFERENCE]
- **ALTERNATİFLER:**
  - Saati modele kovaryat yapmak: saat etiketle bu kadar iç içeyken ekstrapolasyon olur ve FFmpeg zincirinde saat yok.
  - Görmezden gelmek: kabul edilemez.
- **RİSK:** Saat yalnız dosyaların ~%73'ünde var. Saat ile hastalık fizyolojisi (sabah semptomları) gerçekten ilişkili olabilir; bu durumda saat kısmen bir "aracı" (mediator) olur ve onu kontrol etmek gerçek sinyali de azaltır. Bu, yorumda açıkça yazılacak.
- **BİLİMSEL SONUÇ:** Başlık sonucu E2'dir (D-017). Saat kontrolleri, E2 sonucunun ne kadarının saatle açıklanabileceğini gösterir.
- **UYGULAMA:** `recording_map.csv`'deki `creation_local` → katılımcı düzeyinde ilk kaydın yerel saati.

## D-023 — Birincil soru ve test revize edildi: ses, kayıt bağlamı ve yaşın ötesinde bilgi taşıyor mu?
DURUM: KABUL (birincil soru) · **test ayrıntıları kısmen DEĞİŞTİRİLDİ → D-025** (6. tur: ham E2h AUC'sinin şans çizgisi 0.5 değil; kuadratik bağlam modeli yanlış pozitif üretiyor; ΔAUC tavanda güçsüz) · Tarih: 2026-10-08 (5. tur) · **Ses modeli sonuçları görülmeden önce alındı** · D-017'deki "başlık sayısı E2" kuralının yerine geçer · Ayrıntı: rapor Bölüm 6.9
- **KARAR:**
  - **T1 (birincil):** ikinci düzey tekrarlı CV'de "yaş + saat (+ saat²) + tarih" baseline'ına OOF ses skoru eklenir. Raporlananlar: eşleştirilmiş ΔAUC ve katılımcı-bootstrap CI; ses skorunun bağlam-ayarlı odds oranı ve CI'ı.
  - **Destekleyici:** E2h, yani zaman penceresi + aynı 1 saatlik başlangıç dilimi içinde tabakalı AUC.
  - **Her zaman raporlanır:** E1, E1h, E2, E3, **denge tablosuyla** (tasarım içinde saat / tarih / yaş tabakalı AUC'leri).
  - **İddia koşulu:** T1'de ΔAUC CI 0'ı **ve** odds oranı CI'ı 1'i dışlıyor **ve** E2h > 0.5 ile aynı yönde. Aksi halde "bağlamdan bağımsız bir ses sinyali gösterilemedi" raporlanır.
- **NEDEN:**
  - EXP-003'e göre sesi kullanmadan saat + tarih etiketi AUC 0.931 ± 0.032 ile ayırıyor.
  - Zaman penceresinde saat 0.873 (E2 saat dengesi 0.125); aynı-gün tasarımında saat dengesi 0.103.
  - Hiçbir tabakalı tasarım tarih, saat ve yaşı birlikte dengelemiyor; en iyisi E2h (saat 0.378, tarih 0.375, yaş 0.704). [FACT]
  - Model temelli artımlı test tüm katılımcıları kullanır ve soruyu doğru sorar: bağlam *biliniyorken* ses ne ekliyor?
- **ALTERNATİFLER:**
  - E2'yi başlık olarak tutmak: saat confounding'i yok sayar.
  - Yalnız E2h kullanmak: ±0.125 kesinlik; tek başına zayıf.
  - Eşleştirme / ağırlıklandırma (propensity): örtüşme çok sınırlı, ağırlıklar uçlaşır.
  - Bağlamı modele girdi yapmak: hedef ses, bağlam değil.
- **RİSK:**
  - Baseline tavana yakın (0.93) → ΔAUC doğal olarak küçük kalır; odds oranı bu yüzden de raporlanır.
  - Saat kısmen gerçek fizyoloji olabilir (sabah sesi, astım ritmi, bronkodilatör sonrası kayıt). Saat-ayarlı sonuçlar muhafazakârdır ve böyle yorumlanacak.
  - Saat/tarih değişkenlerimizin yakalamadığı bağlam farkları (dakika düzeyi, operatör) kalabilir.
- **BİLİMSEL SONUÇ:** Proje "ses astımı ne kadar iyi tespit ediyor" sorusundan "bağlamdan bağımsız akustik astım bilgisi var mı" sorusuna geçer. Bu, yayınlanmış çalışmanın cevaplamadığı ve bu veri setinde cevaplanabilecek en dürüst sorudur.
- **UYGULAMA:** `participant_context.csv` (Drive): `start_hour`, `recording_date`; Faz 1 değerlendirme modülü T1, E-tasarımları ve denge tablosunu her deney için otomatik üretir.

## D-024 — Kayıt tarihinin kaynağı: dosya zaman damgası öncelikli
DURUM: KABUL · Tarih: 2026-10-08 (5. tur)
- **KARAR:** Zaman analizlerinde `recording_date` = katılımcının en erken zaman damgalı dosyasının yerel tarihi; zaman damgası yoksa CSV "veri toplama tarihi". Dönem (erken/geç) ve zaman penceresi bu tarihle yeniden hesaplanır (Faz 1, `make_splits.py` öncesi).
- **NEDEN:** Sesin gerçekte kaydedildiği an, ses modelini ilgilendiren andır. Apple dosyalarında dosya tarihi CSV ile %98.7 aynı gün. 4 hastada fark var (−30, +1, +5, +182 gün); 101018'in CSV tarihi yok ama dosyasında var. [FACT]
- **ALTERNATİFLER:** CSV tarihi öncelikli (giriş hatalarını taşır); farklı olanları dışlamak (gereksiz veri kaybı).
- **RİSK:** Dosya zaman damgası cihaz saatine bağlı; 4 farkın nedeni veri ekibine soruldu (rapor S14).
- **BİLİMSEL SONUÇ:** —
- **UYGULAMA:** `scripts/analyze_recording_context.py` → `participant_context.csv` (`recording_date`, `recording_date_source`).

## D-025 — Birincil test spesifikasyonu revize edildi (T1, E2h, Spisak testleri)
DURUM: ÖNERİLDİ — **değerlendirme aşamasına ertelendi (D-028)**; model geliştirme sırasında uygulanmaz · Tarih: 2026-10-08 (6. tur) · **Ses modeli sonuçları görülmeden** · D-023'ün test ayrıntılarının yerine geçer (birincil soru aynı) · Ayrıntı: `docs/CONFOUND_CONTROL_DESIGN.md` Bölüm 2–4, 6
- **KARAR:**
  1. **T1 (birincil):** tek lojistik regresyon `y ~ spline(saat, 4 df) + spline(tarih, 4 df) + yaş + ŝ`.
     - ŝ = katılımcının dış-fold OOF ses logit'i (tekrarlar boyunca ortalama).
     - **Karar istatistiği:** ŝ katsayısının olabilirlik oranı testi; SD başına OR ve katılımcı-bootstrap %95 CI.
     - Duyarlılık analizleri: gün-kümeli SE; daha esnek bağlam modeli (spline 6 df).
     - İkinci düzey CV ΔAUC yalnız **betimsel**.
     - Kuadratik saat + doğrusal tarih bağlam modeli (EXP-003 biçimi) T1'de **kullanılmaz**.
  2. **E2h:** ham tabakalı AUC 0.5'e karşı test edilmez.
     - Ham AUC, yaş-only, bağlam-only ve yaş + bağlam referans çizgileriyle birlikte raporlanır.
     - **Test:** E2h içinde 1 saatlik dilimlere göre koşullu lojistik regresyon `y ~ ŝ + yaş + saat + tarih` (E2h-KLR).
  3. **Spisak testleri** (`mlconfound`, OOF skorlar üzerinde, 1 000 koşullu permütasyon):
     - *Tam test* (H0: ŝ ⊥ Y | C) → T1'in parametrik olmayan teyidi.
     - *Kısmi test* (H0: ŝ ⊥ C | Y) → model bağlamı kullanıyor mu?
     - C çok boyutlu verilemiyorsa, çapraz-fit edilmiş bağlam eğilim skoru (`logit p̂(Y | yaş, saat, tarih)`) kullanılır.
  4. **Onaylayıcı aile:** katılımcı düzeyinde toplanmış 6 dondurulmuş backbone için T1, Holm düzeltmesiyle. Görev bazlı ve tasarım bazlı tüm analizler keşifsel.
  5. **İddia C1** ("ses, ölçülen bağlam ve yaşın ötesinde astım bilgisi taşıyor") için hepsi gerekir:
     - Holm-düzeltmeli T1 p < 0.05;
     - OR CI'ı 1'i dışlıyor;
     - gün-kümeli duyarlılıkta ve spline 6 df'de yön korunuyor;
     - E2h-KLR OR'u aynı yönde (CI'ın 1'i dışlaması şart değil);
     - N3 etkiyi yeniden üretmiyor;
     - N5 bilgiliyse anlamlı değil.
     
     **C2** (model bağlamı kullanıyor) ve **C3** (fine-tuning bağlamdan bağımsız bilgiyi artırdı): tasarım belgesi Bölüm 6.
- **NEDEN:**
  - **Bölüm F (rapor):** yalnız yaşı kodlayan bir model E2h'de 0.703 alır; tam kohortta bağlamı öğrenmiş bir model 0.351 alır. Ham E2h AUC'sinin şans çizgisi kestirmenin türüne göre 0.35–0.70 arasında, dolayısıyla tanımsız. [FACT]
  - **SIM-001 (500 simülasyon, gerçek bağlam):**
    - Ses yalnız "sabah mı / geç dönem mi" basamaklarını kodladığında kuadratik bağlam modeliyle T1 yanlış pozitif oranı **0.216**. Spline modelle 0.060, E2h-KLR ile 0.060.
    - Gerçek sinyal saptanabilir düzeydeyken bile (aynı-bağlam AUC 0.70, güç ~%84) ortalama ΔAUC yalnız ≈ 0.004. ΔAUC CI ölçütü bu yüzden neredeyse güçsüz.
    - Tam kohort T1'in gücü E2h-KLR'ninkine yakın: bilgi örtüşme bölgesinde.
  - Dinga ve ark. 2020: confound kontrolü model tahminleri üzerinde yapılmalı. Spisak 2022: tam ve kısmi confounder testleri. [FROM PAPER]
- **ALTERNATİFLER:**
  - D-023'ü olduğu gibi tutmak: yanlış pozitif riski ve güçsüz ΔAUC ölçütü.
  - İkinci düzey CV ΔAUC'yi karar istatistiği yapmak: tavanda güçsüz.
  - Daha esnek bağlam modeli (GBM): 340 kişide aşırı uyum, çıkarım zor.
  - Yalnız Spisak testleri: koşullu dağılım modeline bağımlı, etki büyüklüğü vermez.
- **RİSK:**
  - Ölçülmemiş, güne özgü bağlam varsa (SIM-001 S3) yanlış pozitif ~%12'ye çıkıyor ve gün-kümeli SE bunu düzeltmiyor (65 günün 53'ü tek etiketli).
  - Güç iyimser tahmin edildi: gerçek OOF skoru daha gürültülü.
  - %80 güce ancak aynı-bağlam AUC'si ≈ 0.69'da (α = 0.05), Holm düzeyinde (≈ 0.01) ≈ 0.74'te ulaşılıyor; daha küçük gerçek etkiler kaçırılabilir.
- **BİLİMSEL SONUÇ:** Birincil test artık hem yanlış belirtilmiş bağlam modeline hem tavan etkisine karşı korunuyor. "Olumsuz" sonuç da raporlanabilir hâle geliyor: minimum saptanabilir etki önceden biliniyor.
- **UYGULAMA:**
  - `scripts/analyze_recording_context.py` (Bölüm F referans çizgileri) ve `scripts/simulate_t1_power.py` (SIM-001).
  - Faz 1 `scripts/evaluate_context.py` (L1–L8 lens seti).
  - `requirements-colab.txt`'ye `statsmodels`, `patsy`, `mlconfound` eklenir.

## D-026 — Confounder kontrol yığını: hangi yöntem ne işe yarar
DURUM: ÖNERİLDİ — **değerlendirme aşamasına ertelendi (D-028)**; model geliştirme sırasında uygulanmaz · Tarih: 2026-10-08 (6. tur) · Ayrıntı: `docs/CONFOUND_CONTROL_DESIGN.md` Bölüm 0, 3, 4, 7
- **KARAR:**
  1. **Kanıt yalnız değerlendirme katmanından gelir:** T1, Spisak, E2h-KLR, karşıt-hücre tablosu, N1–N5, zincir kontrolleri. Her model aynı lens setiyle (L1–L8) raporlanır.
  2. **Hijyen her zaman uygulanır**, ama "confounding kontrolü" diye raporlanmaz:
     - katılımcı düzeyinde split; tabakalar etiket × dönem × sabah/öğleden sonra, yaş grubu yalnız her hücrede ≥ 5 kişi kalıyorsa;
     - D-021 harmonizasyonu.
  3. **Azaltma yöntemleri koşulludur** (G1 pozitif ya da kısmi test anlamlıysa):
     - bağlam-dengeli yeniden ağırlıklandırma: ağırlık ∝ 1/p̂(Y|C), train fold'unda çapraz-fit, %1–99'da kırpılır, etkin n raporlanır;
     - hasta-içi, fold-içi doğrusal residualization: yalnız dondurulmuş gömmelerde.
  4. **Yapılmayacaklar:**
     - gürültü giderme, sabit gürültü ekleme, DAW'dan export;
     - tüm veride (CV öncesi) confound regresyonu;
     - marjinal gradient reversal;
     - saat veya tarihi modele girdi olarak vermek.
     
     Etiket-koşullu adversarial yalnız araştırma uzantısıdır (planlanmadı).
  5. **Yeni negatif kontrol N5 (bağlam-vekili):** gömmeden yalnız hastalarda sabah/öğleden sonra tahmin eden modelin OOF skoru T1'e konur. Bilgi eklememeli.
  6. **Gün-gruplu CV**, dondurulmuş problarda duyarlılık analizi olarak.
- **NEDEN:**
  - EXP-004: etiket içinde kaba akustik saat izi yok (hastalar 0.549 ± 0.067) → harmonizasyon saat confound'una dokunmaz.
  - Codec zinciri etiketle ilişkili değil (p = 0.22).
  - EXP-004b: sabah ve öğleden sonra hastaları klinik olarak benzer (0.528 ± 0.071) → etiket içi saat kontrastı temiz bir kontroldür.
  - 65 kayıt gününün 53'ü tek etiketli → güne özgü kestirme riski. [FACT]
  - Literatür:
    - Snoek 2019: tüm veride confound regresyonu negatif yanlılık; fold içinde yapılınca sorun yok; alt örneklemeyle karşı-dengeleme pozitif yanlılık.
    - Dinga 2020: girdi düzeyinde ayarlama yetersiz.
    - Zhao 2020: düşman yalnız etiket-koşullu alt kohortta eğitilmeli.
    - Elazar & Goldberg 2018: adversarial kaldırma eksik kalır.
    - Coppock 2024: rastgele split 0.846 → eşleştirilmiş test ~0.62. [FROM PAPER]
- **ALTERNATİFLER:**
  - Önce azaltma, sonra değerlendirme: azaltmanın işe yaradığı gösterilemez.
  - Adversarial-merkezli tasarım: küçük n, kararsız, kaldırma eksik.
  - Yalnız ses işleme: saat ve tarih yoluna dokunmaz.
  - Yalnız E2h: düşük güç, şans çizgisi belirsiz.
- **RİSK:**
  - Pozitiflik: sabah kaydedilmiş sağlıklı yalnız 12; geç dönemde hiç sağlıklı yok → bilgi örtüşme bölgesine sınırlı.
  - Yeniden ağırlıklandırmada ağırlıklar uçlaşabilir.
  - Hasta-içi residualization, bağlam etkisinin sağlıklılarda da aynı olduğunu varsayar. [HYPOTHESIS]
  - S13 (bronkodilatör) prosedür yolu hiçbir yöntemle ayrılamaz.
- **BİLİMSEL SONUÇ:** Yöntemler "kanıt üreten", "hijyen" ve "koşullu azaltma" olarak ayrılır. Azaltma yöntemlerinin başarısı da aynı lenslerle ölçülür.
- **UYGULAMA:** EXP-020 (G1), EXP-021, EXP-022; `docs/CONFOUND_CONTROL_DESIGN.md`.

## D-027 — Confounding altında fine-tuning protokolü
DURUM: ÖNERİLDİ — **değerlendirme aşamasına ertelendi (D-028)**; model geliştirme sırasında uygulanmaz · Tarih: 2026-10-08 (6. tur) · Ayrıntı: `docs/CONFOUND_CONTROL_DESIGN.md` Bölüm 6 (EXP-030)
- **KARAR:**
  - **Backbone seçimi:** EXP-021'de **T1'e göre** ilk 1–2 backbone. E1'e göre seçilmez; eşitlikte düşük kapasiteli olan seçilir.
  - **Eğitim kapsamı:** kısmi fine-tune (son blok(lar) + baş). Blok sayısı mimari başına önceden yazılır.
  - **Hiperparametreler:** sabit Boll değerleri (Adam, lr 1e-4, wd 1e-4, batch 16).
  - **Epoch:** **sabit sayıda**. Bağlam-confounded doğrulama AUC'siyle early stopping veya checkpoint seçimi yapılmaz. Epoch sayısı fold-0'ın iç train/val kayıp eğrisinden bir kez belirlenir ve dondurulur.
  - **Kayıp:** sınıf-dengeli.
  - **Veri:** tüm kayıtlar tek modelde; katılımcı skoru = kayıt logit'lerinin ortalaması.
  - **Kollar:**
    - Kol A: standart.
    - Kol B: + bağlam-dengeli ağırlık. Yalnız G1+ ise ya da A'da kısmi test anlamlıysa çalıştırılır.
  - **Tekrar:** 3 tekrar × 5 fold. Tekrar r, EXP-021'in r. split'ini ve r. tohumunu kullanır.
  - **Değerlendirme:** aynı lens seti. Fine-tune vs dondurulmuş karşılaştırması aynı katılımcılarda eşleştirilmiş yapılır: T1 sapma azalması farkı (bootstrap) ve kısmi test.
  - **Kayıt ve güvenlik:** smoke test ve checkpoint kuralları geçerli.
- **NEDEN:**
  - Confounded bir metrikle seçim veya early stopping kestirmeyi ödüllendirir.
  - İç doğrulama fold'unda ~9 sağlıklı var → eşik ve early stopping çok gürültülü.
  - Küçük n'de tam fine-tune aşırı uyar.
  - Tohum varyansı büyük olabilir.
  - Kilitli test seti yok (D-009 A) → fine-tune "seçim sonrası" keşifsel analizdir.
- **ALTERNATİFLER:**
  - Doğrulama AUC'siyle early stopping (Boll): kestirmeyi ödüllendirir.
  - Bağlam-ağırlıklı doğrulama kaybıyla early stopping: ~9 sağlıklı ve uç ağırlıklar → çok gürültülü.
  - Tam fine-tune: aşırı uyum.
  - Görev başına fine-tune: 7 kat maliyet.
  - E2h'de fine-tune: fold başına ~90 kişi.
- **RİSK:**
  - Sabit epoch sayısı fold-0'a özgü olabilir; az veya aşırı uyum riski.
  - Bütçe ~10–20 T4 saati [INFERENCE; smoke testte ölçülecek].
  - Beklenen olumsuz sonuç: fine-tuning E1'i artırırken T1'i artırmayabilir ve kısmi confounding'i artırabilir. [HYPOTHESIS]
- **BİLİMSEL SONUÇ:** Fine-tuning'in katkısı "E1 arttı mı?" ile değil, **"bağlamdan bağımsız bilgi arttı mı?"** ile ölçülür. Olumsuz sonuç da raporlanır.
- **UYGULAMA:** EXP-030, Faz 3 `scripts/train_finetune.py`.

## D-028 — Kayıt bağlamı confounder'ı ayrı bir araştırma başlığıdır; değerlendirme model geliştirme bitince yapılır
DURUM: KABUL · Tarih: 2026-10-08 (7. tur, Alper'in yönlendirmesi ve onayı) · D-023'ün birincil sorusunu değiştirmez; ne zaman ve nasıl uygulanacağını belirler
- **KARAR:**
  1. **Ana proje planı değişmez** (D-012 merdiveni: MFCC → dondurulmuş gömme + lineer prob → sıfırdan CNN10 → ham dalga formundan uçtan uca fine-tune). Geliştirme sırasında ek kontrol deneyi veya azaltma yöntemi uygulanmaz. Bunlar ertelenir:
     - bağlam probları (EXP-020) ve azaltma denemeleri (EXP-022);
     - yeniden ağırlıklandırma, adversarial eğitim, gün-gruplu CV;
     - D-025/026/027 önerileri.
  2. **Kayıt bağlamı değerlendirmesi** model geliştirme bitince, **kaydedilmiş tahminler üzerinde** yapılır; yeniden eğitim gerekmez:
     - D-023'ün artımlı testi (T1);
     - E1/E1h/E2/E3, denge tabloları ve referans çizgileriyle;
     - ayarlı E2h;
     - D-005 baseline'ları;
     - negatif kontroller (D-017, D-020);
     - codec zinciri duyarlılığı.
     
     Test ayrıntıları o aşamada D-025 önerisi üzerinden kesinleştirilir.
  3. **Geliştirme sırasında zorunlu kayıtlar** (ek deney değil; sonradan değerlendirmeyi mümkün kılar). Her deney ve her (tekrar, fold) için şunlar Drive'a yazılır:
     - **katılımcı, kayıt ve segment düzeyinde** dış-test tahminleri (kayıt düzeyi tahminler codec zinciri kontrolü için gerekli);
     - fine-tune'da **hem en iyi doğrulama checkpoint'inin hem son epoch'un** tahminleri (`oof_predictions_best.csv`, `oof_predictions_last.csv`);
     - dondurulmuş gömmeler (Faz 2'de bir kez çıkarılır);
     - split dosyalarının sha256'sı ve config.
  4. **Ara sonuçların raporlanması:** Değerlendirme yapılana kadar her ses modeli sonucu "kayıt bağlamından etkilenmiş olabilecek **üst sınır**" olarak raporlanır, astım tespit başarısı olarak değil. Her tabloda şu iki referans satırı bulunur:
     - yalnız bağlam (tarih + saat): AUC 0.931 ± 0.032;
     - yalnız yaş: 0.681 ± 0.076 (EXP-003).
  5. **Fine-tune backbone seçim ölçütü** (D-012) Faz 2 sonunda, sonuçlar görülmeden önce yazılır. Ham AUC'ye göre seçmenin "bağlamı en çok kullanan modeli seçme" riski o kararda açıkça değerlendirilir.
- **NEDEN:**
  - Alper'in önceliği ana hedefe (ham ses tabanlı uçtan uca astım sınıflandırması) odaklanmak.
  - Confounder bulguları zaten belgelendi: EXP-001–004, SIM-001, rapor 6.7–6.10, `docs/CONFOUND_CONTROL_DESIGN.md`.
  - Değerlendirme analizlerinin çoğu yalnız kaydedilmiş tahminlere ihtiyaç duyar. Tahminler saklanırsa ertelemenin bilimsel bedeli düşüktür. [INFERENCE]
- **ALTERNATİFLER:**
  - Kontrol deneylerini geliştirmeyle eşzamanlı yürütmek (D-025–027): daha erken bilgi, ama ana projeyi yavaşlatır.
  - Confounder'ı tamamen sona bırakıp kayıt tutmamak: sonradan analiz için yeniden eğitim gerekir.
- **RİSK:**
  - Bağlam bağımlılığı geliştirme sırasında fark edilmezse, mimari ve hiperparametre seçimleri bağlamı en iyi kullanan modele doğru kayabilir. Bu risk, sonradan değerlendirmede ölçülebilir ama geri alınamaz.
  - Early stopping, confounded bir doğrulama metriği kullanır (D-009). Son epoch tahminlerinin de saklanması bu riski sonradan ölçülebilir kılar.
  - Ölçülemeyen sınırlar değişmez: pozitiflik (sabah kaydedilmiş 12 sağlıklı; geç dönemde sağlıklı yok) ve S13 bronkodilatör yolu.
- **BİLİMSEL SONUÇ:** Sonuçlar olduğundan güçlü gösterilmez. Ara AUC'ler üst sınırdır; "ses astım bilgisi taşıyor" iddiası yalnız değerlendirme aşamasından (D-023) çıkabilir.
- **UYGULAMA:**
  - `docs/COLAB_WORKFLOW.md` Bölüm 7: tahmin dosyaları.
  - Faz 1–3 eğitim ve prob script'leri: kayıt yükümlülükleri.
  - `docs/CONFOUND_CONTROL_DESIGN.md`: değerlendirme aşamasının yol haritası.

## D-029 — Split dosyalarının üretim ayrıntıları
DURUM: KABUL · Tarih: 2026-10-08 (7. tur) · D-003, D-009 ve D-024'ün uygulanışı; yeni bir tasarım kararı değil · **Colab'da teyit edildi** (8. tur): scikit-learn 1.6.1 / numpy 2.1.3 / pandas 2.2.3 ile 5 dosyanın sha256'ı manifestle aynı
- **KARAR:**
  1. **Kohort:** en az bir geçerli kaydı (`recording_map.csv`'de `status == OK`) olan herkes → 342 (283 astım / 59 sağlıklı). Tüm görevler için **tek** split kullanılır; bir görevde kaydı eksik olan katılımcı (101244, görev 4) o görevin analizinde yalnızca yer almaz.
  2. **Tabakalar:** etiket × dönem × yaş grubu (D-009).
     - Dönem: D-024 kayıt tarihiyle. Son sağlıklı kayıt gününe kadar "erken", sonrası "geç", tarih yoksa "bilinmiyor".
     - Yaş grubu: ≤40 / >40 / bilinmiyor.
     - **5 kişiden küçük tabaka**, aynı etiketin en kalabalık uyumlu tabakasına katılır: önce aynı etiket + dönem içinde, yoksa aynı etiket içinde. Hangi tabakanın nereye katıldığı manifeste yazılır.
  3. **Dış döngü:** `StratifiedKFold(5, shuffle=True, random_state=r)`, r = 0…4 (5 tekrar). Ucuz modeller 5 tekrarın hepsini kullanır; fine-tune ilk 1–3 tekrarı.
  4. **İç döngü:** Her (r, k) dış-train kümesi, aynı tabakalarla `StratifiedKFold(5, shuffle=True, random_state=1000 + 10r + k)` ile 5 iç fold'a bölünür.
     - **İç fold 0 = doğrulama kümesi** (~%20; early stopping ve eşik, D-009).
     - İç 5-fold'un tamamı = lineer problarda düzenlileştirme seçimi.
     - Böylece iki ihtiyaç tek, sabit bir dosyadan karşılanır.
  5. **Dosyalar:**
     - `outer_r{r}.csv` (sütunlar: `participant_id, repeat, outer_fold, role, inner_fold, is_inner_val`) yalnız Drive'da (`data_derived/splits/`). Etiket veya klinik değer içermez.
     - Repo'ya yalnız agrega manifest girer (`reports/splits/`): fold başına sayılar, girdi ve çıktı sha256'ları, kütüphane sürümleri.
     - Colab'da yeniden üretilen dosyaların sha256'ı manifestteki ile aynı olmalı.
- **NEDEN:**
  - Her deneyin aynı katılımcılar üzerinde değerlendirilmesi, modeller arası eşleştirilmiş karşılaştırmanın (D-011) ön koşuludur.
  - Tabakalama, küçük sağlıklı grubunun (59) ve dönem / yaş dağılımının fold'lar arasında dengeli kalmasını sağlar; fold'dan fold'a gereksiz varyansı azaltır.
  - Seyrek tabaka yalnız 2 sağlıklı katılımcıyı etkiliyor (biri yaşı, biri tarihi ve yaşı bilinmiyor). [FACT]
  - D-024 iki katılımcının dönemini değiştiriyor: biri erken → geç, biri bilinmiyor → erken. [FACT]
- **ALTERNATİFLER:**
  - Kayıt tablosunda `StratifiedGroupKFold`: eşdeğer, daha az şeffaf.
  - Yalnız etikete göre tabakalama: dönem ve yaş fold'lar arasında rastgele dengesizleşir.
  - Ayrı bir iç 80/20 bölmesi ve ayrı bir iç 5-fold: iki dosya, aynı amaç.
  - Görev başına ayrı split: çok görevli füzyonda fold'lar uyuşmaz.
- **RİSK:**
  - Tabakalama dengeyi yalnız tabakalanan değişkenlerde sağlar. Saat gibi tabakalanmayan bağlam değişkenleri fold'lar arasında rastgele dağılır; manifestte betimsel olarak raporlanır (D-028).
  - `scikit-learn` sürümü değişirse fold atamaları değişebilir → sha256 karşılaştırması bunu yakalar.
- **BİLİMSEL SONUÇ:** Split, hiçbir model sonucuna bakılmadan, yalnız katılımcı tablosundan bir kez üretilir. Sonuca göre split seçimi (selection bias) yapısal olarak imkânsızdır.
- **UYGULAMA:** `scripts/make_splits.py`, `tests/test_make_splits.py`, `reports/splits/splits_manifest.{json,md}`.

## D-030 — Ses önbelleğinin üretim ayrıntıları
DURUM: KABUL (uygulama ayrıntısı; parametrelere Colab'da çalıştırmadan önce itiraz edilebilir) · Tarih: 2026-10-08 (8. tur) · D-008, D-015, D-018, D-021'in uygulanışı · **Colab'da üretildi ve doğrulandı** (9. tur): 2 393 kayıt; 32k 3.08 GB / 16k 1.54 GB; kalan süre medyan 10.26 s (en kısa 2.28 s; 17 kayıt < 4 s); bant genişliğinden zincir ayrımı AUC 0.865 → 0.486; dinleme kontrolü sorunsuz
- **KARAR:**
  1. **Girdi ve veri sürümü:**
     - `recording_map.csv`'de `status == OK` olan 2 393 kayıt işlenir.
     - Her dosyanın sha256'ı denetimdeki (`file_sha256`) ile aynı olmalı; değilse script durur.
  2. **Çözme (D-018):**
     - `audit_audio.decode` (denetimle aynı fonksiyon): ilk ses akışı, native SR, float32, meta veri yok.
     - Tek 44.1 kHz dosya `resample_poly(160, 147)` ile 48 kHz'e çevrilir.
  3. **Kenar kırpma (D-015):**
     - Kural ses denetimindeki aktif-kare kuralıyla aynıdır:
       - 25 ms kare, 10 ms adım;
       - eşik = en yüksek kare − 35 dB;
       - SNR vekili (p95 − p10) > 20 dB ise eşik en az gürültü tabanı + 10 dB.
     - İlk ve son aktif kareden sonra **0.10 s pay** bırakılır.
     - Sınırlar 48 kHz sinyalde saniye olarak bir kez hesaplanır ve iki hedef SR'ye aynen uygulanır.
     - İç sessizliklere dokunulmaz.
  4. **Hedef SR'ler (D-008):** `resample_poly` 48→32 kHz (2/3) ve 48→16 kHz (1/3). Resample ve filtre **tüm sinyale** uygulanır, kesim sonra yapılır; böylece kesim noktalarında filtre geçişi oluşmaz.
  5. **32 kHz alçak geçiren (D-021):**
     - Doğrusal fazlı FIR: 511 tap, Kaiser β = 8.6, sıfır gecikme.
     - −6 dB noktası 11.0 kHz; 10.8 kHz'e kadar düz; en dar kodlayıcı kesiminde (11.27 kHz) −94 dB.
     - 16 kHz yolunda ek filtre yok: `resample_poly`'nin kendi süzgeci 8 kHz'te keser.
  6. **Tepe normalizasyonu (D-015):**
     - Her SR sürümü kendi tepe değerine göre **−1 dBFS**'e (0.891) getirilir.
     - Uygulanan kazanç (dB) dizinde saklanır, yani mutlak şiddet bilgisi kaybolmaz, yalnız girdiden çıkar.
  7. **Depolama (D-018):**
     - SR başına tek bir little-endian float32 dosya (`audio_32k.f32`, `audio_16k.f32`), `np.memmap` ile okunur.
     - Yanında `index.csv` (kayıt başına ofset, uzunluk, kırpma, kazanç, bant genişliği) ve `cache_info.json` (parametreler, sürümler, sha256).
     - Konum: Drive `data_derived/audio_cache_v1/`. Önbellek bir kez üretilir; yeniden üretim bilinçli bir karardır.
  8. **Doğrulama:**
     - NaN/Inf yok;
     - her parçanın tepesi hedefte;
     - ofsetler bitişik;
     - örnek sayısı süreyle tutarlı;
     - ilk 20 kayıt yeniden işlenince birebir aynı;
     - 32 kHz yolunda etkin bant genişliği zincir başına önce / sonra.
     
     Agrega rapor `reports/audio_cache/` altında.
  9. **Pencereleme önbellekte yapılmaz.** 4 s / 2 s pencere ve kısa kayıtların sıfırla doldurulması (D-015) eğitim yükleyicisinde yapılır.
- **NEDEN:**
  - Denetimle aynı kırpma kuralı kullanıldığı için, ne kadar kırpılacağı önceden biliniyor (envanterden tahmin):
    - kalan süre medyan 10.3 s, en kısa 2.3 s;
    - 17 kayıt 4 s'den kısa kalacak;
    - toplam 6.7 saat ses → 32 kHz ≈ 3.1 GB, 16 kHz ≈ 1.5 GB.
  - Baştaki sessizliğin süresi katılımcı düzeyinde etiketle ilişkili: AUC 0.369, sağlıklılarda daha uzun (ses denetimi envanteri). [FACT] Bu, sesle ilgisiz bir ipucudur. 0.10 s payla kırpınca bütün kayıtlarda kalan baş sessizliği en fazla 0.1 s olur. [INFERENCE]
  - Sahte veriyle test (`tests/test_build_audio_cache.py`, 9 kontrol): 1.0 s / 2.0 s sessizlik → 0.88 s / 1.89 s kırpıldı; 24 kHz geniş bantlı kaynak → 11.1 kHz. [FACT]
- **ALTERNATİFLER:**
  - Kırpma sınırını her SR'de ayrı hesaplamak: sürümler arasında farklı sınırlar.
  - Önce kesip sonra resample etmek: kesim noktalarında filtre geçişi.
  - IIR / Butterworth filtre: faz bozulması, `filtfilt` kenar etkileri.
  - Kayıt başına ayrı `.npy`: Drive'dan binlerce küçük dosya okumak yavaş.
  - int16 depolama: boyut yarıya iner ve normalizasyondan sonra kırpma riski yoktur; ama D-018'i değiştirir. Yalnız Drive alanı yetmezse ayrı karar olarak.
  - RMS normalizasyonu: D-015'te değerlendirildi, seçilmedi.
- **RİSK:**
  - Düşük SNR'li kayıtlarda (%12) yalnız tepeye-göre eşik çalışır → bu kayıtlarda daha az kırpılır.
  - Tepe normalizasyonu mutlak şiddeti girdiden siler (D-015 riski).
  - 11 kHz kesim, 32 kHz modellerin 11–14 kHz mel bantlarını boşaltır (D-021 riski).
  - Filtre, kodlayıcıların 11 kHz altındaki spektral şekillendirmesini silemez.
  - ffmpeg sürümü değişirse çözülen örnekler bit düzeyinde değişebilir → önbellek bir kez üretilir ve sha256'ı saklanır.
- **BİLİMSEL SONUÇ:** Bütün modeller aynı, doğrulanmış ve sabit girdiyi görür. Ön işleme farkı modeller arası karşılaştırmayı bozamaz.
- **UYGULAMA:** `scripts/build_audio_cache.py`, `tests/test_build_audio_cache.py`, `notebooks/03_audio_cache.ipynb`, `reports/audio_cache/`.

## D-031 — MFCC baseline'larının uygulanışı (EXP-010 sadık yeniden üretim, EXP-011 bizim protokol)
DURUM: KABUL (uygulama ayrıntısı; Colab'da çalıştırmadan önce itiraz edilebilir) · Tarih: 2026-10-09 (9. tur) · Rapor Bölüm 8'deki planın uygulanışı
- **KARAR:**
  1. **Özellikler (iki deneyde aynı tarif)** [FROM PAPER]: 12 MFCC + Δ + ΔΔ (çerçeve 2048, adım 512, Hamming, 22.05 kHz), zaman ortalaması → 36; ZCR, spektral merkez, bant genişliği ve roll-off'un ortalaması ve SD'si → 8. Toplam 44.
     
     Makalede yazmayan noktalarda verilen kararlar [DECISION]:
     - yalnız ortalama alınır (makale "36 descriptor" diyor);
     - c0 dahildir (librosa varsayılanı);
     - spektral özetlerde librosa varsayılanı Hann penceresi kullanılır.
  2. **EXP-010 girdisi:**
     - Orijinal `.m4a` ffmpeg ile çözülür, `soxr_hq` ile 22.05 kHz'e indirilir (librosa.load'un yaptığı).
     - Kırpma: `librosa.effects.trim(top_db=60)`. Makale "enerji tabanlı VAD" diyor ama eşik vermiyor ve atıf ettiği Sohn 1999 istatistiksel model tabanlı (A5).
  3. **EXP-010 değerlendirmesi** [FROM PAPER]:
     - Görev başına `StratifiedKFold(5, shuffle=True, random_state=42)`, katılımcı düzeyinde.
     - Fold içinde StandardScaler → SMOTE(42) → sınıflandırıcı.
     - 14 model, varsayılan ayarlarla (random_state=42).
     - Makalenin yazmadığı iki yerde [DECISION]:
       - Voting / Stacking / Weighted Voting tabanı = GB + XGBoost + CatBoost + MLP;
       - Weighted Voting ağırlıkları train içinde 3-fold OOF AUC'sinden (sızıntısız; A6).
     - "En iyi" = en yüksek ortalama fold AUC'si (makalenin kuralı). Ayrıca 14 modelin hepsi ve medyanı raporlanır.
  4. **EXP-011 girdisi:** Harmonize önbellek (D-030, 32 kHz) `resample_poly(441, 640)` ile 22.05 kHz'e indirilir. Önbellek zaten kırpılmış ve normalize; aynı 44 özellik çıkarılır. Neden: derin modeller bu girdiyi görecek, RQ1 karşılaştırması aynı girdiyle yapılmalı.
  5. **EXP-011 değerlendirmesi:**
     - Split dosyaları outer_r0–r4 (D-029).
     - Modeller önceden sabit, hepsi raporlanır:
       - **LR = birincil.** Neden: derin gömmelerdeki lineer probla aynı sınıflandırıcı; RQ1'de temsil farkı izole edilir.
       - SVM-RBF ve GradientBoosting ikincil.
     - Düzenlileştirme, split dosyasındaki iç 5-fold'da `neg_log_loss` ile sabit ızgaralardan seçilir:
       - LR: C ∈ {0.001 … 100};
       - SVM: C × gamma, 3 × 3;
       - GB: n_estimators × max_depth, 2 × 2, lr 0.05.
     - Dengesizlik: SMOTE yerine sınıf / örnek ağırlığı. Eşik 0.5; sınıf ağırlıklı modellerde doğal karar sınırı.
     - Görev başına sonuçlar + **füzyon**: katılımcının 7 görev olasılığının ortalaması.
     - Aynı fold'larda sesi kullanmayan referans çizgileri: yaş, bağlam (saat + saat² + tarih), yaş + bağlam (D-005, D-028).
  6. **Raporlama:**
     - Fold AUC ortalaması ± SD ve fold %2.5–97.5 aralığı.
     - Tekrarlar boyunca ortalama OOF olasılığından havuzlanmış AUC + 2 000 katılımcı bootstrap'lı %95 CI.
     - Dengeli doğruluk, duyarlılık / özgüllük.
     - Bütün sonuçlar "üst sınır" uyarısıyla (D-028).
     - OOF tahminleri Drive'da saklanır (D-028); registry'ye agrega satırlar yazılır.
  7. **Kaldığı yerden devam:** Görev (EXP-010) ve tekrar (EXP-011) başına ara sonuçlar Drive'daki `partial/` klasörüne yazılır.
- **NEDEN:**
  - EXP-010, verinin ve özellik tarifinin makaleyle tutarlı olup olmadığını gösterir.
  - EXP-011, makalenin iki iyimserlik kaynağını kaldırır: tek split ve 14 model arasından test sonucuna göre seçim (A2, A7).
  - EXP-011 aynı zamanda derin modellerin karşılaştırılacağı adil tabandır.
  - Sahte veriyle test (`tests/test_mfcc_pipeline.py`, 7 kontrol): yerleştirilmiş sinyal AUC 0.82, saf gürültü 0.52 (sızıntı yok), yaş referansı 0.86. [FACT]
- **ALTERNATİFLER:**
  - EXP-011'de orijinal dosyalardan özellik çıkarmak: derin modellerle girdi farklı olur.
  - EXP-011'de makalenin 14 modelini aynen kullanmak: seçim iyimserliği geri gelir.
  - Model seçimini iç döngüde yapmak (nested): mümkün ama birincil modeli önceden sabitlemek daha şeffaf.
  - SMOTE'u EXP-011'de de kullanmak: sınıf ağırlığı daha basit ve deterministik; derin modellerle aynı.
- **RİSK:**
  - Kütüphane sürümleri makaleden farklı (Colab: Python 3.13; makale: scikit-learn 1.5.0, librosa 0.10.1). Küçük sayısal farklar beklenir.
  - Makalenin kohortu 344 / 2 408 kayıt, bizimki 342 / 2 393.
  - Kırpma eşiği ve fold atamaları makalenin bilinmeyen ayrıntılarından farklı → birebir aynı sayı beklenmez; hedef aralıktır.
  - Δ ve ΔΔ'nin zaman ortalaması neredeyse sıfırdır (A4); 24 boyutun bilgi taşımaması olası.
  - Süre: EXP-010 ~35 dk, EXP-011 ~60 dk (Colab CPU).
- **BİLİMSEL SONUÇ:** Makalenin sayısı, aynı veri üzerinde yeniden üretilebilirlik ve seçim iyimserliği açısından sınanır. Derin modellerin geçmesi gereken taban, bağlamdan etkilenmiş üst sınır olduğu açıkça yazılarak sabitlenir.
- **UYGULAMA:** `scripts/extract_mfcc_features.py`, `scripts/run_mfcc_baselines.py`, `tests/test_mfcc_pipeline.py`, `notebooks/10_mfcc_baseline.ipynb`, `reports/mfcc/`.

## D-032 — MFCC sonuçlarından çıkan üç düzeltme: güçlü ikinci taban, eşik seçimi, birincil birim = füzyon
DURUM: **KABUL** (2026-10-09, 12. tur — Alper onayladı) · Tarih: 2026-10-09 (10. tur) · EXP-011 ve EXP-012 sonuçlarına dayanır (sonuçlar görüldükten sonra; gerekçeler aşağıda)
- **KARAR:**
  1. **RQ1 için iki MFCC tabanı.**
     - Birincil taban değişmiyor: EXP-011 LR.
     - Önceden belirlenmiş tek bir doğrusal olmayan model eklenir: **MLP**, makale pipeline'ı (StandardScaler → SMOTE → varsayılan MLPClassifier).
     - Önbellek özellikleri, aynı split dosyaları, görev başına + füzyon, OOF saklanır.
     - **11. tur notu:** MLP sonuçları EXP-013'te (dengesizlik analizi) hesaplandı. Füzyon AUC'si 0.786–0.790; LR füzyonu 0.776–0.778; fark anlamlı değil. MLP ikinci taban olarak raporlanabilir ama çıtayı pratikte değiştirmiyor.
     - "Derin model MFCC'den iyi" iddiası, ikisini de aynı fold'larda eşleştirilmiş ΔAUC ile geçmeyi gerektirir (D-011).
  2. **Eşik metrikleri iç doğrulamadan.**
     - Bundan sonraki bütün deneylerde eşik, iç doğrulama kümesinde (`is_inner_val`) dengeli doğruluğu en yükselten değer olarak seçilir ve test fold'una uygulanır (D-009'un uygulanışı).
     - EXP-011'de SVM-RBF ve GB'nin dengeli doğruluk / duyarlılık / özgüllük sütunları **geçersiz** sayılır; AUC'ler geçerli.
  3. **Birincil karşılaştırma birimi katılımcı düzeyinde füzyon** (7 görev olasılığının ortalaması).
     - Görev başına sonuçlar raporlanır ama keşifseldir (RQ4 / RQ5).
     - Derin modeller de aynı füzyonla raporlanır.
- **NEDEN:**
  - EXP-012 D: makale pipeline'ında MLP / Stacking tek görevde 0.69–0.72; önceden seçtiğimiz LR ise ~0.66. Taban yalnız LR olursa derin modellerin "geçtiği" çıta yapay olarak alçak olabilir. MLP'yi sonuçları gördükten sonra seçmek iyimser bir seçimdir; ama bu iyimserlik tabanı **güçlendirir**, yani iddiamız aleyhine çalışır. Bu yüzden kabul edilebilir. [INFERENCE]
  - EXP-011: SVM'de duyarlılık 1.00 / özgüllük 0.00. Platt kalibrasyonu sınıf ağırlığını yok sayıyor, sabit 0.5 eşiği anlamsızlaşıyor. [FACT]
  - EXP-011 / EXP-012: tek görev AUC'leri yaş-only düzeyinde ve görev sıralaması kararsız. Füzyon her görevden +0.08–0.13 iyi. Anlamlı birim katılımcı (D-003). [FACT]
- **ALTERNATİFLER:**
  - Yalnız LR tabanı: zayıf taban riski.
  - Makalenin 14 modelinin hepsi: seçim iyimserliği geri gelir.
  - Stacking: 25 fold'da ~20 dk ve karmaşık; MLP'den belirgin farkı yok (gelecek: 0.720 vs MLP diğer görevlerde 0.69–0.70).
  - Eşik için Youden veya sabit önceliğe göre eşik: iç doğrulamada dengeli doğruluk, birincil ikincil metrikle tutarlı.
- **RİSK:**
  - İç doğrulamada ~9 sağlıklı var → eşik gürültülü olur. Dengeli doğruluk bu yüzden ikincil metrik kalır.
  - Füzyon, aynı seansın 7 kaydını birleştirdiği için seans / bağlam bilgisini de güçlendirebilir. Değerlendirme aşamasında (D-028) özellikle füzyon skorları test edilmeli.
- **BİLİMSEL SONUÇ:** RQ1 karşılaştırması zayıf bir tabana karşı yapılmaz. Eşik metrikleri yanıltıcı olmaz. Birincil sonuç katılımcı düzeyindedir.
- **UYGULAMA:** EXP-013'teki MLP sonuçları (`scripts/analyze_imbalance_selection.py`); eşik seçimi D-033'teki iç 5-fold OOF yöntemiyle; Faz 2 prob kodu aynı değerlendirme fonksiyonlarını kullanır.

## D-033 — Dengesizlik ve seçim politikası: SMOTE yok, eşik iç CV'den, kalibrasyon raporlanır, bağlam dengeleme eğitimde yok, seçim önceden ya da iç içe
DURUM: **KABUL** (2026-10-09, 12. tur — Alper onayladı) · Tarih: 2026-10-09 (11. tur) · Ayrıntı: `docs/IMBALANCE_AND_SELECTION.md` · Kanıt: EXP-013, EXP-014, EXP-015 (keşifsel) · D-032'nin 2. maddesini iyileştirir
- **KARAR:**
  1. **Etiket dengesizliği.** SMOTE kullanılmaz; tek istisna EXP-010'daki makale taklidi. Klasik modellerde sınıf ağırlığı, derin modellerde ağırlıklı BCE kullanılır. Focal / LDAM / logit ayarlaması gerekmiyor.
  2. **Eşik.** Her dış fold'da, eğitim katılımcılarının **iç 5-fold OOF** tahminlerinde dengeli doğruluğu en yükselten eşik seçilir ve teste uygulanır. Bu, D-032'deki "yalnız iç doğrulama kümesi"nden (~9 sağlıklı) daha az gürültülüdür.
  3. **Kalibrasyon.** Brier, ortalama tahmin − gerçek oran ve kalibrasyon eğimi her deneyde raporlanır. Klinik olasılık gerekirse iç CV'de yeniden kalibre edilir.
  4. **Kayıt bağlamı.** Eğitim düzeyinde dengeleme (hücre-içi SMOTE, ters eğilim ağırlığı, Group DRO, yalnız örtüşmede eğitim) birincil hatta yapılmaz. Değerlendirme aşamasında yalnız örtüşme bölgesinde ve yalnız duyarlılık analizi olarak yer alabilir (D-026, D-028).
  5. **Model / backbone / görev seçimi.** Faz 2 ve 3'te seçim ya önceden sabitlenir (D-032: birincil birim füzyon) ya da iç içe CV ile yapılır. Seçilmeyen bütün konfigürasyonlar raporlanır.
- **NEDEN:**
  - EXP-013: SMOTE ve sınıf ağırlığı AUC'yi değiştirmiyor, kalibrasyonu bozuyor. Literatürle aynı: van den Goorbergh ve ark. 2022 (AUC iyileşmez, azınlık olasılığı fazla tahmin edilir, eşik kaydırmak aynı etkiyi verir); Blagus & Lusa 2013 (yüksek boyutta sınırlı fayda). [FROM PAPER + FACT]
  - EXP-014: bağlam dengeleme tüm veride fark yaratmıyor; yalnız örtüşmede eğitim AUC'yi ~0.07 düşürüyor. Geç dönemde sağlıklı olmadığı için tam dengeleme imkânsız (pozitiflik). [FACT]
  - EXP-015: makalenin seçim prosedürünün iç içe tahmini (sayılar EXP-015 raporunda); seçim iyimserliği Faz 2'de de oluşur. [FACT + FROM PAPER: Cawley & Talbot 2010; Varma & Simon 2006]
- **ALTERNATİFLER:**
  - SMOTE'u makaleyle uyum için bütün deneylerde tutmak: kalibrasyon kaybı, sentetik veri, ham seste anlamsız.
  - Hiç düzeltme yapmayıp yalnız eşiği ayarlamak: klasik modellerde eşdeğer, ama derin modellerde ağırlıksız eğitim azınlık gradyanını zayıflatabilir.
  - Group DRO'yu birincil yapmak: 12 kişilik grup, aşırı uyum.
- **RİSK:**
  - Ağırlıklı BCE de kalibrasyonu kaydırır → raporlanır ve gerekirse düzeltilir.
  - İç CV eşiği hâlâ ~47 sağlıklıya dayanır.
  - EXP-013/014/015 keşifseldir; MFCC özelliklerinde geçerli olan sonuç derin gömmelerde farklı olabilir. [HYPOTHESIS]
- **BİLİMSEL SONUÇ:** Dengesizlik düzeltmesi ayırma iddiası gibi sunulmaz. Seçim iyimserliği ve split şansı bütün aşamalarda kontrol altında tutulur.
- **UYGULAMA:** Faz 2 prob kodu ve Faz 3 eğitim döngüsü; `scripts/analyze_imbalance_selection.py` (referans uygulama).

## D-034 — Faz 2 protokolü: dondurulmuş gömme + lineer prob
DURUM: **KABUL** (2026-10-09, 13. tur — Alper onayladı: birincil kısa kayıt politikası `zeropad`, `nopad` önceden belirlenmiş duyarlılık. Gerekçe: önceki çalışmayla ve Faz 3 giriş politikasıyla tutarlılık. **Bu seçimin dolgunun yaratabileceği süre ipuçlarını ortadan kaldırdığı varsayılmaz**; META-016 ve EXP-016S ile ayrı değerlendirilir) · 13. turda revize edildi · Tarih: 2026-10-09 (12. tur) · **Faz 2 sonuçları görülmeden** · Ayrıntı: `docs/PHASE2_DESIGN.md` · D-012 basamak 2'nin uygulanışı · Faz 2, Faz 3'ün (ham seste uçtan uca fine-tune) yerine geçmez; ondan önceki kontrollü temsil karşılaştırmasıdır
- **KARAR:**
  1. **Girdi.** Harmonize önbellek (D-030), modelin resmi SR'si. 4.0 s pencere / 2.0 s adım; son pencere kaydın sonuna hizalı (kısmi pencere yok). **4 s'den kısa kayıtlar (17): birincil politika `zeropad` = D-015** (4 s'ye sıfır dolgusu, Boll ile aynı). Dolgunun işlenişi: PANNs maskesiz (resmi API); BEATs resmi `padding_mask` + geçerli yamalarda havuzlama; WavLM Large resmi `attention_mask` + geçerli çerçevelerde havuzlama (dolgu etkisiz); WavLM Base+'ya maske verilmez (HF önerisi; GroupNorm) → dolgu gerçek çerçeveleri de etkiler, yalnız havuzlama maskeli. İki politika da hesaplanır; `nopad` (gerçek uzunluk) önceden belirlenmiş duyarlılık (EXP-016S, tekrar 0). Ek duyarlılık: kısa kaydı olan katılımcılar değerlendirmeden çıkarılır (META-016). Batch = tek kaydın pencereleri; farklı uzunluklar aynı batch'e girmez. *(12. turdaki öneri "nopad birincil" idi; Faz 2–Faz 3 tutarlılığı ve Boll karşılaştırması gerekçesiyle değiştirildi.)* Keşifsel ek görünüm: tüm kayıt tek girdi (EXP-018).
  2. **Gömme.** PANNs: resmi `embedding` (fc1). BEATs / WavLM: forward hook ile her transformer bloğunun çıkışı (katman 0 = ilk bloğun girdisi); zaman (BEATs'te yama) ortalaması; bütün katmanlar saklanır. Kayıt gömmesi = pencere gömmelerinin ortalaması. fp32, deterministik ayarlar.
  3. **Birincil temsil (önceden sabit; katman seçimi yok).** PANNs: fc1. SSL: katman 1..L'nin eşit ağırlıklı ortalaması; her katman, her kanal eğitim fold'unda z-skoruna çevrildikten sonra (`LayerAverage`, Pipeline içinde). Bir modelin bütün katmanları aynı boyutta (assert). Katman katman sonuçlar yalnız keşifsel (EXP-017, tekrar 0) ve hiçbir seçime girmez.
  4. **Prob.** EXP-011 LR protokolü: StandardScaler → LR(class_weight=balanced); C iç 5-fold `neg_log_loss` ile, ızgara 10⁻⁵…10² (MFCC dahil bütün temsillerde aynı); görev başına model; füzyon = 7 görev olasılığının ortalaması (D-032); eşik iç 5-fold OOF'tan, kalibrasyon raporlanır (D-033).
  5. **MFCC kolları aynı kodla:** mfcc_lr ve mfcc_mlp (D-032). mfcc_lr füzyonu EXP-011'i ±0.01 içinde yeniden üretmeli (regresyon testi).
  6. **Aynı birim, aynı fold'lar.** Bütün kollar aynı katılımcı düzeyi dış fold'larda (outer_r0–4) ve aynı katılımcılarda değerlendirilir; birim katılımcı (füzyon). Bir katılımcının kayıtları eğitim ve dış teste dağılmaz (assert'ler).
  7. **Onaylayıcı aile (RQ1):** 6 backbone × füzyon. Fold düzeyinde eşleştirilmiş ΔAUC (25 fold), Nadeau–Bengio düzeltilmiş t, **tek yönlü** (H1: Δ > 0); "b MFCC'den iyi" ⇔ hem mfcc_lr hem mfcc_mlp geçilir (kesişim-birleşim: p = max); Holm (6); Holm-düzeltilmiş p 0.025 ile karşılaştırılır — önceden belirlenmiş, muhafazakâr eşik: yanlış "daha iyi" ilanı için aile bazında hata oranı ≤ 0.025; "daha iyi" kararları için iki yönlü Holm 0.05'ten hiçbir zaman gevşek değil (tasarım belgesi 5.3a). Nadeau–Bengio, 5 tekrar × 5 fold = 25 **bağımsız olmayan** fold skoruna uygulanır; düzeltme sezgiseldir ve raporda belirtilir. Karar: Holm p < 0.025 **ve** iki katılımcı bootstrap %95 CI'ının alt sınırı > 0. DeLong (tekrar başına) destekleyici. İki yönlü p'ler ve bütün Δ'lar raporlanır. D-011'in zaman-örtüşme koşulu D-028 ile değerlendirme aşamasında → "geçici".
  8. **Keşifsel (önceden listeli; her aile içinde Holm):** 15 backbone çifti (RQ3), görev başına Δ vs mfcc_lr (42; RQ4), ünlü − kelimeler (RQ5), EXP-016S, EXP-017, EXP-018. Güç hesabı (saptanabilir fark ≈ 0.06–0.09 AUC) yalnız beklenti içindir; eşik olarak kullanılmaz (varsayımlar tasarım belgesi Bölüm 5.5).
  9. **Meta veri / confounder karşılaştırmaları ayrı rapor (META-016):** bağlam bağımlılığı (EXP-014 ölçüleri), süre dağılımı, yalnız-süre referansı, dışlama duyarlılığı. Hiçbir karar ya da seçim bunlara dayanmaz. Ana tabloda yalnız D-028'in iki referans satırı (test edilmez).
  10. **Kilitli kurallar ve keşifsel analizler ayrı:** tasarım belgesi Bölüm 2.1 (R1–R12, sonuç görülmeden) ve 2.2 (sonuca bakılarak yapılabilecekler; "post hoc" etiketli; kilitli kuralları değiştiremez).
  11. **Smoke test:** gerçek checkpoint'lerle SMK-001 (S1–S14) PASS olmadan çıkarım yok; bir model FAIL verirse nedeni incelenmeden EXP-016'ya geçilmez, eşik gevşetilerek geçirilmez, düzeltmeden sonra SMK-001 baştan çalıştırılır (tasarım belgesi 1.1). Rastgele küçük modellerle yapılan birim testleri (`UNITTEST_random_init_smoke`) SMK-001'in yerine geçmez.
- **NEDEN:**
  - Temsil farkını izole eder; MFCC ile tek değişken temsil (D-012).
  - Kısa kayıt: Faz 2'nin girdisi Faz 3'ünkini yansıtmalı; Faz 3 batch eğitimi gerektirir ve D-015 / Boll bunu sıfır dolgusuyla çözer. Dolgunun yapay içerik riski (PANNs) iki duyarlılıkla ölçülür. [DECISION + INFERENCE]
  - Katman ortalaması: SSL'de bilgi orta katmanlarda yoğunlaşabilir [FROM PAPER: Pasad ve ark. 2021; Chen ve ark. 2022]; seçim yapmadan bütün katmanları kullanır (D-033 madde 5). Z-skoru katmanlar arası ölçek farkını giderir.
  - Tek yönlü test: iddia yönlü; iki yönlü p'lerde MFCC'den anlamlı biçimde kötü bir backbone Holm sırasını bozardı. 0.025 eşiği iki yönlü 0.05 kadar katı.
- **ALTERNATİFLER:** nopad birincil; kısa kayıtları atmak; döngüsel dolgu; yalnız son katman; katman seçimi (dış ya da iç CV); katmanları uç uca eklemek; MLP prob; iki yönlü test. Karşılaştırma: tasarım belgesi Bölüm 3–5.
- **RİSK:**
  - Güç sınırlı; 0.02–0.05'lik farklar çözülemez. [INFERENCE]
  - Model süreyi dolaylı kullanabilir (dolgu, kısa bağlam, iç sessizlik oranı); META-016 ve EXP-016S ölçer. [INFERENCE]
  - LR doğrusal olmayan bilgiyi kaçırabilir → gömmeler aleyhine, muhafazakâr.
  - Gömmeler MFCC'den fazla bağlam taşıyabilir → her AUC üst sınır. [HYPOTHESIS]
  - BEATs iter3+ (AS2M) tokenizer öğretmeni AudioSet etiketleriyle fine-tune edilmiş → "saf öz-gözetimli" değil [FROM PAPER: BEATs Bölüm 4.2, Tablo 2]. Modeller ön-eğitim hedefi, etiket kullanımı, veri alanı, mimari, boyut ve bant genişliğinde aynı anda farklı → RQ3 bir ön-eğitim yöntemi karşılaştırması değil, **hazır temsillerin karşılaştırması** olarak yorumlanır (tasarım belgesi 9.6).
- **BİLİMSEL SONUÇ:** Altı backbone ve iki MFCC tabanı aynı fold'larda, aynı birimde, sonuç görülmeden yazılmış kurallarla karşılaştırılır; Faz 3 için tutarlı bir dondurulmuş taban oluşur.
- **UYGULAMA:** `scripts/backbones.py`, `smoke_test_backbones.py`, `extract_embeddings.py` (`--short-policy`), `run_probes.py` (EXP-016 / 016S / 017 / 018), `report_metadata_monitor.py` (META-016), `tests/test_phase2.py`, notebook 20 ve 21.

## D-035 — Faz 3 backbone seçim kuralı (Faz 2 sonuçlarından önce yazıldı): seçim fold başına
DURUM: **KABUL** (2026-10-09, 13. tur — Alper **koşullu** onayladı: "BEATs/WavLM arasındaki seçim yalnızca ilgili dış değerlendirme katının eğitim verisi içindeki doğrulama sonuçlarıyla yapılmalı; dış değerlendirme katının skorları model seçiminde kullanılmamalı." Koşul kurala işlendi: 12. turdaki "25 fold ortalaması → tek backbone" önerisi yerine **fold başına seçim**) · **Faz 2 sonuçları görülmeden** · Ayrıntı: `docs/PHASE2_DESIGN.md` Bölüm 6
- **KARAR:**
  1. Sabit: PANNs CNN10 pretrained (RQ2: sıfırdan CNN10'un eşi) ve PANNs CNN14 32k (Boll karşılaştırması).
  2. Kurala bağlı: BEATs, WavLM Base+, WavLM Large arasından **her dış fold (tekrar r, fold k) için ayrı** seçim. Ölçüt: yalnız o fold'un eğitim katılımcılarının iç 5-fold OOF tahminleri → 7 görevin katılımcı ortalaması → AUC. En yükseğe 0.01'den yakın olanlar arasından en az parametreli. Dış test fold'unun skorları seçime girmez.
  3. Faz 3, her fold'da o fold'un seçtiği backbone'u fine-tune eder. Faz 3'ün sonucu "iç doğrulamayla seç, sonra fine-tune et" prosedürünün dürüst tahminidir; belirli bir backbone'un değil. Dondurulmuş-vs-fine-tune karşılaştırması aynı fold'da aynı backbone ile eşleştirilir.
  4. Bağlam izleme göstergeleri ölçüt değildir; META-016'da raporlanır.
  5. Faz 3, Faz 2 sonucundan bağımsız yapılır.
  6. Bütçe yetmezse ilk çıkarılan CNN14.
- **NEDEN:** Seçimi sonuçlardan önce sabitlemek (D-033 madde 5). Fold başına seçim, fold k'nın dış test katılımcılarının (diğer fold'ların eğitim kümeleri aracılığıyla bile) seçime girmesini engeller; 12. turdaki ortalama skor bu dolaylı sızıntıya açıktı. RQ2 için aynı mimarinin pretrained ve sıfırdan hâlleri gerekir.
- **ALTERNATİFLER:** Tek backbone (25 fold ortalaması; yorumu kolay ama dolaylı sızıntı); dış havuzlanmış AUC ile seçim; CONFOUND belgesindeki T1'e göre seçim (D-026, ertelendi; bağlam kestirmesini seçmemek açısından güçlü ama D-028'in ertelediği testi öne çeker); bütün backbone'ları fine-tune etmek.
- **RİSK:** Fold'lar farklı backbone seçebilir → Faz 3'te üç backbone'un kod yolu da hazır olmalı (WavLM Large belleği, SMK-001 S12). E1 temelli ölçüt bağlamı en iyi öğreneni seçebilir; azaltma: PANNs iki model sabit, altı backbone'un tahminleri saklanıp değerlendirme aşamasında lens setinden geçer. İç OOF tahminleri C'nin seçildiği aynı iç fold'lardan gelir → hafif, simetrik iyimserlik. [INFERENCE]
- **BİLİMSEL SONUÇ:** Faz 3'e geçiş, dış test fold'una hiç bakmayan, önceden yazılmış bir kurala bağlı; prosedürün iyimserliği dondurulmuş düzeyde ölçülür.
- **UYGULAMA:** `run_probes.select_per_fold` (yalnız iç OOF tablosunu alır; iç satırların eğitim rolünde olduğunu assert eder) → `experiments/frozen_probe/d035_selection.csv`; `procedure_estimate` (dış test AUC'si yalnız değerlendirme için); testler: P6 (kurgulanmış örnek, eşitlik kuralı, sızıntı assert'i), P7 (dış test tahminleri bozulunca seçim birebir aynı).
