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
DURUM: ÖNERİLDİ (Faz 1'de kesinleşecek)
- **KARAR:** Mono → model SR'sine resample → enerji tabanlı kenar kırpma (tepeye göre) → tepe normalizasyonu → 4.0 s pencere / 2.0 s hop, kısa kayıtlar sıfırla doldurulur. Hepsi dosya başına deterministik (split'ten önce güvenli).
- **NEDEN:** Referans makaleyle karşılaştırılabilirlik [FROM PAPER]; kırpma, baş/son sessizlikteki oda gürültüsünü (ortam confounder'ı) azaltır. [INFERENCE]
- **ALTERNATİFLER:** Kırpmasız; tüm kaydı tek girdi olarak vermek (10 s, PANNs ön-eğitimiyle uyumlu); RMS normalizasyonu.
- **RİSK:** Tepe normalizasyonu mutlak ses yüksekliği bilgisini siler (hem olası hastalık sinyali hem mikrofon mesafesi confounder'ı). Kırpma eşiği ses denetimindeki gürültü tabanına göre seçilmeli. [NEEDS VERIFICATION]
- **BİLİMSEL SONUÇ:** —
- **UYGULAMA:** Faz 1.

## D-016 — Augmentation başlangıçta kapalı
DURUM: ÖNERİLDİ
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
DURUM: KABUL · Tarih: 2026-10-08 (5. tur) · **Ses modeli sonuçları görülmeden önce alındı** · D-017'deki "başlık sayısı E2" kuralının yerine geçer · Ayrıntı: rapor Bölüm 6.9
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
