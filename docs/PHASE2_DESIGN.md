# Faz 2 tasarımı: smoke test, dondurulmuş gömme, lineer prob

**Tarih:** 2026-10-09 (12. tur) · **Durum:** tasarım ÖNERİLDİ (D-034, D-035) · **Kod:** `scripts/backbones.py`, `scripts/smoke_test_backbones.py`, `scripts/extract_embeddings.py`, `scripts/run_probes.py`, `tests/test_phase2.py` · **Notebook'lar:** `20_smoke_tests.ipynb`, `21_frozen_embeddings.ipynb`

**Bu belge sonuçlar görülmeden yazıldı.** Faz 2'de hiçbir model henüz bu veride çalıştırılmadı. Buradaki beklentiler ve seçim kuralları, Faz 2 sonuçlarına bakılmadan sabitlenmiştir.

---

## 0. Beş dakikada özet

**Ne yapıyoruz?**
- Hazır (önceden eğitilmiş) altı ses modelini **hiç değiştirmeden** kullanıyoruz.
- Her kaydı modelden geçirip bir sayı vektörü çıkarıyoruz. Bu vektöre "gömme" (embedding) diyoruz.
- Gömmelerin üstüne, MFCC'de kullandığımız lojistik regresyonun aynısını kuruyoruz ("lineer prob").

**Neden?**
- Soru şu: Bu modellerin sesi temsil etme biçimi, astım ile sağlıklıyı MFCC'den daha iyi ayıracak bilgi taşıyor mu?
- Modelleri değiştirmediğimiz için fark yalnız **temsilden** gelir. Eğitim şansı, hiperparametre, epoch sayısı gibi konular karışmaz.
- Ucuz bir adım: GPU'da toplam ~1 saat, gerisi CPU.
- Faz 3'te hangi modelleri fine-tune edeceğimizi seçmemize yardım eder. Seçim kuralı bu belgede **şimdiden** yazılı.

**Benzetme:** Altı farklı uzman aynı ses kaydını dinleyip birer "not kâğıdı" dolduruyor. Uzmanlara astım hakkında hiçbir şey öğretmiyoruz. Yalnız şuna bakıyoruz: Kâğıtlardaki notlardan basit bir kuralla (lineer model) astımı tahmin edebiliyor muyuz? Fine-tune (Faz 3) ise uzmanlara astımı öğretmek demek.

**Altı backbone ve bir taban:**

| Kısa ad | Model | Ön-eğitim | SR | Gömme |
|---|---|---|---|---|
| cnn10 | PANNs CNN10 | AudioSet, gözetimli | 32 kHz | 512 (fc1) |
| cnn14 | PANNs CNN14 | AudioSet, gözetimli | 32 kHz | 2048 (fc1) |
| cnn14_16k | PANNs CNN14_16k | AudioSet, gözetimli | 16 kHz | 2048 (fc1) |
| beats | BEATs iter3+ (AS2M), fine-tune edilmemiş | AudioSet, öz-gözetimli* | 16 kHz | 12 katman × 768 |
| wavlm_base_plus | WavLM Base+ | 94 bin saat İngilizce konuşma, öz-gözetimli | 16 kHz | 12 katman × 768 |
| wavlm_large | WavLM Large | aynı | 16 kHz | 24 katman × 1024 |
| mfcc_lr / mfcc_mlp | 44 MFCC özelliği (EXP-011) | — | 22.05 kHz | 44 |

\* iter3+'ın tokenizer'ı, AudioSet etiketleriyle fine-tune edilmiş bir öğretmenden damıtılmış olabilir. Doğruysa BEATs iter3+ "tamamen öz-gözetimli" sayılmaz. Bu, RQ3 yorumunu (gözetimli mi, öz-gözetimli mi) etkiler. [NEEDS VERIFICATION: BEATs makalesi Bölüm 3–4]

**Deneyler:**

| ID | Ne | Rol |
|---|---|---|
| SMK-001 | Smoke test: yükleme, şekil, determinizm, işlevsel kontrol, Faz 3 bellek ölçümü | Ön koşul |
| EXP-016 | Dondurulmuş gömme + lineer prob, 6 backbone + 2 MFCC kolu, 5 × 5 CV | **Onaylayıcı** (füzyon) + keşifsel (görev başına) |
| EXP-017 | Katman katman prob (BEATs, WavLM), yalnız tekrar 0 | Keşifsel |
| EXP-018 | Tüm kayıt tek girdi vs 4 s pencereler | Keşifsel |

**Onay bekleyen kararlar:** D-034 (protokol) ve D-035 (Faz 3 seçim kuralı). Ayrıntı Bölüm 9.

---

## 1. Bilimsel amaç

### 1.1 Cevaplamak istediğimiz sorular

| Soru | Faz 2'deki biçimi | Rol |
|---|---|---|
| RQ1 MFCC vs pretrained | Her backbone'un füzyon AUC'si, MFCC-LR ve MFCC-MLP füzyonundan yüksek mi? (aynı fold'lar, eşleştirilmiş) | **Onaylayıcı** |
| RQ3 PANNs vs BEATs vs WavLM | Backbone çiftleri arasındaki füzyon farkları | Keşifsel (önceden listelenmiş çiftler) |
| D-008 bant genişliği | CNN14 (32k) vs CNN14_16k | Keşifsel |
| RQ4 / RQ5 görev türü | Görev başına AUC, ünlü (aaa) vs kelimeler | Keşifsel |
| Katman | SSL modellerinde hangi katman bilgi taşıyor? (EXP-017) | Keşifsel |
| Girdi uzunluğu | 4 s pencere vs tüm kayıt (EXP-018) | Keşifsel; Faz 3 girdi tasarımına bilgi |
| Faz 3 seçimi | Hangi backbone fine-tune edilecek? (D-035) | Önceden yazılmış kural |

### 1.2 Cevaplamadığımız sorular (bilerek)

- **"Ses astım bilgisi taşıyor mu, yoksa kayıt bağlamını mı öğreniyor?"** Bu, D-028 ile değerlendirme aşamasına ayrıldı. Faz 2'deki her AUC bir **üst sınırdır**. Bağlam tek başına 0.93, yaş tek başına 0.68 AUC veriyor; bu iki referans çizgisi her tabloda yer alır.
- **"Fine-tune işe yarar mı?"** Faz 3'ün sorusu. Faz 2, fine-tune'un karşılaştırılacağı "dondurulmuş" tabanı üretir.

### 1.3 Neden önce dondurulmuş prob? (D-012'nin tekrarı)

- **Temsili izole eder.** Ağırlıklar sabit; eğitim gürültüsü yok. Fark yalnız temsilden gelir.
- **Deterministik ve ucuz.** Gömmeler bir kez çıkarılır. Problar CPU'da dakikalar–saatler sürer. Altı backbone aynı protokolde karşılaştırılır.
- **MFCC ile birebir aynı sınıflandırıcı.** EXP-011'deki LR protokolü aynen kullanılır. RQ1 karşılaştırmasında tek değişken temsil olur.
- **Faz 3'ün tabanı.** "Fine-tune, dondurulmuş temsile göre ne kazandırıyor?" sorusu ancak bu tabanla cevaplanır.

---

## 2. Verinin yapısı

| Öğe | Değer | Kaynak |
|---|---|---|
| Kohort | 342 katılımcı (283 astım / 59 sağlıklı); görev 4 (ordu) için 341 | D-002, AUD-001 |
| Kayıt | 2 393 (katılımcı × 7 görev; 101244'ün görev 4 kaydı yok) | önbellek raporu |
| Girdi | Harmonize önbellek `audio_cache_v1` (D-030): kenarlar kırpılmış, 32k yolunda 11 kHz alçak geçiren, tepe −1 dBFS, float32 | `reports/audio_cache/` |
| Süre (kırpma sonrası) | min 2.28 · p5 8.92 · medyan 10.26 · max 13.09 s; 17 kayıt < 4 s | önbellek raporu |
| Pencere sayısı (4 s / 2 s, son pencere sona hizalı) | 10.26 s'lik kayıt → 5 pencere; toplam ≈ 12 000 [INFERENCE] | Bölüm 5.2 |
| Split | `outer_r0–4.csv` (D-029): 5 tekrar × 5 dış fold; her dış train içinde 5 iç fold (`inner_fold`) | `reports/splits/` |
| Bağlam ve yaş | `participant_context.csv` (label, age, start_hour, days, recording_date) | D-024 |
| MFCC | `features_cache.csv` (EXP-011 girdisi, 44 özellik) | D-031 |

**İstatistiksel birim:** katılımcı (D-003). Bir katılımcının bütün kayıtları ve pencereleri aynı fold'dadır.

**Toplama hiyerarşisi (Faz 2):**
1. pencere → (zaman ortalaması) pencere gömmesi;
2. pencereler → (ortalama) kayıt gömmesi;
3. kayıt gömmesi → LR → kayıt olasılığı (görev başına = katılımcı başına bir kayıt);
4. 7 görevin olasılık ortalaması → katılımcı füzyon skoru (D-032).

**Bir ayrıntı (doğrusallık):** LR'nin logit'i girdinin doğrusal fonksiyonudur. Bu yüzden "pencere gömmelerinin ortalamasına LR uygulamak", aynı ağırlıklarla "pencere logit'lerinin ortalaması"na eşittir. Yani `docs/PHASE0_REPORT.md` Bölüm 7'deki "kayıt = pencere logit'lerinin ortalaması" kuralıyla tutarlıyız (StandardScaler ve katman ortalaması da doğrusal dönüşümler olduğu için eşitlik korunur). Fark yalnız eğitimde: biz LR'yi kayıt düzeyinde (bağımsız birim) eğitiyoruz, pencere düzeyinde değil. Böylece aynı kaydın 5 penceresi 5 bağımsız örnekmiş gibi sayılmaz. [INFERENCE]

---

## 3. Varsayımlar

| # | Varsayım | Etiket | Nasıl kontrol ediliyor |
|---|---|---|---|
| V1 | Resmi kod + resmi ağırlık + resmi SR + resmi frontend = modelin ön-eğitimde gördüğü girdi dağılımı | [FROM OFFICIAL DOCS] | SMK-001: `strict=True` yükleme, sözleşme assert'leri, PANNs'te "Speech" sınıfı kontrolü |
| V2 | Gömme çıkarımı katılımcıdan katılımcıya bilgi taşımaz (eval modunda her kayıt bağımsız işlenir) | [INFERENCE] | SMK-001: toplu (batch) işleme ile tek tek işleme aynı sonucu veriyor mu? |
| V3 | Gömmeler etiketten habersiz üretilir; split'ten önce bir kez çıkarmak sızıntı yaratmaz | [INFERENCE] | Çıkarım kodu etiket okumaz; öğrenilen hiçbir parametre bizim veride ayarlanmaz |
| V4 | Zaman ortalaması (mean pooling), astımla ilgili bilgiyi büyük ölçüde korur | [HYPOTHESIS] | Test edilmiyor; alternatifler Bölüm 5.3 |
| V5 | 11 kHz alçak geçiren, 32 kHz PANNs'in en üst ~4 mel bandını boşaltır; model bunu tolere eder | [INFERENCE] | 64 bandın 4'ünün merkezi > 11 kHz (librosa mel hesabı). AudioSet'te dar bantlı çok kayıt var. CNN14 vs CNN14_16k karşılaştırması dolaylı bilgi verir |
| V6 | WavLM'in İngilizce ön-eğitimi Türkçe kelimelerde de kullanılabilir akustik temsil üretir | [HYPOTHESIS] | Test edilmiyor; sonuç yorumunda kısıt |
| V7 | 7 görevin olasılık ortalaması makul bir katılımcı skorudur | [DECISION, D-032] | EXP-011'de füzyon her görevden +0.08–0.13 iyi |

---

## 4. Riskler: sızıntı, confounding, seçim yanlılığı

### 4.1 Sızıntı (leakage)

| Risk | Nerede | Önlem |
|---|---|---|
| Katılımcı sızıntısı | Split | Katılımcı düzeyinde split dosyaları; her fold'da `train ∩ test = ∅` assert'i (D-003) |
| BatchNorm'un batch istatistiği kullanması | PANNs | `model.eval()` assert'i; batch-değişmezlik testi (V2) |
| Model içi augmentation | PANNs SpecAugment, WavLM maskeleme | Yalnız `training` modunda çalışırlar; eval'de kapalı. SMK-001 kontrol eder (D-016) |
| Ölçekleme / katman standardizasyonu testi görmesi | Prob | Bütün dönüşümler sklearn Pipeline içinde, yalnız train'de fit edilir |
| Düzenlileştirme (C) ve eşik seçiminin testi görmesi | Prob | C: iç 5-fold `neg_log_loss`. Eşik: iç 5-fold OOF tahminlerinde dengeli doğruluk (D-033) |
| Katman seçiminin testi görmesi | SSL modelleri | Birincil temsil **önceden sabit** (katman ortalaması). Katman katman sonuçlar yalnız keşifsel (EXP-017) |
| Sıfır dolgusunun (padding) süre ipucu taşıması | < 4 s kayıtlar | Dolgu yok (Bölüm 5.2) |

### 4.2 Confounding (D-028 ile ayrı başlık)

- **Kayıt bağlamı (saat, tarih).** Gömmeler MFCC'den çok daha zengin. Oda gürültüsü, cihaz durumu, günün saatine bağlı ses değişimi gibi bilgileri de taşıyabilirler. Bu yüzden derin gömmelerin MFCC'den **daha çok** bağlam öğrenmesi olasıdır. [HYPOTHESIS]
  - Faz 2'de bunu test etmiyoruz (D-028).
  - Ama **unutmamak** için her backbone'a üç ucuz izleme göstergesi ekliyoruz (D-034 madde 8). Bunlar EXP-014'teki "bağlam bağımlılığı" ölçüleri: etiket sabitken (yalnız hastalar ya da yalnız sağlıklılar içinde) füzyon skorunun geç vs erken dönemi ve sabah vs öğleden sonrayı ne kadar ayırdığı.
  - Bu göstergeler hiçbir seçimde kullanılmaz, yalnız raporlanır.
- **Yaş.** Referans çizgisi olarak her tabloda.
- **Kodlama zinciri (Apple / FFmpeg).** Harmonizasyon bant genişliği farkını sildi (AUC 0.865 → 0.486). 11 kHz altındaki spektral şekillendirme kalmış olabilir. Gömme düzeyinde zincir probu değerlendirme aşamasında (D-028). Gömmeler bunun için saklanıyor.

### 4.3 Seçim yanlılığı (D-033 madde 5)

- **Faz 2'nin kendi sayıları:** Bütün backbone'lar, birincil temsil ve karşılaştırma ailesi önceden sabit. Hepsi raporlanır. Raporlanan sayılarda seçim iyimserliği yok.
- **"En iyi backbone" cümlesi:** 6 backbone'un en yükseği, seçim nedeniyle iyimserdir (kazananın laneti; EXP-012'de MFCC görevleri için ~0.02). Bunu ölçmek için **iç içe seçim** tahmini raporlanır (Bölüm 5.7).
- **Faz 3'e geçiş:** Faz 2 sonuçlarına bakarak fine-tune modeli seçmek bir seçimdir. Kural şimdiden yazılı (D-035). Kuralın kendisi dış test tahminlerine değil, iç doğrulama tahminlerine dayanır.

---

## 5. Tasarım kararları

### 5.1 Smoke test (SMK-001)

**Amaç:** Tam çıkarımdan önce, her backbone'un doğru yüklendiğini, doğru girdiyi aldığını ve doğru çıktıyı verdiğini göstermek. Ayrıca Faz 3 için bellek ve hız ölçmek.

Her backbone için kontroller (hepsi geçmeden çıkarım başlamaz):

| # | Kontrol | Geçme ölçütü |
|---|---|---|
| S1 | Checkpoint sha256 | Kaydedilir; `models/MANIFEST.json`'dakiyle aynı (ilk çalıştırmada yazılır). PANNs için Zenodo md5'i indirmede doğrulanır |
| S2 | Sözleşme | SR, mel parametreleri, katman sayısı, gizli boyut, `do_normalize` vb. `configs/model_input_contracts.yaml` ile aynı |
| S3 | `load_state_dict(strict=True)` | Eksik / fazla anahtar = 0 |
| S4 | Parametre sayısı | Kaydedilir (sözleşmedeki yaklaşık değerler doğrulanır) |
| S5 | İleri geçiş | Gerçek bir kaydın pencereleri: çıktı şekli `(pencere, katman, boyut)`, dtype float32, cihaz, sonlu değerler |
| S6 | Eval determinizmi | Aynı girdi iki kez → max fark < 1e-5 |
| S7 | Batch-değişmezlik | Kayıt tek başına vs başka kayıtlarla aynı batch'te → max fark < 1e-4 |
| S8 | Kanca (hook) doğruluğu | Son katman kancası, modelin resmi son çıktısıyla aynı (WavLM Large'da son LayerNorm uygulandıktan sonra) |
| S9 | İşlevsel kontrol (PANNs) | Kelime kayıtlarında ortalama AudioSet çıktısında "Speech" (sınıf 0) ilk 3'te |
| S10 | İşlevsel kontrol (hepsi) | Aynı katılımcının farklı görev kayıtları, farklı katılımcılarınkinden daha benzer (kosinüs). Zayıf bir akıl sağlığı kontrolü: çökmüş (sabit) çıktıları yakalar |
| S11 | Kaydet → yükle → aynı çıktı | Faz 3 checkpoint akışı için |
| S12 | Faz 3 bellek / hız | `train()` modunda (augmentation kapalı, D-016), batch 16 × 4 s, fp16 autocast, 1 AdamW adımı: kayıp sonlu, gradyan sonlu, parametre değişti, GPU tepe belleği ve adım süresi. WavLM Large'da gerekirse gradient checkpointing ile tekrar |
| S13 | Çıkarım hızı | Pencere/saniye → tam çıkarım süresi tahmini |

**Not:** S9 güçlü bir kontroldür. SR veya frontend yanlışsa (ör. 16 kHz sesi 32 kHz modele vermek), "Speech" sınıfı tutarlı biçimde üstte çıkmaz. SSL modellerinin sınıflandırıcısı yok; onlar için S2, S3, S8 ve S10'a güveniyoruz. [INFERENCE]

### 5.2 Girdi: pencereleme

- **KARAR:**
  - 4.0 s pencere, 2.0 s adım (D-015).
  - **Son pencere kaydın sonuna hizalanır.** Örnek: 10.26 s → başlangıçlar 0, 2, 4, 6 ve 6.26 s. Kaydın her anı en az bir pencerede; dolgu yok.
  - **4 s'den kısa kayıtlar (17 kayıt, en kısa 2.28 s) dolgusuz, olduğu uzunlukta tek pencere olarak işlenir.** Bu, D-015'teki "kısa kayıtlar sıfırla doldurulur" kuralından **sapmadır** ve onay gerektirir.
  - Keşifsel ek görünüm (EXP-018): **tüm kayıt tek girdi** (dolgusuz, ~10 s).
- **NEDEN:**
  - 4 s / 2 s: Boll ile karşılaştırılabilirlik ve Faz 3'te aynı segmentasyon. Dondurulmuş-vs-fine-tune karşılaştırması ancak girdi aynıysa temizdir.
  - Sona hizalı son pencere: kalan kısmı atmak (son ≤2 s kaybolur) ya da sıfırla doldurmak (yapay sessizlik) yerine.
  - Dolgusuz kısa kayıtlar: Sıfır dolgusu modelde yapay bir "dijital sessizlik" bölgesi yaratır. PANNs'te log-Mel'i −100 dB'ye düşer; ortalama gömmeyi kaydın **uzunluğuna** göre kaydırır. Kayıt süresi sesle ilgisiz bir ipucu olabilir (kişi, seans, prosedür farkı). Faz 2'de batch zorunluluğu yok (her kayıt kendi başına işlenir), bu yüzden dolguya gerek de yok. Üç modelin üçü de değişken uzunluğu destekler. [FROM OFFICIAL DOCS + INFERENCE]
  - Tüm kayıt görünümü (keşifsel): Kayıtların neredeyse hepsi ~10 s. PANNs ve BEATs, AudioSet'in 10 s'lik klipleriyle eğitildi. Yani tüm kayıt, ön-eğitim koşuluna 4 s pencereden daha yakın. Boll'un 4 s seçimi ablasyonsuz. Ek maliyet kayıt başına bir ileri geçiş. Sonuç Faz 3'ün girdi tasarımına bilgi verir; Faz 2'de seçim için kullanılmaz.
- **ALTERNATİFLER:**
  - (a) D-015'e aynen uymak: kısa kayıtları sıfırla doldurmak. 17 kayıtta süre ipucu riski.
  - (b) Döngüsel dolgu (kaydı tekrarlamak): yapay periyodiklik.
  - (c) < 4 s kayıtları atmak: 17 kayıt kaybı; kayıp rastgele olmayabilir.
  - (d) Birincil görünümü tüm kayıt yapmak: ön-eğitime daha uygun ama Boll ve Faz 3 ile tutarsız. Bu yüzden keşifsel.
- **RİSK:**
  - 17 kısa kayıt diğerlerinden daha kısa bağlam görür. Etkisi küçük olmalı (%0.7). [INFERENCE]
  - Faz 3'te batch eğitimi için kısa kayıtlar yine bir karar gerektirecek (dolgu + maske ya da batch 1). Faz 3 tasarımında ayrıca ele alınır.
- **BİLİMSEL SONUÇ:** Gömmeler yapay içerik barındırmaz. Kayıt süresi, dolgu yoluyla temsile sızmaz.
- **UYGULAMA:** `extract_embeddings.py` → `window_starts()`; birim testle doğrulanır.

### 5.3 Katman ve havuzlama

- **KARAR:**
  - **PANNs:** resmi `embedding` çıktısı (fc1 + ReLU; CNN10 512, CNN14 2048). Tek katman.
  - **BEATs ve WavLM:** her transformer bloğunun çıkışı forward hook ile yakalanır.
    - Katman 0 = ilk bloğun girdisi; katman i = i'inci bloğun çıkışı.
    - Her katmanda zaman (ve BEATs'te frekans yaması) üzerinden ortalama alınır.
    - **Hepsi saklanır:** BEATs ve Base+ 13 × 768, Large 25 × 1024.
  - **Birincil temsil (önceden sabit): katman 1..L'nin eşit ağırlıklı ortalaması.**
    - Her katman önce **eğitim fold'unda** z-skoruna çevrilir, sonra katmanlar ortalanır, sonra LR.
    - Bu adım Pipeline içindedir (`LayerAverage`); test verisini görmez.
  - **Katman katman sonuçlar (EXP-017):** yalnız keşifsel, tekrar 0, hiçbir seçimde kullanılmaz.
- **NEDEN:**
  - Öz-gözetimli konuşma modellerinde son katman, ön-eğitim hedefine özelleşir. Paralinguistik bilgi (ses kalitesi, konuşmacı, duygu) genellikle orta katmanlarda yoğunlaşır. [FROM PAPER: Pasad ve ark. 2021, wav2vec 2.0 katman analizi; Chen ve ark. 2022, WavLM — SUPERB'de katmanların ağırlıklı toplamı]
  - Yalnız son katmanı almak SSL modellerini sistematik olarak dezavantajlı kılabilir. Katmanı dış sonuçlara bakarak seçmek ise seçim iyimserliği yaratır (D-033).
  - Eşit ağırlıklı ortalama, SUPERB'deki öğrenilmiş ağırlıklı toplamın seçimsiz, önceden sabitlenebilen en basit hâlidir.
  - Z-skoru gerekli: WavLM Large "stable layer norm" kullanır; ara katmanların ölçeği çok farklıdır. Ham ortalamada büyük normlu katmanlar baskın olur.
  - Katman 0 hariç: transformer öncesi yerel özellikler. Birincil temsil "transformer katmanları"dır; katman 0 EXP-017'de ayrıca görülür.
- **ALTERNATİFLER:**
  - (a) Yalnız son katman: basit ama SSL modelleri aleyhine yanlı olabilir.
  - (b) En iyi katmanı dış CV'den seçmek: seçim iyimserliği.
  - (c) En iyi katmanı iç CV'de seçmek (iç içe): geçerli, ama 25 katman × 7 görev × 25 fold'da kararsız ve pahalı. EXP-015, küçük örneklemde seçimin kararsız olduğunu gösterdi.
  - (d) Bütün katmanları uç uca eklemek: Large'da 25 600 boyut, ~270 örnek. Çok yüksek boyut.
  - (e) Öğrenilmiş ağırlıklı toplam: lineer prob değil; Faz 3'ün doğal parçası.
  - Havuzlama için: ortalama + standart sapma (boyut iki katı), dikkat havuzlama (öğrenilen parametre). Ortalama en basit ve en yaygın olanı.
- **RİSK:**
  - Eşit ortalama, bilgi taşıyan birkaç katmanı bilgisiz katmanlarla seyreltebilir. EXP-017 bunu gösterir. Birincil sonucu değiştirmez.
  - Ortalama havuzlama, kayıt içindeki kısa olayları (ör. nefes alma, ses kırılması) seyreltir. [HYPOTHESIS]
- **BİLİMSEL SONUÇ:** Altı backbone, seçim iyimserliği olmadan, aynı kurallarla karşılaştırılır.
- **UYGULAMA:** `backbones.py` (kancalar), `run_probes.py` → `LayerAverage`.

### 5.4 Prob (sınıflandırıcı)

- **KARAR:** EXP-011'in LR protokolü + D-033:
  - `LayerAverage` (yalnız SSL) → `StandardScaler` → `LogisticRegression(class_weight="balanced", max_iter=5000)`.
  - C, split dosyasındaki iç 5-fold'da `neg_log_loss` ile seçilir. Izgara **10⁻⁵ … 10²** (8 değer); bütün temsiller için aynı. MFCC dahil.
  - Görev başına ayrı model; katılımcı skoru = 7 görev olasılığının ortalaması (D-032).
  - Eşik: her dış fold'da, eğitim katılımcılarının iç 5-fold OOF tahminlerinde dengeli doğruluğu en yükselten değer (D-033). Füzyon için eşik, iç OOF füzyon skorlarından.
  - Kalibrasyon: Brier, ortalama tahmin − gerçek oran, kalibrasyon eğimi (D-033).
  - C'nin ızgaranın ucuna düştüğü fold oranı raporlanır (ızgara yeterli mi?).
- **NEDEN:**
  - MFCC ile aynı sınıflandırıcı → temsil farkı izole.
  - Izgara genişletildi: 2048 boyut ve ~220 eğitim örneğinde (p ≫ n) en iyi C, EXP-011'in alt sınırı 0.001'in altında olabilir. Çok küçük C'de L2-LR, sınıf ortalamaları farkı yönüne yakın bir sınıflandırıcıya dönüşür; p ≫ n'de bu genellikle iyi çalışır. [INFERENCE] MFCC için ızgaranın genişlemesi zararsız (seçilmezse etkisi yok).
  - Ağırlıklı BCE / sınıf ağırlığı, EXP-011 ile tutarlılık için (D-033 madde 1).
- **ALTERNATİFLER:**
  - Doğrusal olmayan prob (MLP) gömmeler üzerinde: daha güçlü ama artık "temsil" değil "temsil + öğrenilen dönüşüm" ölçülür; seçenek sayısı artar.
  - PCA + LR: ek bir seçim (bileşen sayısı).
  - Görevleri birleştirip tek prob (7 × daha çok kayıt): MFCC protokolünden farklı olur; Faz 3'te düşünülebilir.
- **RİSK:**
  - LR, gömmelerdeki doğrusal olmayan bilgiyi kaçırabilir. Bu, gömmeler **aleyhine** bir yanlılıktır. MFCC tarafında ise bir de MLP tabanı var (D-032). Karşılaştırma bu yüzden muhafazakârdır. [INFERENCE]
  - Sınıf ağırlığı kalibrasyonu kaydırır (EXP-013) → raporlanır.
- **BİLİMSEL SONUÇ:** RQ1'de tek değişken temsildir.
- **UYGULAMA:** `run_probes.py`.

### 5.5 MFCC kolları aynı kodla yeniden çalıştırılır

- **mfcc_lr:** EXP-011'deki LR, yeni ızgara ve D-033 eşiğiyle, aynı split'lerde.
- **mfcc_mlp:** D-032'nin ikinci tabanı: StandardScaler → SMOTE(42) → varsayılan MLPClassifier(42). Izgara yok. Eşik iç OOF'tan.
- **Neden yeniden?** Eşleştirilmiş karşılaştırma aynı fold'larda, aynı kodla, aynı eşik kuralıyla yapılmalı. Ayrıca bir regresyon testi: mfcc_lr füzyonu EXP-011 LR füzyonunu (0.771 ± 0.078) ±0.01 içinde yeniden üretmeli. Üretmezse önce kod incelenir, sonra yorum.

### 5.6 Karşılaştırmalar ve istatistik

**Onaylayıcı aile (RQ1, 6 hipotez):**
- Her backbone b için, füzyon düzeyinde:
  - H0(b, LR): AUC(b) = AUC(mfcc_lr);
  - H0(b, MLP): AUC(b) = AUC(mfcc_mlp).
- "b, MFCC'den iyi" iddiası **iki tabanı da** geçmeyi gerektirir (D-032). İstatistiksel olarak bu bir kesişim-birleşim testidir: p(b) = max(p_LR, p_MLP). Bu, ek düzeltme gerektirmeden α düzeyini korur. [FROM PAPER: Berger 1982]
- Altı backbone arasında **Holm** düzeltmesi.
- Test: 25 fold'un eşleştirilmiş ΔAUC'sinde **Nadeau–Bengio düzeltilmiş t-testi** (D-011; df = 24, test/train oranı gerçek fold boyutlarından).
- **Tek yönlü** p (H1: Δ > 0), eşik 0.025. İddia yönlü ("daha iyi"). İki yönlü p kullanılsaydı, MFCC'den anlamlı biçimde *kötü* bir backbone Holm sırasının başına geçer ve diğer backbone'ların düzeltmesini gevşetirdi. 0.025 eşiği, iki yönlü 0.05'in pozitif yarısıyla aynı katılıktadır. İki yönlü p'ler de raporlanır.
- Destek: tekrarlar boyunca ortalama OOF olasılığında **katılımcı bootstrap'lı** ΔAUC %95 CI (2 000 tekrar) ve tekrar başına **DeLong** eşleştirilmiş testi (D-011).
- **Karar kuralı:** "Destekleniyor (üst sınır)" ancak Holm-düzeltilmiş tek yönlü p < 0.025 **ve** iki bootstrap CI'ı da 0'ı dışlıyorsa. D-011'in ikinci koşulu (yönün zaman-örtüşen alt kohortta korunması) D-028 ile değerlendirme aşamasına ertelendi. Bu yüzden Faz 2'nin her "destekleniyor"u **geçicidir**.

**Keşifsel (önceden listelenmiş; her aile içinde Holm, ama yorum keşifsel):**
- RQ3: 15 backbone çifti, füzyon ΔAUC (D-008'in cnn14 − cnn14_16k'sı ve ölçek karşılaştırması wavlm_large − wavlm_base_plus bu ailenin içinde).
- RQ4: görev başına ΔAUC (backbone − mfcc_lr), 42 karşılaştırma.
- RQ5: ünlü (aaa) − kelimelerin ortalaması, kol başına fold düzeyinde.
- EXP-017: katman eğrileri. EXP-018: tüm kayıt − 4 s pencere.

**Güç (önceden, dürüstçe):**
- Havuzlanmış AUC ≈ 0.80, 283 / 59 için Hanley–McNeil SE ≈ 0.03. [INFERENCE]
- İki ses temsilinin skorları arasındaki korelasyon ρ = 0.5–0.7 varsayılırsa, ΔAUC'nin SE'si ≈ 0.023–0.030. [INFERENCE]
- %80 güçle saptanabilir fark ≈ 0.065–0.085; Holm (6 test) ile ≈ 0.08–0.10.
- **Yani:** MFCC-LR füzyonu 0.77 iken, bir backbone'un "anlamlı biçimde iyi" çıkması için füzyon AUC'sinin kabaca **≥ 0.85** olması gerekir. 0.02–0.05'lik farklar bu örneklemde çözülemez. "Anlamlı değil" sonucu "fark yok" demek değildir.

### 5.7 Seçim iyimserliğinin ölçülmesi (iç içe seçim)

- Her dış fold'da, backbone'lar arasından **iç OOF füzyon AUC'si** en yüksek olan seçilir. Dış test füzyon AUC'si kaydedilir.
- 25 fold'un ortalaması = "en iyi backbone'u seç, sonra kullan" **prosedürünün** dürüst performansı.
- "Sonradan bakarak en iyi sabit backbone" ile arasındaki fark = kazananın laneti.
- İki kapsamda: 6 backbone ve D-035'in 3 adayı.

### 5.8 Kayıt bağlamı: Faz 2'de neyi kaydediyoruz? (D-028 yükümlülükleri)

| Yükümlülük | Faz 2'de |
|---|---|
| Katılımcı düzeyi OOF tahminleri | ✓ füzyon skoru, her (tekrar, fold) |
| Kayıt düzeyi OOF tahminleri | ✓ görev başına = kayıt başına |
| Segment düzeyi | Uygulanamaz: prob kayıt düzeyinde eğitilir. Pencere gömmeleri `--save-windows` ile isteğe bağlı saklanır |
| En iyi / son epoch | Uygulanamaz (eğitim yok) |
| Gömmeler | ✓ hepsi (bütün katmanlar), Drive `embeddings_v1/` |
| Split hash'leri | ✓ her rapor ve registry satırında |
| Üst sınır uyarısı + 2 referans çizgisi | ✓ bağlam ve yaş, aynı fold'larda |
| İzleme göstergeleri (D-034 madde 8) | EXP-014'teki üç "bağlam bağımlılığı" ölçüsü, etiket sabitken füzyon skorunun bağlamı ayırma gücü (0.5 = bağımlılık yok): hastalarda geç vs erken, hastalarda sabah vs öğleden sonra, sağlıklılarda sabah vs öğleden sonra. MFCC-LR'de 0.556 / 0.524 / 0.464; yalnız-bağlam referansında 0.939 / 0.739 / 1.000. Yalnız raporlanır |

**CONFOUND_CONTROL_DESIGN'daki EXP-021 ile ilişkisi:** O belgedeki EXP-021 (dondurulmuş prob + lens seti L1–L8) D-028 ile ertelendi. Lens seti, değerlendirme aşamasında **EXP-016'nın kaydedilmiş tahminleri ve gömmeleri** üzerinde uygulanır. Yeni bir eğitim gerektirmez.

### 5.9 Hesap ayrıntıları

- **Hassasiyet:** çıkarım fp32. TF32 kapalı, `cudnn.deterministic=True`, `benchmark=False`. Dondurulmuş çıkarım ucuz; fp16'nın küçük sayısal farkları gereksiz bir değişken olur.
- **Saklama** (Drive `data_derived/embeddings_v1/<backbone>/`; katılımcı düzeyi → git'e girmez, D-014):
  - `rec_win4.npy` float32 `[kayıt, katman, boyut]` (pencere ortalaması);
  - `rec_full.npy` float32 (tüm kayıt görünümü);
  - isteğe bağlı `win4_windows.npy` float16 + `windows.csv`;
  - `recordings.csv`, `info.json` (backbone, checkpoint sha256, sözleşme, sürümler, git commit, önbellek index sha256), `MANIFEST.sha256`.
  - Boyut tahmini: Large ~0.5 GB, diğerleri toplam ~0.4 GB. [INFERENCE]
- **Kaldığı yerden devam:** 100 kayıtlık parçalar (`shards/`) atomik yazılır (önce `.tmp`, sonra yeniden adlandırma). Yeniden çalıştırmada biten parçalar atlanır. Problarda (tekrar, backbone) başına `partial/`.
- **Süre tahmini:** GPU çıkarımı backbone başına 2–10 dk, toplam < 1 saat (T4). Problar CPU'da: EXP-016 birkaç saat, EXP-017 ~1–2 saat. [INFERENCE — SMK-001 S13 ve ilk tekrar ölçer]
- **Ağırlıklar:** Drive `models/`. PANNs Zenodo'dan (md5 doğrulamalı), WavLM HF'den (çözümlenen commit hash'i kaydedilir, `save_pretrained` ile Drive'a kopya), BEATs OneDrive'dan **elle** (Bölüm 8).

### 5.10 Faz 3'e hangi backbone'lar geçer? (D-035, sonuçlardan önce)

- **KARAR:**
  1. **Sabit:** PANNs **CNN10** (pretrained). Neden: RQ2'nin eşi. Faz 3'teki "sıfırdan CNN10" ancak aynı mimarinin pretrained hâliyle karşılaştırılırsa ön-eğitimin etkisi mimariden ayrılır.
  2. **Sabit:** PANNs **CNN14** (32 kHz). Neden: Boll'un en iyi modeli (konuşmada AUC 0.93); karşılaştırılabilirlik.
  3. **Kurala bağlı:** PANNs dışı adaylardan (BEATs, WavLM Base+, WavLM Large) **bir** tanesi.
     - Ölçüt: **doğrulama skoru** = 25 dış fold'un her birinde, yalnız eğitim katılımcılarının iç 5-fold OOF füzyon AUC'si; bunların ortalaması. Dış test tahminleri ölçüte girmez.
     - Eşitlik: en yüksek skora 0.01'den yakın olanlar arasından **en az parametreli** olan (BEATs ~90 M < Base+ ~95 M < Large ~316 M).
  4. Bağlam izleme göstergeleri ölçüt **değildir**; seçilen modelinki ayrıca raporlanır.
  5. Faz 3, Faz 2'nin sonucu ne olursa olsun yapılır (hiçbir backbone MFCC'yi geçmese de). Faz 3'ün sorusu farklı: fine-tune ve ön-eğitim ne kazandırır (RQ2)?
  6. Faz 3 hesap bütçesi (SMK-001 S12 ölçümleriyle) yetmezse ilk çıkarılan **CNN14**'tür. CNN10 RQ2 için, PANNs dışı aday RQ3 için gereklidir.
- **NEDEN:**
  - Seçimi sonuçlardan önce yazmak, "sonuca bakıp en iyisini seçtik" iyimserliğini sınırlar (D-033 madde 5).
  - İç doğrulama skoru, dış test fold'larını seçime katmaz. Tam bağımsız değildir: tekrarlı CV'de her katılımcı başka bir fold'un eğitim kümesindedir. Kalan iyimserlik Bölüm 5.7'deki iç içe seçim tahminiyle ölçülür. [INFERENCE]
  - Seçim nedeniyle, seçilen backbone'un dondurulmuş skoru biraz şişkin olabilir. Bu, Faz 3'teki "fine-tune − dondurulmuş" farkını **küçültür** (ortalamaya dönüş). Yani "fine-tune işe yarıyor" iddiası aleyhine çalışır; muhafazakâr. [INFERENCE]
  - Eşitlikte küçük model: T4 bütçesi ve küçük veride aşırı uyum riski.
- **ALTERNATİFLER:**
  - (a) Dış havuzlanmış füzyon AUC'siyle seçmek: daha basit ama seçimi test fold'larına bağlar.
  - (b) CONFOUND_CONTROL_DESIGN'daki öneri (D-026, ertelendi): **artımlı test T1'e göre** seçmek (bağlamın ötesinde bilgi). Bilimsel olarak daha güçlü bir ölçüt: "en çok kestirme öğreneni seçme" riskini azaltır. Ama D-028 bağlam değerlendirmesini model geliştirme sonuna bıraktı. T1'i şimdi seçim için kullanmak, ertelenen testi öne çekmek demek.
  - (c) Her dış fold'da ayrı backbone fine-tune etmek (tam iç içe): pahalı ve yorumu zor.
  - (d) Bütün backbone'ları fine-tune etmek: 40+ GPU saati (D-012).
- **RİSK:**
  - Ölçüt E1 (tam kohort) temelli. Bağlamı en iyi öğrenen backbone seçilebilir. Azaltma: (i) PANNs iki model seçimden bağımsız sabit; (ii) altı backbone'un **hepsinin** dondurulmuş tahminleri saklanır ve değerlendirme aşamasında lens setinden geçer; seçim bağlam kaynaklıysa orada görünür; (iii) izleme göstergeleri raporlanır.
  - İç OOF tahminleri, C'nin seçildiği aynı iç fold'lardan gelir. Doğrulama skoru bu yüzden hafif iyimserdir; iyimserlik boyuta ve katman sayısına göre kollar arasında biraz değişebilir. [INFERENCE]
  - Sen (b)'yi tercih edersen: T1'in (D-025) kesinleştirilmesi ve EXP-016 tahminleri üzerinde yalnız seçim için çalıştırılması gerekir. Bu bir tasarım değişikliğidir; ayrıca konuşalım.
- **BİLİMSEL SONUÇ:** Faz 3'e geçiş, Faz 2 sonuçları görülmeden yazılmış bir kurala bağlı.
- **UYGULAMA:** `run_probes.py` EXP-016 raporunda doğrulama skorlarını ve kuralın seçtiği backbone'u otomatik yazar. Karar yine de sana sunulur.

---

## 6. Önceden yazılmış beklentiler [HYPOTHESIS]

Bunlar sonuçlar görülmeden yazıldı. Yanlış çıkmaları da raporlanır.

| # | Beklenti | Gerekçe |
|---|---|---|
| B1 | Backbone füzyon AUC'leri 0.70–0.85 aralığında; en az birinin MFCC-LR'yi (0.77) sayısal olarak geçmesi olası, ama **onaylayıcı testi geçen olmaması** daha olası | Güç hesabı (Bölüm 5.6); MFCC füzyonu zaten yüksek |
| B2 | Dondurulmuş gömmelerin bağlam izleme göstergeleri MFCC'ninkinden yüksek | Daha zengin temsil oda / cihaz / saat bilgisini de taşır |
| B3 | cnn14 − cnn14_16k: |Δ| < 0.03 | 32k yolu zaten 11 kHz'e kesildi; iki model arasındaki bant farkı 11 vs 8 kHz |
| B4 | WavLM'de katman eğrisi ortada tepe yapar; son katman ortalamadan kötü | Pasad ve ark. 2021; Chen ve ark. 2022 |
| B5 | Görevler arasında güvenilir bir sıralama çıkmaz | EXP-011/012'de MFCC için de çıkmadı |
| B6 | Tüm kayıt vs 4 s pencere: |Δ| < 0.02 | Kayıtlar ~10 s; ortalama havuzlama iki görünümde benzer bilgi toplar |
| B7 | WavLM Large, Base+'tan anlamlı biçimde iyi değil | 342 katılımcıda ölçek avantajı görünmeyebilir |

---

## 7. Çıktılar

| Ne | Nerede | Git'e girer mi? |
|---|---|---|
| SMK-001 raporu | `reports/phase2/SMK-001_smoke.{md,json}` | Evet (agrega) |
| Gömmeler | Drive `data_derived/embeddings_v1/` | Hayır |
| OOF ve iç OOF tahminleri, fold metrikleri | Drive `experiments/EXP-016_frozen-probe/` vb. | Hayır |
| EXP-016/017/018 raporları | `reports/frozen/` | Evet (agrega) |
| Registry satırları | `results/registry.csv` | Evet |

---

## 8. Senin yapman gerekenler (sırasıyla)

1. **BEATs ağırlığını indir.** `github.com/microsoft/unilm/tree/master/beats` README'sindeki tabloda **"BEATs_iter3+ (AS2M)"** bağlantısı (fine-tuned **olmayan**, üçüncü sütun). OneDrive'dan indir, Drive'da `MyDrive/asthma-voice/models/BEATs_iter3_plus_AS2M.pt` adıyla kaydet. Dosya ~350 MB civarında olmalı. [NEEDS VERIFICATION: boyut]
2. **D-034 ve D-035'i onayla ya da değiştir** (Bölüm 9).
3. `notebooks/20_smoke_tests.ipynb` (GPU, T4): PANNs ve WavLM ağırlıklarını Drive'a indirir, testleri ve SMK-001'i çalıştırır, raporu bir dala gönderir.
4. SMK-001 raporunu birlikte okuyalım. Hepsi geçerse:
5. `notebooks/21_frozen_embeddings.ipynb`: önce GPU'da gömme çıkarımı, sonra (aynı ya da CPU oturumunda) problar.

---

## 9. Onay bekleyen kararlar

| Karar | Özet | Neden onay gerekiyor |
|---|---|---|
| **D-034** | Dondurulmuş gömme protokolü: pencereleme (kısa kayıtlar dolgusuz — D-015'ten sapma), katman ortalaması, prob, MFCC kolları, onaylayıcı aile, iç içe seçim tahmini, bağlam izleme göstergeleri | D-015'i değiştiriyor; onaylayıcı aileyi sabitliyor |
| **D-035** | Faz 3 backbone seçim kuralı, Faz 2 sonuçlarından **önce**: CNN10 (RQ2 eşi) + CNN14 (Boll) sabit; PANNs dışı tek backbone iç doğrulama füzyon AUC'siyle | Seçimi şimdiden bağlıyor; CONFOUND belgesindeki "T1'e göre seç" önerisinden (D-026, ertelendi) ayrılıyor |

**Uygulama düzeltmesi (karar değil):** `model_input_contracts.yaml`'daki CNN14_16k notu yanlıştı. "Cnn14 sınıfı bu parametrelerle" değil, resmi kodda **ayrı bir `Cnn14_16k` sınıfı** var; sınıf kendi içinde SR = 16000, pencere 512, adım 160, 64 mel, 50–8000 Hz assert ediyor. [FROM OFFICIAL DOCS: `pytorch/models.py`, commit d2f4b8c]
