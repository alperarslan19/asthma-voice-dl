# Faz 2 tasarımı: smoke test, dondurulmuş gömme, lineer prob

**Tarih:** 2026-10-09 (12. tur) · **Revizyon:** 13. tur (Alper'in soruları üzerine) · **Durum:** D-034 **KABUL** (13. tur; birincil kısa kayıt politikası `zeropad`) · D-035 **KABUL** (Alper'in koşuluyla: seçim fold başına) · Son ek (13. tur, ikinci bölüm): Holm eşiği, fold bağımlılığı, SMK-001 başarısızlık protokolü, ön-eğitim kaynakları
**Kod:** `scripts/backbones.py`, `scripts/smoke_test_backbones.py`, `scripts/extract_embeddings.py`, `scripts/run_probes.py`, `scripts/report_metadata_monitor.py`, `tests/test_phase2.py` · **Notebook'lar:** `20_smoke_tests.ipynb`, `21_frozen_embeddings.ipynb`

**Bu belge gerçek verideki sonuçlar görülmeden yazıldı.** Faz 2'de hiçbir model henüz bu veride çalıştırılmadı. Bu belgede performans hakkında hiçbir çıkarım yoktur.

---

## 0. Faz 2'nin projedeki yeri

- **Ana hedef Faz 3'tür:** ham ses üzerinde, önceden eğitilmiş modelle uçtan uca fine-tune.
- **Faz 2, Faz 3'ün yerine geçmez.** Faz 3'ten önce yapılan **kontrollü bir temsil karşılaştırmasıdır**: modeller değiştirilmez, yalnız "hazır temsil astım / sağlıklı bilgisini ne kadar taşıyor?" sorusu sorulur.
- Faz 2 sonucu ne olursa olsun Faz 3 yapılır (D-035 madde 4). Faz 2'nin Faz 3'e kattıkları:
  1. fine-tune'un karşılaştırılacağı "dondurulmuş" taban (aynı fold'larda, aynı girdiyle);
  2. Faz 3'te hangi PANNs dışı backbone'un fine-tune edileceği (D-035, fold başına);
  3. Faz 3 için bellek ve hız ölçümü (SMK-001 S12).
- Bu yüzden Faz 2'nin girdi biçimi Faz 3'ünkiyle **aynı** olmalı: 4 s / 2 s pencere ve kısa kayıtlar için aynı politika (Bölüm 3).

**Benzetme:** Altı farklı uzman aynı ses kaydını dinleyip birer "not kâğıdı" dolduruyor (gömme). Uzmanlara astım hakkında hiçbir şey öğretmiyoruz; yalnız kâğıtlardaki notlardan basit bir kuralla (lineer model) astımı tahmin edebiliyor muyuz diye bakıyoruz. Faz 3'te ise uzmanlara astımı öğreteceğiz (fine-tune).

**Altı backbone ve iki MFCC tabanı:**

| Kısa ad | Model | Ön-eğitim | SR | Gömme |
|---|---|---|---|---|
| cnn10 | PANNs CNN10 | AudioSet, gözetimli (etiketli) | 32 kHz | 512 (fc1) |
| cnn14 | PANNs CNN14 | AudioSet, gözetimli (etiketli) | 32 kHz | 2048 (fc1) |
| cnn14_16k | PANNs CNN14_16k | AudioSet, gözetimli (etiketli) | 16 kHz | 2048 (fc1) |
| beats | BEATs iter3+ (AS2M), fine-tune edilmemiş | AudioSet; maskeli tahmin, ama tokenizer öğretmeni **AudioSet etiketleriyle fine-tune edilmiş** → etiket dolaylı yoldan girer | 16 kHz | 12 katman × 768 |
| wavlm_base_plus | WavLM Base+ | 94 bin saat İngilizce konuşma, öz-gözetimli (etiket / transkript yok) | 16 kHz | 12 katman × 768 |
| wavlm_large | WavLM Large | aynı | 16 kHz | 24 katman × 1024 |
| mfcc_lr / mfcc_mlp | 44 MFCC özelliği (EXP-011) | — | 22.05 kHz | 44 |

Ayrıntı ve yorum sınırı: Bölüm 9.6. **Faz 2, ön-eğitim yöntemlerinin (gözetimli / öz-gözetimli) saf bir karşılaştırması değildir; farklı hazır ses temsillerinin karşılaştırmasıdır.**

**Deneyler ve raporlar:**

| ID | Ne | Rol |
|---|---|---|
| (birim testleri) | `tests/test_phase2.py`, rastgele küçük modeller, CPU | Kod yolları; **SMK-001 değildir** |
| SMK-001 | Gerçek checkpoint'lerle smoke test, GPU | Çıkarımın ön koşulu |
| EXP-016 | 6 backbone + 2 MFCC tabanı, 5 × 5 CV, katılımcı düzeyi füzyon | **Onaylayıcı** (Bölüm 2.1 R6) + önceden listelenmiş keşifsel |
| EXP-016S | Kısa kayıtlarda diğer dolgu politikası, tekrar 0 | Önceden belirlenmiş duyarlılık |
| META-016 | Bağlam bağımlılığı, süre analizi, yalnız-süre referansı, dışlama duyarlılığı | **Ayrı** meta veri / confounder raporu; ana akışa girmez |
| EXP-017 | Katman katman prob (BEATs, WavLM), tekrar 0 | Keşifsel; **hiçbir seçimde kullanılmaz** |
| EXP-018 | Tüm kayıt tek girdi vs 4 s pencereler | Keşifsel |

---

## 1. Gerçek checkpoint'li smoke test ile rastgele küçük modelli testler AYRI şeylerdir

| | Birim testleri (`tests/test_phase2.py`) | SMK-001 (`20_smoke_tests.ipynb`) |
|---|---|---|
| Model ağırlıkları | **Rastgele**, küçük yapılandırma (ör. 2 katman × 64) | **Resmi checkpoint'ler** (Zenodo, HF, OneDrive) |
| Veri | Sentetik ses ve sentetik gömmeler | Gerçek önbellek (8 katılımcı × 7 görev) |
| Donanım | CPU (yerel ya da Colab) | Colab GPU (T4) |
| Neyi gösterir | Kod yolları doğru: şekiller, kancalar, dolgu yolu, sızıntı assert'leri, istatistik fonksiyonları, D-035'in dış testten bağımsızlığı, rapor biçimi | Resmi ağırlık eksiksiz yükleniyor, resmi girdi biçimi, determinizm, batch-değişmezlik, PANNs "Speech" kontrolü, aynı-kişi benzerliği, Faz 3 bellek |
| Neyi **göstermez** | Modellerin doğru yüklendiğini ya da anlamlı çıktı verdiğini | Performansı (bilerek) |
| Rapor adı | `UNITTEST_random_init_smoke.*` (geçici klasörde; repoya girmez) | `reports/phase2/SMK-001_smoke.*` |

- Birim testlerinin hepsi geçti (P1–P9, P7b, P7c). Bu, **SMK-001'in geçtiği anlamına gelmez**.
- Notebook 21, gömme çıkarımından önce `SMK-001_smoke.json`'ı okur; `random_init` ise ya da herhangi bir backbone PASS değilse durur.
- WavLM işlemcilerinin `return_attention_mask` değerleri (Base+ false, Large true) sözleşmede `expect` alanında; SMK-001 gerçek dosyalarla assert eder. [NEEDS VERIFICATION]

### 1.1 SMK-001 başarısız olursa (önceden yazılmış protokol)

1. **EXP-016'ya (ve gömme çıkarımına) geçilmez.** Notebook 21 bunu zorunlu kılar: `SMK-001_smoke.json`'da `random_init` ya da PASS olmayan bir backbone varsa durur.
2. **Önce neden incelenir**, sonra düzeltilir. Rapor her kontrolün değerini JSON'a yazar (ör. eksik anahtar listesi, sha256, en büyük fark, "Speech" sırası). Olası nedenler ve ilk bakılacak yer:
   - S1 sha256 farkı → ağırlık dosyası değişmiş / yanlış dosya; yeniden indirme ve Zenodo md5'i.
   - S2 / S3 → sözleşme ya da sürüm uyumsuzluğu (ör. torchlibrosa, transformers anahtar adları); yükleme yöntemi `info.load`.
   - S6 / S7 / S11 → determinizm ya da batch bağımlılığı (eval modu, GPU algoritması); toleransın kendisi değil, nedeni incelenir.
   - S8 → kanca tanımı ile resmi çıktı arasında fark (ör. son LayerNorm).
   - S9 → yanlış SR / frontend (PANNs) — en ciddi işaret.
   - S10 → çökmüş (sabit) temsil.
   - S12 → bellek yetmezliği (WavLM Large; gradient checkpointing denemesi raporda).
   - S14 → dolgu yolu / maske hatası.
3. Neden ve düzeltme EXPERIMENTS.md'de SMK-001 kaydına yazılır. **Eşik gevşetilerek "geçirme" yapılmaz**; bir eşiğin yanlış konduğu düşünülürse bu ayrı bir karar kaydıyla ve gerekçesiyle değiştirilir.
4. SMK-001 düzeltmeden sonra **baştan** çalıştırılır; önceki başarısız rapor git geçmişinde kalır (olumsuz sonuç silinmez).
5. SMK-001 raporu birim testlerinden ayrı yazılır ve ayrı commit edilir (`reports/phase2/SMK-001_smoke.*`).

---

## 2. Gerçek veride sonuç görülmeden kilitlenen kurallar ve sonuçlara göre yapılabilecek keşifsel analizler

### 2.1 Kilitli kurallar (sonuç görülmeden; değiştirmek yeni bir D-kaydı ve "sonuç görüldükten sonra" notu gerektirir)

| # | Kural | Ayrıntı |
|---|---|---|
| R1 | **Aynı dış fold'lar, aynı birim.** Bütün backbone'lar ve iki MFCC tabanı `outer_r0–4.csv`'nin aynı katılımcı düzeyi dış fold'larında değerlendirilir. Birincil değerlendirme birimi **katılımcı** (7 görev olasılığının ortalaması = füzyon, D-032) | Bölüm 7 |
| R2 | **Katılımcı ayrımı.** Bir katılımcının 7 kaydı da (ve kayıtlarının bütün pencereleri) aynı role düşer: ya eğitim ya dış test. İç fold'lar da katılımcı düzeyinde | Bölüm 7 |
| R3 | **Girdi.** 4 s / 2 s pencere, son pencere sona hizalı. Kısa kayıt politikası D-034 madde 1 (onayına bağlı; Bölüm 3). Diğer politika EXP-016S'de duyarlılık | Bölüm 3 |
| R4 | **Temsil.** PANNs: fc1. SSL: katman 1..L'nin, her katman eğitim fold'unda z-skoruna çevrildikten sonraki eşit ortalaması. **Katman seçimi yok** | Bölüm 4 |
| R5 | **Prob.** LR (class_weight=balanced); C ∈ 10⁻⁵…10², iç 5-fold `neg_log_loss`; eşik iç 5-fold OOF'tan (D-033); görev başına model | Bölüm 6.2 |
| R6 | **Onaylayıcı aile.** 6 hipotez (backbone füzyonu > MFCC). Tek yönlü Nadeau–Bengio; kesişim-birleşim (LR ve MLP); Holm (6); karar: Holm p < 0.025 **ve** iki bootstrap CI'ının alt sınırı > 0 | Bölüm 5 |
| R7 | **Keşifsel liste (önceden sabit).** 15 backbone çifti (yorum: hazır temsillerin farkı, ön-eğitim yöntemi farkı değil; Bölüm 9.6); görev başına Δ (42); ünlü − kelimeler; EXP-017 katman eğrileri; EXP-018 tüm kayıt − 4 s; EXP-016S. Her aile içinde Holm. Yorum keşifsel | Bölüm 5.4 |
| R8 | **D-035.** Faz 3'teki PANNs dışı backbone her dış fold'da, yalnız o fold'un eğitim verisindeki iç doğrulamayla seçilir | Bölüm 6 |
| R9 | **Meta veri / confounder karşılaştırmaları ayrı.** Yalnız META-016'da. Hiçbir karar, seçim ya da onaylayıcı sonuç bunlara dayanmaz. Ana tabloda yalnız D-028'in iki referans satırı (bağlam, yaş) var; onlarla test yapılmaz | Bölüm 8 |
| R10 | **Beklentiler** (B1–B7, Bölüm 10) sonuçtan önce yazıldı; yanlış çıkarsa da raporlanır | Bölüm 10 |
| R11 | **SMK-001 PASS** olmadan çıkarım yok; birim testleri yerine geçmez | Bölüm 1 |
| R12 | **Regresyon testi.** mfcc_lr füzyonu EXP-011 LR füzyonunu (0.771) ±0.01 içinde yeniden üretmeli; üretmezse önce kod incelenir, yorum yapılmaz | Bölüm 6.3 |

### 2.2 Sonuçlara bakarak yapılabilecek keşifsel analizler (izinli, ama etiketli)

Gerçek sonuçlar görüldükten sonra şunlar yapılabilir:
- hata analizi (hangi görevlerde, hangi alt gruplarda (yaş, cinsiyet) hata yoğun) — agrega düzeyde;
- katman eğrilerinin (EXP-017) ve kalibrasyonun yorumlanması;
- beklenmedik bir sonucun nedenini araştırmak (ör. regresyon testi başarısızsa kod incelemesi; bir backbone şans düzeyindeyse girdi kontrolü);
- Faz 3 tasarımı için fikir üretmek (ör. ağırlıklı katman toplamı, tüm kayıt girdi).

Kurallar:
1. Bu analizler raporda **"post hoc / keşifsel"** diye etiketlenir ve önceden kilitlenmiş sonuçlardan ayrı bölümde durur.
2. R1–R12'yi değiştiremezler. Onaylayıcı sonucu, D-035 seçimini ya da Faz 2'nin birincil temsilini değiştiremezler.
3. Bir keşifsel bulgu yeni bir hipotez doğurursa, o hipotez **yeni veride ya da Faz 3'te önceden yazılarak** sınanır; aynı Faz 2 sonuçlarıyla "doğrulanmaz".
4. Olumsuz ve beklenmedik sonuçlar silinmez.

---

## 3. Kısa kayıtlar, dolgu ve süre bilgisi (D-015, D-034 madde 1)

### 3.1 Mevcut kural tam olarak ne?

- **D-015 (KABUL):** "Mono → model SR'sine resample → enerji tabanlı kenar kırpma (tepeye göre) → tepe normalizasyonu → 4.0 s pencere / 2.0 s hop, **kısa kayıtlar sıfırla doldurulur**." [DECISION]
- Kaynağı Boll ve ark.: "Recordings are zero-padded to 4.0 s if they are shorter than the window length." [FROM PAPER: Boll ve ark. 2026, Bölüm 4.3]
- Önbellek raporu (D-030): kırpma sonrası **17 kayıt 4 s'den kısa**, en kısası 2.28 s, 2 s'den kısa yok. Medyan 10.26 s. [FACT: `reports/audio_cache/`]
- 12. turdaki tasarım bu kuraldan sapıp kısa kayıtları **dolgusuz** işlemeyi önermişti. Bu revizyonda öneri değişti (Bölüm 3.6).

### 3.2 Kod yolu: kısa bir kayıt nasıl işleniyor?

`scripts/extract_embeddings.py` kayıtları önbellek sırasıyla **teker teker** işler:

```python
x = cache.audio(r, bb.sr)                                  # kırpılmış, normalize kayıt (T,)
ws = B.window_starts(len(x), win=4*sr, hop=2*sr)            # T > 4 s → [(0,4s), (2s,4s), …, (T-4s, 4s)]; T ≤ 4 s → [(0, T)]
if len(x) < win:                                            # kısa kayıt
    both = short_embeddings(bb, x, win)                     # İKİ politika da hesaplanır
    #   nopad  : bb.embed(x[None, :])                       # (1, T) gerçek uzunluk
    #   zeropad: xp = zeros(1, 4s); xp[0, :T] = x
    #            bb.embed(xp, lengths=[T])                  # (1, 4 s) + gerçek uzunluk bilgisi
    E = both[short_policy]                                  # birincil → rec_win4.npy
    short_alt.append(both[diğer])                           # diğeri → short_alt_win4.npz (EXP-016S)
else:
    W = stack([x[s:s+4s] for s in ws])                      # (pencere, 4 s) — hepsi tam 4 s, dolgu yok
    E = bb.embed(W)                                         # (pencere, n_store, dim)
rec_embedding = E.mean(0)                                   # (n_store, dim)
```

`bb.embed(wav, lengths)` dolgu bölgesinin gerçekten sıfır olduğunu assert eder. Model ailelerine göre dolgunun işlenişi (`scripts/backbones.py`):

| Aile | `zeropad` politikasında | Kaynak |
|---|---|---|
| PANNs | Düz sıfır dolgusu; **maske yok** (resmi `forward` maske almaz). Dolgu çerçeveleri log-Mel'de çok düşük değer (≈ −100 dB) olarak modele girer; global max + mean havuzlamaya katılır | Boll'un yaptığı; [FROM OFFICIAL DOCS: PANNs `models.py`] |
| BEATs | Resmi `padding_mask` (örnek → fbank çerçevesi → yama, `BEATs.forward_padding_mask`); dolgu yamaları dikkatte maskelenir; havuzlama **yalnız gerçek yamalar** üzerinden | Resmi fine-tune sınıflandırıcısı da maskeli ortalama yapar [FROM OFFICIAL DOCS: `BEATs.py`] |
| WavLM Large | Resmi özellik çıkarıcı **yalnız gerçek kısımla** normalize eder ve sıfırla doldurur; `attention_mask` modele verilir (işlemcide `return_attention_mask=True`); havuzlama yalnız gerçek çerçeveler (`_get_feat_extract_output_lengths`). Sonuç: dolgu **etkisiz** (birim testi: 4 s ve 6 s'ye dolgu aynı gömmeyi verir; zeropad ≈ nopad) | HF önerisi [FROM OFFICIAL DOCS] |
| WavLM Base+ | Özellik çıkarıcı normalize etmez (`do_normalize=False`), sıfırla doldurur; HF önerisiyle `attention_mask` modele **verilmez** (`return_attention_mask=False`, `feat_extract_norm=group`); havuzlama yalnız gerçek çerçeveler. Ama özellik kodlayıcının GroupNorm'u bütün girdi (dolgu dahil) üzerinden hesaplanır → dolgu gerçek çerçevelerin temsilini de **değiştirir**. Maskeli havuzlama bunu gidermez | HF önerisi [FROM OFFICIAL DOCS]; etki [INFERENCE] (rastgele küçük modelde ölçüldü) |

`nopad` politikasında üç aile de kaydı gerçek uzunluğunda alır (üçü de değişken uzunluğu destekler).

### 3.3 Değişken uzunluklar batch oluşturulurken nasıl ele alınıyor?

- Faz 2'de batch = **tek bir kaydın pencereleri**. 4 s'den uzun bir kaydın bütün pencereleri tam 4 s'dir (son pencere sona hizalı olduğu için kısmi pencere yok).
- Kısa kayıt **tek satırlık** bir batch'tir (nopad: gerçek uzunluk; zeropad: 4 s + `lengths`).
- Yani **farklı uzunluklar asla aynı batch'e girmez.** Dolgu yalnız `zeropad` politikasında ve yalnız 17 kısa kayıtta oluşur.
- Modeller eval modunda; BatchNorm batch istatistiği kullanmaz. Batch'in kompozisyonunun sonucu değiştirmediği SMK-001 S7'de (başka katılımcıların kayıtlarıyla aynı batch) sınanır.
- Faz 3'te (eğitim) durum farklı: batch'ler birden çok kayıttan oluşur ve PANNs'in BatchNorm'u eğitim modunda batch istatistiği kullanır. Faz 3'ün kısa kayıt politikası Faz 2'ninkiyle aynı olmalı (Bölüm 3.6).

### 3.4 Model süre bilgisini dolaylı biçimde kullanabilir mi?

**Evet, mümkün.** Prob kayıt süresini ya da pencere sayısını doğrudan görmez (pencereler ortalanır). Ama süre gömmeye dolaylı yollarla girebilir: [INFERENCE]

| Yol | Ne zaman | Büyüklük beklentisi |
|---|---|---|
| (a) Sıfır dolgusu temsile girer: PANNs'te havuzlamaya, WavLM Base+'ta GroupNorm istatistiğine | yalnız `zeropad`, yalnız 17 kayıt | En güçlü yol; kısa kayıtların gömmesi sistematik olarak kayar |
| (b) Kısa kayıtta daha kısa bağlam (dikkat / evrişim kenar etkileri) | iki politikada da, 17 kayıt | Küçük; maskeli havuzlama bunu yok etmez |
| (c) Kayıt içi sessizlik oranı / konuşma hızı | bütün kayıtlar | Bu gerçek içeriktir (kenarlar kırpıldı, iç duraklamalar kaldı); MFCC tabanında da aynı biçimde var (zaman ortalaması) |
| (d) Sona hizalı son pencerenin örtüşme ağırlığı | bütün kayıtlar | İhmal edilebilir |

Bunun bir sorun olup olmadığı, **sürenin etiketle ilişkili olup olmadığına** bağlıdır. Bunu ana akışa taşımadan, ayrı META-016 raporunda ölçüyoruz:
- görev ve etiket başına süre dağılımı;
- 4 s'den kısa kayıtların etikete ve göreve göre sayısı;
- yalnız-süre referans çizgisi (D-005'teki kayıt-koşulu tabanı; aynı dış fold'lar);
- önceden belirlenmiş duyarlılık: kısa kaydı olan katılımcılar **değerlendirmeden** çıkarıldığında füzyon AUC'leri.

### 3.5 Karşılaştırılabilirliğe etkisi

| | zeropad (D-015) | nopad |
|---|---|---|
| Boll ile birebirlik | Evet (PANNs) | Hayır (17 kayıtta) |
| Kabul edilmiş D-015 ile | Uyumlu | Sapma |
| Faz 3 ile tutarlılık | Doğal (batch eğitimi için dolgu gerekir) | Faz 3'te ayrı bir çözüm gerekir (tek satırlık batch → PANNs BatchNorm eğitimde sorunlu) |
| Yapay içerik | PANNs'te var (maskesiz) · WavLM Base+'ta var (GroupNorm dolguyu görür; yalnız havuzlama maskeli) · BEATs'te çok küçük (resmi maske; fbank sınır çerçeveleri ve yamanın kısmen dolu satırı) · WavLM Large'da yok | Yok |
| Etkilenen veri | 17 / 2 393 kayıt (%0.7); en çok 17 katılımcı (%5); her biri füzyonda 7 görevden yalnız 1'i | aynı |

İki politikanın etkisinin büyüklüğü önceden bilinemez; bu yüzden ikisi de hesaplanır ve fark EXP-016S'de (tekrar 0) raporlanır. MFCC kolları etkilenmez (dolgu kavramı yok).
- EXP-016S, WavLM Large için tanım gereği ≈ 0'dır (maske dolguyu tamamen dışlar); asıl bilgi PANNs ve WavLM Base+'tan gelir.
- En çok 17 / 342 katılımcı etkilendiği için füzyon AUC farkı seyrelir. EXP-016S bu yüzden ayrıca **etkilenen katılımcılarda** füzyon olasılığının ne kadar değiştiğini (ortalama ve en büyük |Δp|) raporlar.
- EXP-018'in "tüm kayıt − 4 s" farkı, 17 kısa kayıtta dolgu politikası farkını da içerir (tüm kayıt görünümü her zaman dolgusuzdur). Etkisi küçük; yorumda belirtilir.

### 3.6 Karar (D-034 madde 1; KABUL)

- **KARAR (KABUL, Alper, 13. tur):** Birincil politika **zeropad = D-015** (Boll ile aynı); dolgunun işlenişi Bölüm 3.2'deki tablodaki gibi (resmi API ne destekliyorsa). Alper'in gerekçesi: önceki çalışmayla ve Faz 3'teki giriş politikasıyla tutarlılık. **Bu seçimin dolgunun yaratabileceği süre ipuçlarını ortadan kaldırdığı varsayılmaz;** bunlar META-016 ve EXP-016S ile ayrı değerlendirilir. `nopad` önceden belirlenmiş duyarlılık (EXP-016S). Ek duyarlılık: kısa kaydı olan katılımcıların değerlendirmeden çıkarılması (META-016).
- **NEDEN:** Faz 2, Faz 3'ün (ana hedef) girdisini yansıtmalı. Faz 3 batch eğitimi gerektirir; kabul edilmiş D-015 ve Boll bu durumu sıfır dolgusuyla çözer. 12. turdaki "dolgusuz" önerim yalnız dondurulmuş çıkarımın temizliğini düşünüyordu; Faz 2–Faz 3 tutarlılığını ve Boll karşılaştırmasını gözden kaçırıyordu.
- **ALTERNATİFLER:** (a) nopad birincil (dondurulmuş çıkarım için en temiz; Faz 3'te ayrı çözüm gerekir); (b) döngüsel dolgu (yapay periyodiklik); (c) kısa kayıtları atmak (kayıp rastgele olmayabilir; görev başına kohort değişir).
- **RİSK:** PANNs ve WavLM Base+'ta 17 kayıtta dolgu yapay içerik üretir ve süre bilgisi gömmeye girebilir. META-016 süre–etiket ilişkisini, EXP-016S politikanın etkisini gösterir.
- **BİLİMSEL SONUÇ:** Politika sonuçlar görülmeden sabitlenir; etkisi iki bağımsız duyarlılıkla ölçülür.
- **UYGULAMA:** `extract_embeddings.py --short-policy zeropad` (varsayılan); notebook 21'deki `SHORT_POLICY`. Onayında `nopad` seçersen yalnız bu değişken değişir; iki politika da zaten hesaplanıyor.

---

## 4. Katman ortalaması tam olarak nasıl hesaplanıyor? (BEATs, WavLM)

### 4.1 Adım adım tensor boyutları

| Adım | BEATs (4 s pencere) | WavLM Base+ (4 s) | WavLM Large (4 s) |
|---|---|---|---|
| Girdi | (B, 64 000) | (B, 64 000) | (B, 64 000) |
| Ön işleme | fbank (B, 398, 128) → yamalar 24 × 8 = 192 | 7 katmanlı CNN → 199 çerçeve | aynı, 199 çerçeve |
| Kanca çıktısı, katman i (i = 0..L) | (192, B, 768) — **T × B × C** | (B, 199, 768) — **B × T × C** | (B, 199, 1024) |
| Zaman ortalaması (dolgu varsa yalnız geçerli) | (B, 768) | (B, 768) | (B, 1024) |
| Katmanları yığ (`_stack`; bütün katmanlarda C aynı, assert) | (B, 13, 768) | (B, 13, 768) | (B, 25, 1024) |
| Pencereler üzerinden ortalama → kayıt | (13, 768) | (13, 768) | (25, 1024) |
| Dosya `rec_win4.npy` | (2393, 13, 768) | (2393, 13, 768) | (2393, 25, 1024) |
| Prob: birincil katmanlar 1..L | (n, 12, 768) | (n, 12, 768) | (n, 24, 1024) |
| Düzleştir (sklearn için) | (n, 9 216) | (n, 9 216) | (n, 24 576) |
| `LayerAverage` (fit yalnız eğitim fold'u) | (n, 768) | (n, 768) | (n, 1024) |
| `StandardScaler` → `LogisticRegression` | | | |

Katman 0 = ilk transformer bloğunun **girdisi** (konum evrişiminden sonra; BEATs ve WavLM Base+'ta ardından LayerNorm da uygulanmış hâli, WavLM Large'da (stable layer norm) LayerNorm'suz); katman i = i'inci bloğun **çıkışı**. Katman 0 birincil temsile girmez. WavLM Large'da (stable layer norm) son LayerNorm modelin çıkışına uygulanır; bizim katman 24'ümüz ondan önceki değerdir. SMK-001 S8, son kancanın (gerekirse bu LayerNorm uygulanmış hâlinin) resmi çıktıyla aynı olduğunu doğrular. Birim testi P3, ara katman kancalarının resmi ara çıktılarla (BEATs `tgt_layer` yolu, HF `hidden_states`) aynı olduğunu doğrular.

### 4.2 Aynı boyuta getirme ve normalizasyon

- **Aynı boyut:** Bir modelin bütün transformer katmanları aynı gizli boyuttadır (BEATs ve Base+ 768, Large 1024). İzdüşüm gerekmez. `_stack` bunu assert eder; `emb_arm` dosya şeklini `info.json` ile karşılaştırır.
- **Normalizasyon:** Katmanların ölçekleri çok farklı olabilir (özellikle Large'ın LayerNorm'suz artık akışı). Ham ortalamada büyük normlu katmanlar baskın olurdu. Bu yüzden `LayerAverage`:

```python
class LayerAverage(BaseEstimator, TransformerMixin):          # scripts/run_probes.py
    def fit(self, X):                                         # X: (n_train, L·D) — YALNIZ eğitim fold'u
        Z = X.reshape(n, L, D)
        self.mean_ = Z.mean(0)                                # (L, D): katman × kanal ortalaması
        self.std_ = Z.std(0) (0'a yakınsa 1)                  # (L, D)
    def transform(self, X):
        Z = X.reshape(n, L, D)
        return ((Z - self.mean_) / self.std_).mean(1)         # her katman z-skoru → katmanlar ortalanır → (n, D)
```

- Bu dönüşüm sklearn `Pipeline`'ın ilk adımıdır; `GridSearchCV` içinde her iç fold'da ve dış fold'un eğitim kümesinde yeniden fit edilir. Test verisinin istatistiği hiçbir zaman kullanılmaz.
- **Varsayım:** Aynı kanal indeksinin (j) farklı katmanlarda benzer bir anlam taşıdığı (artık akış ortak bir taban paylaşır). Bu yüzden z-skorlarını kanal kanal ortalamak anlamlıdır. SUPERB'deki katman ağırlıklı toplamı da aynı varsayıma dayanır. [INFERENCE]
- **Alternatifler:** kare başına LayerNorm (parametresiz, ama kare düzeyi enerji bilgisini siler); katmanları uç uca eklemek (Large'da 24 576 boyut, ~220 eğitim örneği); öğrenilmiş ağırlıklı toplam (lineer prob olmaktan çıkar; Faz 3'ün doğal parçası).

### 4.3 EXP-017 keşifsel kalır; katman seçimi yapılmaz

- EXP-016'nın kolları yalnız birincil temsili kullanır. `emb_arm`, SSL modellerinde birincil katmanların tam olarak 1..L olduğunu assert eder.
- D-035 seçim fonksiyonu yalnız birincil kol adlarını (`beats`, `wavlm_base_plus`, `wavlm_large`) kabul eder. Katman kolları (`<backbone>@L<l>`) ayrı bir çalıştırmada (EXP-017) üretilir ve EXP-016'ya, D-035'e ya da Faz 3 seçimine girmez.
- EXP-017 yalnız tekrar 0'da çalışır ve raporu "keşifsel; hiçbir seçimde kullanılmaz" başlığını taşır.

---

## 5. İstatistik (EXP-016)

### 5.1 Değerlendirme birimi

- **Birim katılımcıdır.** Her katılımcı için 7 görevin olasılık ortalaması (füzyon) alınır; 101244'te 6 görev.
- Her (tekrar r, dış fold k) için, o fold'un dış test katılımcılarında (68–69 kişi; ≈ 56–57 astım / ≈ 11–12 sağlıklı) füzyon AUC'si hesaplanır.
- Bütün kollar aynı 25 (r, k) hücresinde, **aynı katılımcılarda** değerlendirilir (assert). Bu yüzden karşılaştırmalar eşleştirilmiştir.

### 5.2 Onaylayıcı test

Her backbone b ve her MFCC tabanı m ∈ {mfcc_lr, mfcc_mlp} için:

1. Fold düzeyinde fark: d_rk = AUC_b(r, k) − AUC_m(r, k). **25 değer = 5 tekrar × 5 dış fold** (tekrarlı CV); her değer aynı fold'un aynı test katılımcılarında hesaplanır (eşleştirilmiş).
2. **Nadeau–Bengio düzeltilmiş tekrarlı-CV t-testi** [FROM PAPER: Nadeau & Bengio 2003; tekrarlı k-fold uygulaması: Bouckaert & Frank 2004]:
   t = d̄ / √((1/J + n_test/n_train) · s²_d), J = 25, df = 24, n_test/n_train ≈ 0.25 (split dosyalarından ölçülür).
   **Bu 25 skor bağımsız değildir:**
   - aynı tekrardaki 5 fold'un eğitim kümeleri büyük ölçüde örtüşür (her biri verinin %80'i);
   - 5 tekrar aynı 342 katılımcıyı yeniden böler; her katılımcı her tekrarda bir kez test edilir;
   - aynı modelin iki farklı fold'daki AUC'si bu yüzden pozitif korelelidir.
   Düz (bağımsızlık varsayan) t-testi bu yüzden varyansı küçümser ve fazla iyimser p verir. `n_test/n_train` terimi bu korelasyonu kabaca telafi eden bir düzeltmedir; kesin değil, sezgisel (heuristic) bir düzeltmedir ve kullanılan df = 24 de yaklaşık kalır. Bu sınır her EXP-016 raporunda yazılır (D-011).
3. **Tek yönlü** p (H1: d̄ > 0).
4. **Kesişim-birleşim:** p_b = max(p_b,LR, p_b,MLP). "b MFCC'den iyi" iddiası iki tabanı da geçmeyi gerektirir (D-032); bu birleşim ek düzeltme olmadan α'yı korur. [FROM PAPER: Berger 1982]
5. **Holm** düzeltmesi 6 backbone üzerinde; Holm-düzeltilmiş p'ler **0.025** ile karşılaştırılır (Bölüm 5.3a).
6. **Karar:** "destekleniyor (üst sınır, geçici)" ⇔ Holm-düzeltilmiş p_b < 0.025 **ve** iki tabana karşı da havuzlanmış ΔAUC'nin katılımcı bootstrap %95 CI'ının alt sınırı > 0.
   - Havuzlanmış: her katılımcının 5 tekrardaki OOF olasılıklarının ortalaması; 2 000 katılımcı bootstrap'ı (eşleştirilmiş).
   - %95 iki yönlü CI'ın alt sınırı > 0, tek yönlü %2.5 testine karşılık gelir; t-testiyle tutarlıdır.
7. Tekrar başına **DeLong** eşleştirilmiş testi (D-011) yalnız destekleyici bilgi olarak raporlanır; kararı belirlemez.
8. "Geçici": D-011'in ikinci koşulu (yönün zaman-örtüşen alt kohortta korunması) D-028 ile değerlendirme aşamasına ertelendi.

### 5.3 Neden tek yönlü?

- İddia yönlüdür: "backbone MFCC'den **daha iyi**". Tek yönlü test bu iddianın testidir.
- İki yönlü p'ler Holm'a girseydi, MFCC'den anlamlı biçimde **kötü** bir backbone en küçük p ile Holm sırasının başına geçer, diğer backbone'ların eşiklerini gevşetirdi. Bu, "daha iyi" iddiası için istenmeyen bir etkidir.
- Eşik 0.025, iki yönlü 0.05'in pozitif yarısıyla aynı katılıktadır; tek yönlü test bize ek kolaylık sağlamaz.
- **Bedel:** tek yönlü test "daha kötü" sonucunu kanıtlayamaz. Bu yüzden iki yönlü p'ler ve bütün Δ'lar (negatif olanlar dahil) raporlanır; olumsuz sonuçlar betimsel olarak korunur.

### 5.3a Holm'dan sonra neden `p < 0.025`?

- **Önceden belirlenmiş, muhafazakâr bir karar eşiğidir.** Sonuçlar görülmeden sabitlendi (R6); sonuca göre değiştirilmez.
- **Tek bir test için:** tek yönlü 0.025, iki yönlü 0.05 testinin pozitif kuyruğuyla aynıdır. Yani "daha iyi" kararı için tek bir karşılaştırmada iki yönlü 0.05'ten daha gevşek değildir.
- **Altı testlik ailede (çoklu karşılaştırma) doğru ifade şudur:** Holm, tek yönlü p'lere 0.025 düzeyinde uygulandığında, altı backbone'dan **en az birini yanlışlıkla "MFCC'den iyi" ilan etme** olasılığını (aile bazında hata oranı, FWER) en çok **0.025**'te tutar (Holm, herhangi bir bağımlılık yapısında güçlü kontrol sağlar). [FROM PAPER: Holm 1979]
- Bu, "iki yönlü Holm 0.05 ile **tam olarak** eşdeğerdir" demek değildir:
  - İki yönlü Holm 0.05, **herhangi bir yönde** yanlış "fark var" kararını ≤ 0.05'te tutar; sıralamayı |Δ|'ya göre yapar.
  - İki yönlü sürümde Holm adımlarının eşikleri her adımda tek yönlü sürümle aynıdır (iki yönlü p ≤ 0.05/m' ⇔ tek yönlü p ≤ 0.025/m', pozitif Δ için). Ama iki yönlü sürümde MFCC'den anlamlı biçimde *kötü* backbone'lar sıranın başına geçip pozitif farkları daha gevşek adımlara itebilir. Tek yönlü sürümde kötü backbone'ların p'si büyüktür ve sona gider.
  - Sonuç: "daha iyi" kararları için tek yönlü Holm 0.025, iki yönlü Holm 0.05'ten **hiçbir zaman daha gevşek değildir**; çoğu durumda aynı, bazı durumlarda daha katıdır.
- **Ek muhafazakârlık katmanları:**
  - kesişim-birleşim: iki MFCC tabanını da geçmek gerekir, p = max (iki testin büyüğü);
  - iki bootstrap %95 CI'ının alt sınırı da > 0 olmalı;
  - Nadeau–Bengio düzeltmesi düz t-testinden geniş varyans kullanır.
  Bu yüzden gerçek yanlış-pozitif oranı büyük olasılıkla 0.025'in altındadır; bedeli güçtür (Bölüm 5.5). [INFERENCE]

### 5.4 Keşifsel aileler (önceden listelenmiş; her aile içinde Holm; yorum keşifsel)

- RQ3: 15 backbone çifti (cnn14 − cnn14_16k ve wavlm_large − wavlm_base_plus dahil), iki yönlü. Farklı **hazır temsillerin** karşılaştırması olarak yorumlanır; ön-eğitim yöntemi (gözetimli / öz-gözetimli) hakkında çıkarım yapılmaz (Bölüm 9.6).
- RQ4: görev başına Δ (backbone − mfcc_lr), 42 karşılaştırma.
- RQ5: ünlü (aaa) − kelimelerin ortalaması, kol başına.
- EXP-016S, EXP-017, EXP-018.

### 5.5 "Kaç AUC'lik fark görülebilir?" — varsayımlarıyla, eşik DEĞİL

Bu hesap yalnız beklentiyi ayarlamak içindir. Hiçbir kararda eşik olarak kullanılmaz.

| Varsayım | Değer |
|---|---|
| Havuzlanmış AUC | 0.80 civarı |
| Örneklem | 283 astım / 59 sağlıklı |
| Tek AUC'nin SE'si (Hanley–McNeil) | 0.027 |
| İki ses temsilinin AUC tahminleri arasındaki korelasyon ρ | 0.5 ya da 0.7 (bilinmiyor; varsayım) |
| ΔAUC'nin SE'si = SE · √(2(1 − ρ)) | 0.027 (ρ = 0.5) · 0.021 (ρ = 0.7) |
| Normal yaklaşım, %80 güç, tek yönlü α | 0.025 (Holm'un en gevşek adımı) ile 0.025/6 (en sıkı adımı) arası |

**Sonuç:** saptanabilir fark kabaca **0.06–0.09 AUC** (ρ = 0.7 ve gevşek adımda 0.058; ρ = 0.5 ve sıkı adımda 0.093). [INFERENCE]

Sınırlar:
- Bu, havuzlanmış katılımcı düzeyi bir yaklaşımdır. Asıl test fold düzeyindeki Nadeau–Bengio'dur; onun gücü fold'lar arası Δ'nın SD'sine bağlıdır ve veri görülmeden bilinemez.
- Kesişim-birleşim iki tabanı da geçmeyi gerektirir; güç bu hesaptan **düşüktür**.
- AUC tavana yaklaştıkça SE küçülür.
- 12. turdaki "0.08–0.10" ifadesi kaba bir yuvarlamaydı; yukarıdaki tablo onun yerine geçer.
- Yorum: 0.02–0.05'lik farklar bu örneklemde büyük olasılıkla çözülemez; "anlamlı değil" sonucu "fark yok" demek değildir.

---

## 6. D-035: Faz 3 seçimi fold başına (KABUL, Alper'in koşuluyla)

### 6.1 Kural

- **Sabit adaylar:** CNN10 (RQ2'nin eşi: sıfırdan CNN10 ile aynı mimari) ve CNN14 (Boll).
- **BEATs / WavLM Base+ / WavLM Large arasından seçim, her dış fold (r, k) için ayrı yapılır:**
  - Ölçüt: **yalnız o fold'un eğitim katılımcılarının** iç 5-fold OOF tahminleri → 7 görevin katılımcı ortalaması → AUC.
  - En yükseğe 0.01'den yakın olanlar arasından en az parametreli seçilir.
  - Dış test fold'unun skorları seçime **girmez**.
- **Sonuç:** fold'lar farklı backbone seçebilir. Faz 3, her fold'da o fold'un seçtiğini fine-tune eder. Faz 3'ün sonucu belirli bir backbone'un değil, **"iç doğrulamayla seç, sonra fine-tune et" prosedürünün** dürüst tahminidir.
- Bütçe yetmezse ilk çıkarılan CNN14'tür (D-035 madde 5).

### 6.2 12. turdaki kuraldan farkı (neden önemli)

- 12. turdaki öneri, iç doğrulama skorlarını **25 fold boyunca ortalayıp** tek bir backbone seçiyordu.
- Bu ortalama, fold k için yapılan seçime **diğer fold'ların** iç skorlarını da katıyordu. Tekrarlı CV'de fold k'nın dış test katılımcıları diğer fold'ların eğitim kümelerindedir. Yani fold k'nın test katılımcıları seçimi dolaylı yoldan etkiliyordu. Küçük ama gerçek bir sızıntı.
- Fold başına seçim bunu tamamen kaldırır. Bedeli yorumdadır: tek bir "seçilen backbone" yerine bir prosedür tahmin edilir.

### 6.3 Kodda nasıl güvence altında?

1. **İç tahminler yalnız eğitim verisinden.** `run_arm_repeat` → `fit_one(arm, X[tr], y[tr], X[te], inner)`: iç OOF tahminleri `cross_val_predict(est, X[tr], y[tr], cv=PredefinedSplit(inner_fold))` ile üretilir. Dış test satırları bu çağrıya hiç girmez.
2. **Seçim fonksiyonu dış testi parametre olarak almaz.** `select_per_fold(inner_raw, C, splits, candidates, params)` yalnız iç OOF tablosunu okur.
3. **Assert:** Her (r, k) için iç tablodaki her katılımcının split dosyasında o fold için `role == "train"` olduğu doğrulanır; dış test katılımcısı varsa hata verir.
4. **Çıktı ayrı dosyada:** `experiments/frozen_probe/d035_selection.csv` (tekrar, fold, adayların iç AUC'leri, eşit olanlar, seçilen; katılımcı ID'si yok). Faz 3 bu dosyayı okur.
5. **Testler:**
   - P6: kurgulanmış örnekte seçim ve eşitlik kuralı doğru; iç tabloya dış test katılımcısı konursa assert hata verir.
   - P7: dış test tahmin dosyaları kasıtlı olarak "kusursuz" yapılıp EXP-016 yeniden çalıştırılır → `d035_selection.csv` **birebir aynı** kalır; aynı çalıştırmada bozulmuş dosyaların gerçekten kullanıldığı (o kolun AUC'sinin 1'e çıktığı) da doğrulanır.
6. **Prosedürün dürüst tahmini:** seçimden sonra, her fold'da seçilen kolun o fold'daki dış test füzyon AUC'si yalnız **değerlendirme** için okunur (`procedure_estimate`). Sonradan bakarak en iyi sabit kolla farkı "kazananın laneti" olarak raporlanır.

Kalan küçük iyimserlik: iç OOF tahminleri, C'nin seçildiği aynı iç fold'lardan gelir. Simetrik bir iyimserliktir; boyut ve katman sayısıyla kollar arasında biraz değişebilir. [INFERENCE]

---

## 7. Aynı dış fold'lar, aynı birim, katılımcı ayrımı — koddaki kontroller

| Gereksinim | Kontrol | Yer |
|---|---|---|
| Bütün kollar aynı split dosyalarını kullanır | Tek bir `splits` sözlüğü bütün kollara verilir; dosya sha256'ları ara sonuç imzasında | `run_probes.main`, `run_all` |
| Bütün kollar her görevde aynı katılımcı kümesini kapsar | assert (görev başına küme eşitliği) | `run_probes.main` |
| Her (r, k, görev) ve füzyon için kollar aynı dış test katılımcılarında | assert | `check_same_units` |
| Bir katılımcının kayıtları eğitim ve dış teste dağılmaz | **Asıl güvence yapısaldır:** roller split dosyasından katılımcı düzeyinde gelir ve her görevin kayıtları katılımcının rolüne göre atanır. Fold düzeyinde assert: eğitim ∩ test = ∅ ve kohorttaki herkes bir rolde; görev düzeyinde: her kaydın katılımcısı bir rolde (`(tr | te).all()`; split'te olmayan katılımcıyı yakalar). Görev düzeyindeki diğer assert'ler yapı gereği her zaman doğrudur; yalnız akıl sağlığı kontrolüdür | `run_arm_repeat` |
| Pencereler kayıttan ayrılmaz | Pencereler kayıt gömmesine ortalanır; prob kayıt düzeyinde eğitilir | `extract_embeddings` |
| İç fold'lar da katılımcı düzeyinde | `inner_fold` split dosyasından katılımcıya bağlı | `run_arm_repeat` |
| Test: dış test ve iç (eğitim) katılımcıları her (kol, r, k) için ayrık | Ara dosyalardan doğrulanır | `tests/test_phase2.py` P7 |

---

## 8. Meta veri / confounder karşılaştırmaları ayrı raporda (META-016)

- `scripts/report_metadata_monitor.py` → `reports/frozen/META-016_metadata_monitor.{md,json}`. EXP-016'nın kaydedilmiş dış test tahminlerini okur; yeni model eğitmez (yalnız yalnız-süre referans LR'si).
- İçerik:
  1. EXP-014'teki üç bağlam bağımlılığı AUC'si (hastalarda geç vs erken, hastalarda sabah vs öğleden sonra, sağlıklılarda sabah vs öğleden sonra), her kol ve referanslar için;
  2. süre dağılımı (görev × etiket), kısa kayıt sayıları;
  3. yalnız-süre referans çizgisi (aynı dış fold'lar);
  4. kısa kaydı olan katılımcılar dışlandığında füzyon AUC'leri.
- Bu raporun hiçbir sayısı onaylayıcı karara, D-035 seçimine ya da katman seçimine girmez. Kayıt bağlamının asıl değerlendirmesi model geliştirme bitince yapılır (D-028).
- **Ana tabloda kalanlar:** D-028 (KABUL) her sonuçla birlikte iki referans satırını (yalnız bağlam, yalnız yaş; aynı fold'larda) istiyor. Bu satırlar EXP-016 tablosunda "referans çizgisi (D-028; test edilmez)" etiketiyle duruyor; onlarla hiçbir test yapılmıyor. Bunları da ayrı rapora taşımak D-028'i değiştirmek olur; istersen ayrı bir kararla yapılabilir.

---

## 9. Diğer tasarım ayrıntıları

### 9.1 Verinin yapısı

| Öğe | Değer | Kaynak |
|---|---|---|
| Kohort | 342 katılımcı (283 astım / 59 sağlıklı); görev 4 (ordu) için 341 | D-002, AUD-001 |
| Kayıt | 2 393 (101244'ün görev 4 kaydı yok) | önbellek raporu |
| Girdi | Harmonize önbellek `audio_cache_v1` (D-030) | `reports/audio_cache/` |
| Süre (kırpma sonrası) | min 2.28 · p5 8.92 · medyan 10.26 · max 13.09 s; 17 kayıt < 4 s | önbellek raporu |
| Pencere | 10.26 s → 5 pencere; toplam ≈ 12 000 [INFERENCE] | Bölüm 3.2 |
| Split | `outer_r0–4.csv` (D-029): 5 tekrar × 5 dış fold; her dış eğitim kümesinde 5 iç fold | `reports/splits/` |
| Bağlam ve yaş | `participant_context.csv` | D-024 |
| MFCC | `features_cache.csv` (EXP-011 girdisi) | D-031 |

**Toplama hiyerarşisi:** pencere → (zaman ortalaması) → pencere gömmesi → (ortalama) → kayıt gömmesi → LR → kayıt olasılığı → 7 görevin ortalaması → katılımcı füzyonu. LR'nin logit'i doğrusal olduğu için (StandardScaler ve katman ortalaması da doğrusal) "ortalama gömmeye LR" = "pencere logit'lerinin ortalaması" (PHASE0_REPORT Bölüm 7). LR kayıt düzeyinde eğitilir; aynı kaydın pencereleri bağımsız örnek sayılmaz.

### 9.2 Varsayımlar

| # | Varsayım | Etiket | Kontrol |
|---|---|---|---|
| V1 | Resmi kod + ağırlık + SR + frontend = ön-eğitimdeki girdi dağılımı | [FROM OFFICIAL DOCS] | SMK-001 S2, S3, S9 |
| V2 | Eval modunda katılımcılar arası bilgi akışı yok | [INFERENCE] | SMK-001 S7 |
| V3 | Gömmeler etiketten habersiz; split'ten önce çıkarmak sızıntı yaratmaz | [INFERENCE] | Çıkarım kodu etiket okumaz |
| V4 | Zaman ortalaması astımla ilgili bilgiyi büyük ölçüde korur | [HYPOTHESIS] | Test edilmiyor |
| V5 | 11 kHz alçak geçiren, 32 kHz PANNs'in üst ~4 mel bandını boşaltır; model tolere eder | [INFERENCE] | CNN14 vs CNN14_16k dolaylı bilgi |
| V6 | WavLM'in İngilizce ön-eğitimi Türkçe kelimelerde kullanılabilir temsil üretir | [HYPOTHESIS] | Test edilmiyor |
| V7 | 7 görevin olasılık ortalaması makul bir katılımcı skorudur | [DECISION, D-032] | EXP-011 |
| V8 | Katman ortalamasında kanal indeksleri katmanlar arasında hizalıdır | [INFERENCE] | Bölüm 4.2 |

### 9.3 Sızıntı tablosu

| Risk | Önlem |
|---|---|
| Katılımcı sızıntısı | Bölüm 7 |
| BatchNorm'un batch istatistiği | eval assert'i; S7 |
| Model içi augmentation | yalnız eğitim modunda; eval'de kapalı (D-016) |
| Ölçekleme / katman standardizasyonu | Pipeline içinde, yalnız eğitimde fit |
| C ve eşik | yalnız iç döngü |
| Katman seçimi | yok (Bölüm 4.3) |
| Faz 3 seçimi | fold başına, yalnız iç doğrulama (Bölüm 6) |
| Dolgu yoluyla süre | Bölüm 3; META-016; EXP-016S |

### 9.4 Prob ve MFCC kolları

- LR: `[LayerAverage] → StandardScaler → LogisticRegression(class_weight="balanced", max_iter=5000)`; C ∈ {10⁻⁵ … 10²} (8 değer, bütün temsillerde aynı); iç 5-fold `neg_log_loss`; eşik iç OOF'ta dengeli doğruluğu en yükselten değer; kalibrasyon (Brier, ortalama tahmin − oran, eğim). C'nin ızgara ucuna düştüğü fold oranı raporlanır.
- Izgaranın alt ucu 10⁻⁵: p ≫ n (2048 boyut, ~220 örnek) durumunda en iyi C 0.001'in altında olabilir. [INFERENCE]
- mfcc_lr: aynı LR. mfcc_mlp: D-032 (StandardScaler → SMOTE → MLP). Doğrusal olmayan bir MFCC tabanına karşı doğrusal gömme probu → karşılaştırma gömmeler aleyhine, muhafazakâr.

### 9.5 SMK-001 kontrolleri (gerçek checkpoint'ler)

S1 sha256 (Zenodo md5'i indirmede) · S2 sözleşme · S3 strict yükleme · S4 parametre sayısı · S5 şekil / dtype / cihaz / sonluluk · S6 eval determinizmi · S7 batch-değişmezlik (başka katılımcılarla) · S8 son kanca = resmi çıktı · S9 PANNs "Speech" ilk 3'te · S10 aynı-kişi benzerliği · S11 kaydet-yükle · S12 Faz 3 eğitim adımı (bellek, süre) · S13 hız · **S14 dolgu yolu** (tam uzunlukta `lengths` sonucu değiştirmez = PASS ölçütü; kısa parça zeropad vs nopad benzerliği INFO).

### 9.6 Ön-eğitim kaynakları ve etiketli veri / öğretmen model kullanımı

| Model | Ön-eğitim verisi | Hedef / yöntem | Etiket ya da öğretmen | Kaynak |
|---|---|---|---|---|
| PANNs CNN10 / CNN14 (32 kHz), CNN14_16k (16 kHz) | AudioSet (YouTube klipleri, ~1.9 M klip, ~5 000 saat; 32 kHz'e örneklenmiş; 16k varyantı 16 kHz) | 527 AudioSet sınıfı üzerinde çok etiketli **gözetimli** sınıflandırma (eğitimde mixup + SpecAugment) | **Doğrudan AudioSet etiketleri** | [FROM PAPER: Kong ve ark. 2020]; Zenodo 3987831 |
| BEATs iter3+ (AS2M), fine-tune edilmemiş | AudioSet tam eğitim kümesi (AS-2M), 16 kHz | Akustik tokenizer'ın ürettiği ayrık etiketleri maskeli yamalardan tahmin etme | **Dolaylı etiket:** iter3+ tokenizer'ı, **AudioSet etiketleriyle fine-tune edilmiş** bir BEATs modelinden damıtılmıştır. Makale: "the self-distilled tokenizer for BEATs_iter3+ pre-training takes the supervised fine-tuned BEATs_iter3 as the teacher model"; tablo dipnotu: "We use AS-2M supervised data during pre-training." (Makalenin metni ile tablosu öğretmenin iter2 mi iter3 mü olduğu konusunda tutarsız; etiket kullanımı her ikisinde de açık.) | [FROM PAPER: Chen ve ark. 2022, BEATs, arXiv 2212.09058, Bölüm 4.2, Tablo 2] |
| WavLM Base+ | 94 bin saat İngilizce konuşma: ~60 bin s Libri-Light, ~10 bin s GigaSpeech, ~24 bin s VoxPopuli | Maskeli konuşma tahmini + gürültü giderme; hedefler k-means sözde etiketleri (HuBERT Base 2. iterasyon, 9. katman) | **Etiket / transkript yok** (öz-gözetimli); ancak sözde etiketler başka bir öz-gözetimli modelden (HuBERT) gelir | [FROM PAPER: Chen ve ark. 2022, WavLM, arXiv 2110.13900] |
| WavLM Large | aynı 94 bin saat | aynı | aynı | aynı |
| MFCC | — | elle tasarlanmış özellik | yok | D-031 |

**Yorum sınırı (RQ3 için bağlayıcı):**
- Modeller aynı anda **çok şeyde** farklıdır: ön-eğitim hedefi (gözetimli / maskeli tahmin / gürültü giderme), etiket kullanımı (doğrudan / dolaylı / yok), veri alanı (genel ses / İngilizce konuşma), veri miktarı, mimari (CNN / transformer), parametre sayısı, girdi SR'si ve bant genişliği (32k yolu 11 kHz'e kesildi; diğerleri 8 kHz), frontend (log-Mel / fbank / ham dalga).
- Bu yüzden Faz 2 bir **ön-eğitim yöntemi karşılaştırması değildir**. "Gözetimli ön-eğitim öz-gözetimliden iyi / kötü" gibi bir çıkarım yapılamaz. Örneğin BEATs iter3+ bile AudioSet etiketlerini dolaylı kullandığı için "saf öz-gözetimli" grupta sayılamaz.
- Doğru ifade: "**Bu veri setinde, bu hazır checkpoint'lerin dondurulmuş temsilleri** (bu protokolle: 4 s pencere, katman ortalaması, lineer prob) şu sonuçları verdi."
- Kısmen kontrollü tek karşılaştırmalar: CNN14 vs CNN14_16k (aynı aile, farklı SR / bant; ama ayrı eğitilmiş checkpoint'ler) ve WavLM Base+ vs Large (aynı veri ve hedef, farklı ölçek). Bunlar da keşifseldir.

### 9.7 Hesap ayrıntıları

- fp32, TF32 kapalı, deterministik cuDNN.
- Drive `data_derived/embeddings_v1/<backbone>/`: `rec_win4.npy`, `rec_full.npy`, `short_alt_win4.npz`, `recordings.csv`, `info.json` (kısa kayıt politikası dahil), `MANIFEST.sha256`.
- Kaldığı yerden devam: 100 kayıtlık parçalar, imzalı (backbone, checkpoint sha256, önbellek sha256, pencere, politika); imza uyuşmazsa yeniden üretilir.
- Süre: GPU çıkarımı < 1 saat; problar CPU'da saatler. [INFERENCE — SMK-001 S13 ölçer]

---

## 10. Önceden yazılmış beklentiler [HYPOTHESIS]

Sonuçlar görülmeden yazıldı; yanlış çıkmaları da raporlanır.

| # | Beklenti | Gerekçe |
|---|---|---|
| B1 | Backbone füzyon AUC'leri 0.70–0.85; onaylayıcı testi geçen olmaması daha olası | Bölüm 5.5; MFCC füzyonu zaten 0.77 |
| B2 | Bağlam bağımlılığı göstergeleri (META-016) MFCC'ninkinden yüksek | Daha zengin temsil oda / cihaz / saat bilgisini de taşır |
| B3 | \|cnn14 − cnn14_16k\| < 0.03 | 32k yolu 11 kHz'e kesildi |
| B4 | WavLM'de katman eğrisi ortada tepe yapar | Pasad ve ark. 2021; Chen ve ark. 2022 |
| B5 | Görevler arasında güvenilir sıralama yok | EXP-011/012 |
| B6 | \|tüm kayıt − 4 s\| < 0.02 | Kayıtlar ~10 s |
| B7 | WavLM Large, Base+'tan anlamlı biçimde iyi değil | 342 katılımcı |
| B8 | \|zeropad − nopad\| (EXP-016S) füzyon AUC'sinde < 0.01; WavLM Large'da ≈ 0 (tanım gereği) | Etkilenen kayıt %0.7; asıl bilgi etkilenen katılımcılardaki \|Δp\| |

---

## 11. Senin yapman gerekenler

1. ~~D-034 onayı~~ → KABUL (zeropad birincil).
2. **BEATs ağırlığı:** README tablosundaki "BEATs_iter3+ (AS2M)" (fine-tune edilmemiş, üçüncü sütun) → Drive `MyDrive/asthma-voice/models/BEATs_iter3_plus_AS2M.pt`.
3. PR'ı birleştir → `20_smoke_tests.ipynb` (T4) → SMK-001 raporunu birlikte okuyalım.
4. Hepsi PASS ise `21_frozen_embeddings.ipynb`.
