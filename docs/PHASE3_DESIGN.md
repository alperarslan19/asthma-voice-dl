# Faz 3 — Uçtan uca fine-tuning: tasarım (EXP-030)

**Durum:** **KABUL (D-042)**, 2026-10-11, 20. tur (önerildi: 18. tur; G1 ve SMK-002 sonuçları okunduktan sonra kabul edildi, tarif değişmedi).
**Değişiklik (D-043, KABUL, 22. tur; post hoc, dış fold eğitiminden önce; DIAG-001 = H2 → değerlendirme fp32):** kalibrasyonda (CAL-001) kayıp kuralı 4 / 5 kolda 1. epoch'u seçti. Epoch kuralı iç doğrulama AUC'si + 1-SE olur; ısınma sabit epoch sayısı olur; CNN14 önce DIAG-001 ile teşhis edilir. Aşağıdaki "Epoch sayısı" satırı v1'dir; geçerli kural için DECISIONS D-043. EVAL-016 Aşama 1 raporundan **sonra** yazıldı, yani "T1 sonrası" etiketli (D-037 madde 3).

- Sonuçlardan önce sabitlenmiş ve burada yalnız uygulanan kurallar:
  - backbone seçimi (D-035);
  - C3 ölçütü (CONFOUND 6);
  - D-027 Kol B'nin koşulu;
  - gün-gruplu değerlendirme (D-041).
- Bu belgede ilk kez önerilen seçimler **†** ile işaretli.

**Hiçbir fine-tuning hesaplaması yapılmadı.** Kod ve sentetik testler hazır. Kalibrasyon ve eğitim, D-042 kabul edilene kadar kod kapısıyla kapalı.

---

## 1. Soru ve bu fazın ne gösterebileceği

- **Soru:** Önceden eğitilmiş ses modelleri astım / sağlıklı görevinde uçtan uca eğitilirse, katılımcı skoru dondurulmuş probun skoruna göre iki şeyi değiştiriyor mu?
  - (a) daha iyi ayırıyor mu (E1);
  - (b) ölçülen bağlamın ötesinde daha fazla bilgi taşıyor mu (C3)?
- **Faz 3 protokolde keşifsel** (CONFOUND 5.11): kilitli test seti yok ve backbone seçimi iç doğrulamayla yapılıyor. Sonuçlar "seçim sonrası, keşifsel" etiketiyle raporlanır.
- **Önceden yazılmış beklenti** [HYPOTHESIS; CONFOUND 6]: fine-tuning E1'i T1'den daha çok artıracak ve bağlam kullanımını artırabilir, çünkü kodlayıcı etiketi ayıran her şeye uyum sağlar. Böyle çıkarsa bu bir **olumsuz ama değerli sonuçtur**.

## 2. Veri ve split

- **KARAR:** Gün-gruplu split'ler (`splits_day/`), 3 tekrar × 5 dış fold, her dış train içinde 5 iç fold. Dış ve iç fold'lar kayıt gününe göre gruplu, etikete göre tabakalı (`make_splits.py --group-by-day`).
- **NEDEN:**
  - 65 kayıt gününün 53'ü tek etiketli. Katılımcı-gruplu CV'de test katılımcısının gün eşleri eğitimde kalır; model günün akustik izini öğrenip etiketi oradan tahmin edebilir.
  - Sentetik testte (F11) yalnız gün izi taşıyan, astım bilgisi hiç olmayan bir gömme, katılımcı-gruplu CV'de T1'i p = 3.6e-9 ile "anlamlı" yaptı; gün-gruplu CV'de p = 0.26. [FACT, sentetik]
  - Fine-tuning bu yola dondurulmuş probdan daha açık. [INFERENCE]
  - 3 tekrar: D-027.
- **ALTERNATİFLER:**
  - Katılımcı-gruplu split: Faz 2 ile doğrudan karşılaştırılabilir, ama gün izi açık kalır.
  - İki split türünde de fine-tune: GPU maliyeti iki katı.
- **RİSK:**
  - Eğitim günleri daha az çeşitli; E1 Faz 2'den düşük çıkabilir. Bu, modelin kötüleştiği anlamına gelmez; dondurulmuş baz çizgisi aynı split'lerde koşulduğu için karşılaştırma adil kalır.
  - Test fold'larında sağlıklı sayısı daha değişken (gerçek bağlam yapısında 9–14; katılımcı-gruplu split'te ~12).
- **UYGULAMA:** `notebooks/24_phase3_prep.ipynb`.

## 3. Kollar

| kol | ne | kaynak |
|---|---|---|
| `cnn10` | PANNs CNN10, AudioSet ön-eğitimli, uçtan uca | D-035 madde 1 (sabit) |
| `cnn14` | PANNs CNN14 32 kHz, uçtan uca | D-035 madde 1 (sabit); bütçe yetmezse ilk çıkarılan |
| `ssl` | her (tekrar, fold) için BEATs / WavLM Base+ / WavLM Large arasından, **yalnız o fold'un eğitim katılımcılarının iç OOF dondurulmuş prob AUC'siyle** seçilen backbone (gün-gruplu split'lerde, EXP-016DG) | D-035 madde 2–3 |
| `cnn10_scratch` † | aynı CNN10 mimarisi, **rastgele başlangıç**, aynı kod, split, tohum ve epoch kuralı | D-036'nın önerdiği kontrol kolu (RQ2: ön-eğitim katkısı; Boll'da da var). Ucuz (CNN10 ~5 M parametre) |

- **D-027 Kol B** (bağlam-dengeli ağırlık) **planlanmıyor.** Önceden yazılmış koşulu (G1+ ya da L3 kısmi test p < 0.05) EVAL-016'da sağlanmadı: G1 hepsi negatif, L3 p ≥ 0.08.

## 4. Eğitim tarifi

| öğe | karar | gerekçe |
|---|---|---|
| Ne eğitilir | **Uçtan uca** †: bütün katmanlar. Tek istisna WavLM'in konvolüsyonel özellik kodlayıcısı (donuk) | Alper'in hedefi ve Boll (bütün katmanlar). D-027'nin "kısmi fine-tune" önerisinin yerine geçer. WavLM / wav2vec2 fine-tuning'de özellik kodlayıcısını dondurmak yaygın uygulama [INFERENCE: HF örnekleri] |
| Girdi | Harmonize ham dalga, modelin SR'ında; modelin kendi frontend'i (D-007) | Faz 2 ile aynı |
| Eğitim örneği | Her epoch'ta her kayıttan bir rastgele 4 s parça (tohumlu, kollar arasında aynı). Kısa kayıt: sıfır dolgu + gerçek uzunluk (BEATs / WavLM maskesi; PANNs'te maske yok, D-034) | Faz 2'nin pencere uzunluğu; 7 görevin hepsi tek modelde (D-027) |
| Temsil | PANNs: fc1 gömmesi. SSL: 1..L katman çıktılarının zaman ortalamalarının öğrenilebilir ağırlıklı toplamı; başlangıç = eşit ağırlık † | Başlangıçta temsil Faz 2'nin birincil temsiliyle aynı, yani fine-tuning dondurulmuş baz çizgisinden başlar |
| Baş | LayerNorm → Linear(d, 256) → ReLU → Dropout 0.3 → Linear(256, 1) | Boll'un MLP başı (dropout 0.3). Gizli boyut 256 bizim seçimimiz † (Boll belirtmiyor) |
| Kayıp | Sınıf-dengeli BCE: pos_weight = n_sağlıklı / n_astım (kayıt düzeyi, yalnız eğitim kümesinden) | D-027; D-033 (SMOTE yok) |
| Optimizasyon | Adam, weight decay 1e-4, batch 16 | Boll |
| Öğrenme hızı | PANNs (ön-eğitimli ve sıfırdan): 1e-4. SSL gövdesi: **2e-5** + ilk %10 adımda doğrusal ısınma †. Baş ve katman ağırlıkları: 1e-3 † | PANNs: Boll. 300 M parametreli bir transformer'ı 1 900 kayıtla 1e-4'te tümden eğitmek ön-eğitimli temsili bozma riski taşır; SSL fine-tuning'de 1e-5–5e-5 yaygın [INFERENCE]. Ayar araması yok |
| Kararlılık | fp16 autocast + GradScaler (frontend'ler fp32, SMK-001); gradyan normu ≤ 1.0 † | SMK-001'de fp16 taşması görüldü; atlanan adımlar loglanır |
| Augmentation | Kapalı: SpecAugment, maskeleme, layerdrop (D-016). Dropout varsayılan | D-016; layerdrop kapalı → her adımda bütün katmanlar (temsil tanımı sabit) |
| Bellek | WavLM Large: 8'lik iki mikro-batch (birikimli 16) †; diğerleri 16 | SMK-001 S12: Large 13 GB (T4 15 GB) |
| **Epoch sayısı** | Sabit. Kol başına bir kez belirlenir: tekrar 0 / fold 0'ın iç train'inde eğit, iç doğrulama kaybının 3 epoch'luk hareketli ortalamasının en küçük olduğu epoch (üst sınır PANNs 40, SSL 20). Dış fold'larda erken durdurma ve checkpoint seçimi **yok**; son epoch'un modeli test edilir | D-027: bağlamla karışmış bir doğrulama AUC'siyle seçim kestirmeyi ödüllendirir. İç doğrulama AUC'si yalnız izlemek için loglanır. Bilinen küçük risk: tek bir skaler hiperparametre, başka fold'ların test katılımcılarını içeren veriyle seçiliyor |
| Tohum | Kol, tekrar ve fold'a göre sabit; aynı fold'da bütün kollar aynı kayıt sırasını ve kesim noktalarını görür | Eşleştirme |

## 5. Değerlendirme (sonuçlardan önce sabit)

### 5.1 G1 — dondurulmuş problar gün-gruplu split'lerde (D-041)
- EXP-016'nın aynı probu `splits_day` üzerinde (EXP-016DG).
- Katılımcı ŝ_DG ile EVAL-016'nın aynı T1'i: Firth PLR, CV3, bootstrap OR.
- **Okuma kuralı:** iddiadaki backbone'lar (cnn10, beats, wavlm_large; sınırdaki cnn14) için ŝ_DG ile T1-PLR p < 0.05 ve OR > 1 ise "gün parmak izi yoluyla açıklanmıyor". Değilse o backbone için iddia "gün / seans artefaktı dışlanamıyor"a düşer ve `docs/EVAL016_CONCLUSION.md` buna göre güncellenir. Holm ve CV3 betimsel.
- Bu tablo aynı zamanda C3'ün dondurulmuş baz çizgisi ve D-035 seçiminin kaynağıdır.

### 5.2 Fine-tune (eğitim bittikten sonra; `evaluate_phase3.py finetune`, eğitim bitmeden yazılacak)
- **Skor:**
  - kayıt logit'i = kaydın 4 s / 2 s pencere logit'lerinin ortalaması;
  - katılımcı logit'i = kayıt logit'lerinin ortalaması;
  - ŝ_FT = 3 tekrarın ortalaması.
- **E1:** fold AUC ort ± SD (15 fold) ve tekrar başına havuzlanmış AUC [%95 bootstrap]. Dengeli doğruluk yalnız betimsel (eşik 0 logit; sınıf-dengeli kayıpla eğitildi).
- **T1:** ŝ_FT ile EVAL-016'nın aynı T1'i (Firth PLR, CV3, bootstrap OR).
- **C3 (CONFOUND 6; keşifsel):** aynı katılımcılarda T1 sapma azalması farkı, LR(ŝ_FT) − LR(ŝ_dondurulmuş).
  - Dondurulmuş eş, aynı fold'da aynı backbone: `ssl` kolunda her fold'un seçtiği backbone'un EXP-016DG tahmini.
  - Katılımcı bootstrap'ı (B 2000; her örneklemde iki T1 de yeniden fit edilir).
  - **C3 pozitif:** CI 0'ı dışlıyor ve fark > 0.
- **C2 (bağlam kullanımı):** ŝ_FT için etiket-içi permütasyon kısmi testi (EVAL-016 S10) ve L5 hasta-içi PM katsayısı. Aynı lensler ŝ_dondurulmuş için de raporlanır.
- **RQ2 (ön-eğitim):** cnn10 − cnn10_scratch, fold düzeyinde eşleştirilmiş ΔAUC (Nadeau–Bengio) ve iki kolun T1'i.
- **Sunum:** E1 artışı tek başına başarı sayılmaz (C3). Bütün kollar ve olumsuz sonuçlar raporlanır.
- **Eklenen betimsel okumalar (post hoc, 2026-10-12; karar kuralı yok):**
  - **O1 aşırı uyum / kalibrasyon:** test kalibrasyon eğimi (y ~ ŝ; < 1 aşırı güven), ortalama |ŝ|, son eğitim kaybı. Kullanıcının aşırı uyum sorusu üzerine.
  - **ΔE1 FT − dondurulmuş:** fold-eşli, Nadeau–Bengio. E1'i dondurulmuş eşle yan yana okumak için; C3'ün yerine geçmez.

## 6. Kayıt, kurtarma, kapılar

- **Checkpoint:** `latest.pt` atomik yazılır: CNN10'da her epoch, SSL'de 2, CNN14'te 5 epoch'ta bir (Drive'a yazma süresi ve alanı; kesintide en çok bu kadar epoch kaybolur). İçinde eğitilen parametreler, BatchNorm tamponları, optimizer, scaler, scheduler, RNG durumları ve epoch var.
- **Log ve yapılandırma:** `train_log.csv` (epoch, kayıp, lr, adım, atlanan adım, süre, GPU GB) ve `config.json` (kol, tohum, imzalar: split, bağlam, önbellek, checkpoint sha256, kod).
- **Fold bitince:** `preds.csv` ve `fold.json` yazılır, `latest.pt` silinir (Drive alanı).
- **Checkpoint yeri (D-042 kabul notu; mühendislik):** Colab'da `latest.pt` yalnız yerel diskte (`--ckpt-root /content/ft_ckpt`); günlük, yapılandırma, `preds.csv` ve `fold.json` Drive'da. Drive'da silinen / üzerine yazılan dosyalar çöpe gidip kotadan sayıldığı için. Makine değişirse yarım kalan fold aynı tohumla baştan başlar (en çok bir fold kaybı).
  - Yeniden çalıştırmada bitmiş fold atlanır; yarım kalan fold kaldığı yerden sürer. CPU'da sürdürülen koşu kesintisiz koşuyla bire bir aynı (F6). GPU'da cuDNN nedeniyle çok küçük farklar olabilir [INFERENCE].
  - Bir fold farklı bir yapılandırmayla yarım kalmışsa (imza ya da epoch farklı) script durur.
- **"En iyi doğrulama checkpoint'i" yok.** Tasarım gereği doğrulamaya göre seçim yapılmıyor; kalibrasyonun bütün eğrisi loglanıyor.
- **Kapılar:**
  - Kalibrasyon ve eğitim, `finetune.PHASE3["status"] == "KABUL"` olmadan çalışmaz.
  - Notebook 25, SMK-002 PASS ve G1 raporu main'de değilse durur.
- **SMK-002 (GPU, zorunlu):** her kol için şunlar sınanır:
  - split sızıntı kapıları;
  - ileri geçiş şekilleri;
  - gerçek eğitim döngüsüyle 2 küçük epoch (kayıp sonlu, parametreler değişiyor);
  - değerlendirme;
  - checkpoint'ten kurulan modelin aynı tahmini vermesi;
  - tepe GPU belleği < 14.5 GB;
  - adım süresinden bütçe tahmini.

## 7. Bütçe [INFERENCE: SMK-001 S12 / S13; SMK-002 ölçecek]

Varsayımlar: eğitim kümesi ~1 900 kayıt, ~120 adım / epoch; test ~480 kayıt × ~4 pencere.

| kol | s / adım (S12) | epoch | en çok epoch | fold (dk, üst sınır) | 15 fold |
|---|---|---|---|---|---|
| cnn10 | 0.06 | ~8 s | 40 | ~6 | ~1.5 saat |
| cnn10_scratch | 0.06 | ~8 s | 40 | ~6 | ~1.5 saat |
| cnn14 | 0.13 | ~15 s | 40 | ~11 | ~2.7 saat |
| BEATs | 0.32 | ~38 s | 20 | ~13 | ~3.3 saat (EXP-016'da 25 fold'un 24'ünde seçilmişti) |
| WavLM Large | 1.0 | ~2 dk | 20 | ~42 | seçilirse fold başına |

- Kalibrasyon: kol başına ~bir fold.
- Toplam üst sınır **~10 T4 saati**. Gerçek epoch sayıları kalibrasyonda belirlenir; genellikle daha az. Colab oturumları kesilebilir; kaldığı yerden devam ettirilir.

## 8. Sınırlar
- Kilitli test yok: sonuçlar keşifsel.
- Fine-tuning hiperparametreleri literatürden sabit; ayar araması yapılmadı. Kötü sonuç "fine-tuning işe yaramaz" değil, "bu tarifle bu veride" demektir.
- G1 dondurulmuş aile için bağlam iddiasını günceller. Fine-tune için ayrı bir N3 / bağlam lensi yok (D-041).
- Prosedür yolu (S13) ve dış geçerlilik açık.

## 9. Çalıştırma sırası
1. **Notebook 24 (CPU):** gün-gruplu split → EXP-016DG problar (D-035 seçimi dahil) → G1 raporu → GitHub.
2. **Notebook 25, Bölüm 1 (T4):** SMK-002 → GitHub. FAIL varsa durulur.
3. D-042 kabulü (SMK-002'nin ölçtüğü bellek ve süreyle).
4. **Notebook 25, Bölüm 2 (T4; birden çok oturum):** kalibrasyon (kol başına bir kez) → eğitim (cnn10 → cnn10_scratch → ssl → cnn14). Biten fold atlanır.
5. `evaluate_phase3.py finetune` → rapor → sonuç bildirimi.
