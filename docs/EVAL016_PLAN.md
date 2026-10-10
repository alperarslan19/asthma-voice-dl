# EVAL-016 — Dondurulmuş aile için bağlam değerlendirmesi: analiz planı

**Durum:** D-025 (içerik değişmeden) ve D-037 **KABUL** (2026-10-10, 15. tur, Alper). **Uygulama protokolü v1.2 KABUL (D-038, 16. tur):** S9–S12 kabul; K2 kabul, gün-kümeli test **jackknife CV3** ile uygulanıyor, çünkü planın ilk metnindeki sandviç (CR1) bu veri yapısında geçerli bulunmadı (Bölüm 10, SIM-002). Analiz ayarları sabitlendi (D-038 madde 4). Kod (`scripts/evaluate_context.py`), sentetik testler (`tests/test_eval016.py`, E1–E16) ve `notebooks/22_context_eval_frozen.ipynb` hazır.
**Gerçek veride ÇALIŞTIRILMADI: hiçbir T1, C1, C2 sonucu yok.** Script, onaysız protokolle ya da sabit ayarlardan farklı bir ayarla gerçek veride çalışmayı reddeder.
**Yazıldığı an:** EXP-016'nın E1 sonuçları (AUC tabloları) görüldükten sonra, ama **bu plandaki hiçbir test gerçek veride çalıştırılmadan önce**. T1, E2h-KLR, Spisak, G1, N5 ve L5 sonuçlarının hiçbiri hesaplanmadı.

---

## 0. Bu analiz neden şimdi ve bu bir protokol değişikliği mi?

**Soru (D-023, projenin birincil sorusu):** Ses, ölçülen kayıt bağlamı (saat, tarih) ve yaşın ötesinde astım bilgisi taşıyor mu?

**Neden şimdi:**
- Faz 2'de en iyi ses temsili (BEATs 0.921 ± 0.046), yalnız-bağlam referansının (0.934 ± 0.022) altında kaldı. E1 AUC'si bu soruya cevap veremiyor. [FACT]
- Bu analiz yalnız Drive'daki kaydedilmiş tahminleri ve gömmeleri kullanıyor: yeniden eğitim yok, GPU yok (Aşama 2'deki N3 hariç, o da koşullu).
- Onaylayıcı aile zaten önceden tanımlıydı: **katılımcı düzeyinde toplanmış 6 dondurulmuş backbone için T1, Holm** (`docs/CONFOUND_CONTROL_DESIGN.md` 5.9; D-025 madde 4). Fine-tune sonuçları protokolde "seçim sonrası, keşifsel" (5.11). Yani projenin onaylayıcı cevabı Faz 2 tahminlerinden çıkacak; Faz 3'ü beklemek bu cevabı değiştirmez, yalnız geciktirir.
- Asıl bağlam tasarımı (CONFOUND Bölüm 6, sonuçlardan önce yazıldı) zaten şu sırayı öngörüyordu: EXP-020 (G1) → EXP-021 (dondurulmuş lens seti) → EXP-022 (koşullu) → EXP-030 (fine-tune). D-028 bunu "geliştirme bitince"ye erteledi. D-037 bu sıraya geri dönmeyi öneriyor.

**Neyin önceden belirlendiği, neyin şimdi sabitlendiği:**

| Öğe | Durum |
|---|---|
| T1 modeli, karar istatistiği, duyarlılıklar, E2h-KLR, Spisak testleri, onaylayıcı aile, C1 / C2 / C3 kuralları | **Önceden belirlenmiş** (D-025, CONFOUND 4 ve 6; ses sonuçlarından önce yazıldı) — içerik değişmiyor |
| Analizin zamanı (Faz 3'ten önce) | D-037 — **EXP-016'nın E1 sonuçlarından sonra** önerildi; T1 sonuçlarından önce |
| D-025'in "o aşamada kesinleştirilir" dediği ayrıntılar (aşağıda Bölüm 2–4'te) | **Şimdi sabitleniyor**, sonuçlardan önce |
| Önceden yazılmış metinden sapmalar | Bölüm 7'de tek tek, gerekçesiyle |

Sonuçlar görüldükten sonra bu plandaki herhangi bir değişiklik **post hoc** olarak etiketlenir ve ayrı raporlanır.

---

## 1. Veri ve birim

- **Kohort:** EXP-016'nın katılımcıları (342: 283 astım / 59 sağlıklı) içinde saat, tarih ve yaşı tam olanlar.
  - EXP-003'e göre beklenen: 339 (282 / 57) [NEEDS VERIFICATION: çalıştırmada sayılır].
  - Dışlananların sayısı ve etiketi raporlanır; başka dışlama yok.
- **Ses skoru ŝ (katılımcı başına, kol başına):**
  - Tekrar r'de katılımcının dış-test füzyon olasılığı p̄ᵣ, EXP-016'daki gibi görev olasılıklarının ortalaması (7 görev; eksik görevde 6).
  - ŝ = (1/5) Σᵣ logit(clip(p̄ᵣ, 1e-6, 1 − 1e-6)).
  - T1 örneklemi içinde z-standardize edilir; OR "ŝ'nin 1 SD'si başına".
  - Her tekrarda her katılımcının tam bir dış-test tahmini vardır (EXP-016'da assert edildi; burada yeniden assert edilir).
- **Kollar:**
  - Onaylayıcı aile: cnn10, cnn14, cnn14_16k, beats, wavlm_base_plus, wavlm_large.
  - Referans (aileye girmez): mfcc_lr, mfcc_mlp.
- **Bağlam değişkenleri** (`participant_context.csv`):
  - `start_hour` (ondalık saat);
  - `days` (2024-01-01'den gün);
  - `age`;
  - `recording_date` (gün kümesi);
  - `in_time_overlap` (E2 penceresi);
  - `period` / `ampm`, `analyze_imbalance_selection.add_cells` ile: erken/geç sınırı = son sağlıklı kaydın tarihi; AM = saat < 12.
- **Girdi bütünlüğü (assert):**
  - `partial/<kol>_r<r>_test.csv` imzaları (split ve bağlam sha256'sı) şimdiki dosyalarla aynı;
  - bütün kollarda aynı katılımcı kümesi;
  - bir tekrarda bir katılımcı tek fold'da;
  - rapora katılımcı ID'si yazılmaz (D-014; ID taraması testte).

## 2. Aşama 1 — CPU, şimdi (bütün kollar)

### L2 — T1 (birincil)
- **Model:** `y ~ cr(start_hour, df=4) + cr(days, df=4) + age + ŝ`, lojistik regresyon.
  - Spline'lar patsy `cr`, `constraints='center'`, düğümler patsy varsayılanı (kantiller).
  - Sütunlar z-standardize; sabit terim var. SIM-001 ile birebir aynı bağlam matrisi.
  - **Fit: Firth cezalı olabilirlik** (S9; KABUL, D-038). Klasik ML fit ve ML LRT yalnız karşılaştırma için raporlanır.
- **Karar istatistiği:** ŝ için **cezalı olabilirlik oranı testi** (PLR, 1 sd, χ²₁; Heinze & Schemper 2002), iki yönlü p.
  - Sıfır modeli: ŝ katsayısı 0'a sabit, ceza yine tam tasarımın bilgi matrisiyle (logistf ile aynı tanım).
  - Firth Wald p ayrıca raporlanır.
- **Etki büyüklüğü:** SD başına OR.
  - %95 CI: katılımcı bootstrap, B = 2000, persentil, her örneklemde T1 modeli (Firth) yeniden fit edilir.
  - Yakınsamayan örneklemler sayılır ve raporlanır.
- **Önceden belirlenmiş duyarlılıklar:**
  1. gün-kümeli (`recording_date`) çıkarım (K2, D-038):
     - **gün-kümeli jackknife (CV3):** her kayıt günü dışarıda bırakılarak Firth T1 tam modeli yeniden fit edilir; V = (G−1)/G · Σ (b₍g₎ − b̂)², t(G−1) referansı (MacKinnon, Nielsen & Webb 2023). C1 koşul 3 bunu kullanır;
     - Firth skor katkılarıyla sandviç (CR1, G/(G−1)): planın ilk metni; bu veri yapısında anti-konservatif bulundu (SIM-002), **yalnız betimsel**;
  2. `cr(·, df=6)` bağlam modeli.
- **Betimsel:** ikinci düzey CV ΔAUC (bağlam vs bağlam + ŝ; tabakalı 5-fold × 20). Karar için kullanılmaz (SIM-001: güçsüz).
- **Aile:** 6 backbone, Holm, α = 0.05.
- **Kararlılık (L8 parçası):** T1 her tekrarın ŝ'siyle ayrı ayrı da hesaplanır; LR istatistiğinin tekrarlar arası aralığı raporlanır.

### L4 — E2h-KLR
- **E2h** = `in_time_overlap` ve saati bilinen katılımcılar. Tabaka = ⌊start_hour⌋. Yalnız iki etiketi de içeren tabakalar.
- **Koşullu lojistik regresyon:** `y ~ ŝ + age + start_hour + days`, z-standardize. ŝ için OR (Wald %95 CI).
  - **Fit: Jeffreys-cezalı (Firth tipi) koşullu logit** (Heinze & Puhr 2010; S12, KABUL, D-038). Tabaka içinde ayrışmada klasik fit hata veriyor. Klasik ML koşullu logit yalnız karşılaştırma için.
- **Ham E2h AUC'si** aynı alt kümede, yeniden hesaplanan üç referans çizgisiyle birlikte raporlanır: yalnız yaş, yalnız bağlam, yaş + bağlam (OOF, CONFOUND 2.3 yöntemi). 0.5'e karşı test edilmez.

### L3 — Spisak tam testi ve kısmi test (C2)
- **Tam test** (destekleyici), H0: ŝ ⊥ Y | C → T1'in parametrik olmayan teyidi.
  - `mlconfound` 0.21.3, `cond_dist_method='linear'`, `num_perms=1000`, `cat_y=True`, `cat_yhat=False`, `cat_c=False`, `random_state=0`.
  - Paket yalnız tek boyutlu C kabul ediyor [FROM OFFICIAL DOCS: paket docstring'i].
  - Bu yüzden C = çapraz-fit edilmiş bağlam eğilim skoru logit p̂(Y | yaş, cr(saat, 4), cr(tarih, 4)): 5-fold tabakalı, katılımcı düzeyi, Firth fit. Bu yedek yol CONFOUND 8'de önceden yazılmıştı.
  - Y ikili olduğu için Y | C modeli lojistiktir; `linear` ayarı bu testi etkilemez [FROM OFFICIAL DOCS].
- **Kısmi test** (C2'nin ölçütü), H0: ŝ ⊥ C | Y → model bağlamı kullanıyor mu? (S10; KABUL, D-038)
  - **Etiket-içi tam koşullu permütasyon testi:** C satırları her etiket grubunun içinde karıştırılır (1000 permütasyon).
  - İstatistik: `ŝ ~ 1 + y + C` regresyonunda C'nin kısmi R²'si.
  - C = cr(saat, 4) + cr(tarih, 4) (kayıt bağlamı). Yaşlı sürüm ayrıca raporlanır (K6).
  - `mlconfound` kısmi testi yalnız bilgi amaçlı raporlanır.

### L5 — Karşıt hücreler
- Etiket × AM/PM hücrelerinde ŝ ortalaması (z), katılımcı-bootstrap %95 CI.
- **Hastalarda:** `ŝ ~ PM + cr(days, 4)` doğrusal regresyon; PM katsayısı ve HC3 %95 CI (C2 ölçütü).
- **Sağlıklılarda:** AM (≈12) vs PM (≈46) farkı; yalnız betimsel (güç çok düşük).

### EXP-020 — G1 bağlam probu (gömmeler üzerinde)
- **(a) Hastalarda AM / PM:**
  - Katılımcı-ortalama gömme (görevler üzerinden ortalama; SSL'de EXP-016'daki katman ortalaması pipeline'ı).
  - LR, C ızgarası iç CV ile (EXP-016 protokolü), 5-fold × 5 tekrar, tabakalı, yalnız hastalar.
  - OOF gömme skoru e (tekrarlar boyunca ortalama logit).
  - Test: `PM ~ cr(days, 4) + e`, e için LR testi.
  - **G1+ (backbone başına):** Holm (6) düzeltilmiş p < 0.05 ve OR > 1. İncremental CV AUC betimsel.
- **(b) Hastalarda erken / geç dönem (N1):** aynı prob, hedef `period`. Klinik profilin dönemi ayırma AUC'si (0.855; EXP-002) referans olarak verilir. Betimsel.

### N5 — Bağlam-vekili kontrolü (yalnız G1+ backbone'larda bilgili)
- (a)'daki AM/PM modeli (yalnız hastalarda eğitilmiş) bütün katılımcılara uygulanır:
  - hastalarda OOF;
  - sağlıklılarda 5 fold modelinin ortalaması.
- Bu vekil skor, ŝ'nin yerine T1'e konur.
- **Beklenti:** bilgi eklememesi (LR p ≥ 0.05). Eklerse, ses bağlamı ölçülen saat ve tarihten daha ince kodluyordur ve o backbone için pozitif T1 tek başına astım bilgisi diye okunamaz.

### N2 — Etiket içinde skor ↔ tarih
Hastalarda `ŝ ~ cr(days, 4) + cr(start_hour, 4)`: tarih terimlerinin HC3-dayanıklı ortak Wald testi. Betimsel; META-016'nın erken/geç göstergesinin ayarlı hâli.

### L7 — Codec zinciri (S8: Aşama 1'e alındı; betimsel)
- Kayıt düzeyinde zincir bilgisi önbellek `index.csv`'sinde var (`chain`).
- Katılımcı içi eşleştirilmiş skor farkı: ffmpeg − apple, kayıt logit'leri. Görevler zincirle karışık, betimsel.
- Kayıt düzeyi zincir probu: gömmelerden, katılımcı-gruplu 5-fold CV; AUC ve katılımcı-küme bootstrap CI.

### Kalibrasyon tanılaması (K3; D-038 ile K2 geçerlilik kuralını da taşır)
- Gerçek etiket ve gerçek bağlam, sinyalsiz yapay skor (SIM-001 S1 / S2 / S3, γ = 0, 500 simülasyon). Ses / model skoru kullanılmaz.
- T1-PLR, CR1 ve CV3'ün bu kohorttaki yanlış pozitif oranı (iki yönlü, Wilson %95 CI).
- PLR için betimsel: karar kuralını değiştirmez.
- **K2 geçerlilik kuralı (D-038, önceden yazıldı):** CV3'ün p < 0.05 ret oranının Wilson %95 alt sınırı S1 ve S3'te ≤ 0.075 olmalı (Bradley'nin liberal ölçütü, 1.5α) ve hiçbir jackknife fit'i başarısız olmamalı. Değilse koşul 3 "geçersiz"; diğer Aşama-1 koşulları sağlansa bile C1 "kurulamadı (K2)" yazılır; yerine başka test konmaz, karar gerekir. Kural yalnız kısıtlayıcı yönde çalışır: C1'i hiçbir zaman kolaylaştırmaz.
- Bu, SIM-002'nin (Bölüm 10; yerel özel kopya) Drive'daki bağlam dosyasıyla çalıştırma anında tekrarıdır.

## 3. Aşama 2 — yalnız T1'i (L2) Holm sonrası pozitif olan backbone'lar için

C1 iddiası ancak bunlarla tamamlanır; negatif T1'de gerekmez.
- **N3 — yalnız arka plan:**
  - Her kaydın en düşük enerjili %20 karesinden oluşturulan sinyalden gömme (yalnız o backbone).
  - Aynı prob protokolü → ŝ_N3 → T1.
  - Beklenti: bilgi eklememesi.
  - GPU: backbone başına dakikalar (Faz 2 çıkarım sürelerine göre) [INFERENCE].
  - Kare seçimi ayrıntısı Aşama 2 başlamadan, sonuç görülmeden yazılır.
  - "N3 etkiyi yeniden üretiyor" tanımı (K7, Aşama 2'den önce kesinleşecek; öneri): ŝ_N3 ile T1-PLR p < 0.05 ve OR > 1 (Holm'suz; çürütme kontrolü olduğu için katı).
- **Gün-gruplu CV duyarlılığı:**
  - Probu `recording_date` gruplarına göre CV ile yeniden fit et (CPU, saatler);
  - T1'i yeni ŝ ile tekrarla.

## 4. Karar kuralları (CONFOUND 6'dan; D-038 ile kabul edilen uygulama değişiklikleri işaretli)

- **C1** (bir backbone için) — hepsi gerekir:
  1. Holm-düzeltilmiş T1 LR p < 0.05 ve OR > 1;
  2. OR bootstrap CI'ı 1'i dışlıyor;
  3. **gün-kümeli jackknife (CV3) p < 0.05 ve OR > 1** (K2, D-038; önceki metin "gün-kümeli duyarlılıkta yön korunuyor" boş bir koşuldu). Kalibrasyon tanılaması CV3'ü geçersiz bulursa C1 "kurulamadı (K2)";
  4. E2h-KLR OR'u > 1 (CI şart değil);
  5. cr df 4 ve df 6 aynı yön;
  6. N3 etkiyi yeniden üretmiyor;
  7. ~~G1+ ise N5 anlamlı değil~~ — **S11 (KABUL, D-038): N5 yalnız betimsel, C1'e dahil değil** (Bölüm 9 K15).
- **C1 kimse için sağlanmazsa:** "Bu veriyle bağlamdan bağımsız bir ses sinyali gösterilemedi". Güç sınırıyla birlikte raporlanır (iyimser değerler): T1 tek başına %80 güce aynı-bağlam AUC'si ≈ 0.69'da (α 0.05) / ≈ 0.74'te (Holm düzeyi) ulaşıyor (SIM-001); Aşama-1 kuralıyla (koşul 1 ∧ koşul 3, CV3) Holm düzeyinde ≈ 0.77'de (SIM-002).
- **C2** (backbone başına; Holm yok): L3 kısmi test (etiket-içi permütasyon, C = saat + tarih; S10) p < 0.05 **veya** L5 hasta-içi PM katsayısının CI'ı 0'ı dışlıyor.
  - Gerekçe: C2 bir uyarı iddiasıdır. Kaçırmak, yanlış alarmdan daha pahalıdır.
  - C1 ve C2 birlikte doğru olabilir.
- **MFCC kolları:** aynı lensler, yalnız referans. "MFCC bile bağlamın ötesinde bilgi taşıyor mu?" sorusuna betimsel cevap.

## 5. Faz 3'e etkisi (sonuçlardan önce sabitleniyor)

1. **Backbone seçimi değişmez:** D-035 (KABUL; CNN10, CNN14 sabit; BEATs / WavLM fold başına). D-027'nin "T1'e göre seç" önerisi D-035 ile yerini aldı. T1 sonucuna bakıp seçimi değiştirmek post hoc olur; yapılmaz.
2. **D-027 kol B'nin koşulu:** bağlam-dengeli ağırlıklandırma kolu, Faz 3'te kullanılacak bir backbone G1+ ise ya da onun L3 kısmi testi p < 0.05 ise planlanır. Bu koşul önceden yazılmıştı; burada yalnız uygulanıyor.
3. **Faz 3'ün ölçeği, epoch kuralı ve GPU bütçesi** `docs/PHASE3_DESIGN.md`'de, EVAL-016 Aşama 1 raporundan sonra yazılır. Bu belge **"T1 sonrası"** diye etiketlenir. Faz 3 protokolde zaten keşifseldir (CONFOUND 5.11).
4. **Faz 3'ün değerlendirme ölçütü önceden sabit:** C3 = fine-tune vs dondurulmuş, aynı katılımcılarda eşleştirilmiş T1 sapma azalması farkı (bootstrap CI) + L3 kısmi test. E1 AUC artışı tek başına başarı sayılmaz.
5. **GPU şartları:** EVAL-016 Aşama 1 raporu, `PHASE3_DESIGN.md` kabulü ve SMK-002 PASS olmadan Faz 3 GPU koşusu başlamaz.

## 6. Yorum sınırları (klinik anlam abartılmaz)

- Pozitif C1 bile "astım tanısı koyan bir ses testi" anlamına gelmez:
  - vaka-kontrol tasarımı, tek merkez, tek cihaz;
  - sağlıklı n = 59 (sabah 12);
  - geç dönemde sağlıklı yok (pozitiflik sınırı);
  - dış doğrulama yok;
  - spirometri / bronkodilatör sonrası kayıt olasılığı (S13; veri ekibinden cevap bekleniyor) saat kontrolüyle ayrılamaz.
- T1, **ölçülmüş** bağlamı ayarlar. Ölçülmemiş, güne özgü bağlamda T1-PLR'nin yanlış pozitif oranı ~%11–12 (p < 0.05) olabilir (SIM-001 S3; SIM-002). Sandviç CR1 bunu düzeltmez. CV3 (koşul 3) simülasyonda bunu nominalin altına indiriyor (SIM-002), ama bu yalnız **sentetik, normal dağılımlı** bir gün etkisi için gösterildi; gerçek gün etkileri başka biçimde olabilir. N3 ve gün-gruplu CV (Aşama 2) bu riski ayrıca sınar.
- Olumsuz T1, "astım bilgisi yok" demek değildir; "saptanabilir büyüklükte kanıt yok" demektir.

## 7. Önceden yazılmış metinden sapmalar (hepsi sonuçlardan önce)

| # | Önceki metin | Bu plan | Neden |
|---|---|---|---|
| S1 | D-028: değerlendirme model geliştirme bitince | Faz 3'ten önce (D-037) | Bölüm 0. **E1 sonuçlarından sonra** önerildi |
| S2 | EXP-021: lens setiyle probu yeniden çalıştır | EXP-016'nın kaydedilmiş tahminleri kullanılır | Prob protokolü aynı (D-034); yeniden eğitim gereksiz GPU/CPU harcaması |
| S3 | Spisak: `cond_dist_method` önerilen `gam` | `linear` | `mlconfound` 0.21.3 `pygam==0.8.0`'a sabitli; bu pygam sürümü güncel scipy'de çalışmıyor (`csr_matrix.A` kaldırılmış; yerelde numpy 2.5 / scipy 1.18 ile doğrulandı). `linear` çalışıyor (yerelde doğrulandı; `setuptools<81` gerekiyor) [FACT, yerel test]. Colab'da yeniden doğrulanır |
| S4 | G1: "artımlı AUC'nin alt %95 sınırı permütasyon dağılımının %97.5'ini aşıyor" (permütasyon şeması tanımsız) | `PM ~ cr(days) + e` LR testi, Holm, OR > 1; artımlı CV AUC betimsel | Tanımsız şema yerine T1 ile aynı, doğrulanmış makine; uygulanabilir ve önceden tam belirli |
| S5 | EXP-021: gün-gruplu CV ve E4 duyarlılıkları birlikte | Gün-gruplu CV Aşama 2'de (koşullu); E4 (pencere içinde eğitim) keşifsel, sonraya | CPU / iş yükü; C1 için gerekli olanlar korunuyor |
| S6 | N3 her modelde | Yalnız T1-pozitif backbone'larda (Aşama 2) | N3 yalnız pozitif bir iddiayı çürütmek için gerekli; GPU tasarrufu |
| S7 | T1 kohortu (SIM-001): `in_paper_cohort` | EXP-016 kohortu ∩ bağlamı tam olanlar | Ses skorları EXP-016 kohortunda; sayılar raporlanır |
| S8 | (bu planın ilk sürümü) L7 Aşama 2'de, koşullu | L7 Aşama 1'de, bütün kollar | CONFOUND lens tablosu L7'yi "zorunlu" sayıyor; zincir bilgisi önbellek index'inde var, CPU'da ucuz (K1) |
| S9 | D-025: T1 "olabilirlik oranı testi" (klasik ML) | Firth cezalı olabilirlik + PLR (tahmin, test, bootstrap, gün-kümeli sandviç); ML LRT ikincil | Sentetik testte yarı-tam ayrışmada ML fit'i yakınsamadı, 200 bootstrap örneğinin 200'ü başarısız oldu → C1 koşul 2 mekanik olarak "✗" çıktı (gerçek sinyale rağmen). Gerçek veride aynı yapı var (son sağlıklı kayıttan sonra yalnız hasta; bağlam AUC 0.93; güçlü ŝ). **KABUL (D-038)** (K4) |
| S10 | D-025: Spisak kısmi testi (`mlconfound`) C2'nin ölçütü | Etiket-içi tam koşullu permütasyon testi; C = saat + tarih (yaşlı sürüm ayrıca); mlconfound kısmi yalnız bilgi | Sentetik testte `mlconfound` kısmi testi (`linear`) bağlamı hiç kullanmayan güçlü bir skoru p < 0.001 ile "bağlam kullanıyor" diye işaretledi (yanlış pozitif). Y ikili olduğu için tam permütasyon testi model varsayımı gerektirmez. **KABUL (D-038)** (K5, K6) |
| S11 | CONFOUND 6 / D-025 madde 5: C1 için "N5 bilgiliyse anlamlı değil" | N5 raporlanır ama C1'de veto değil | Sentetik testte (E12) hem gerçek astım sinyali hem bağlam kodlayan bir gömmede N5 p < 0.001 çıktı: hastalarda eğitilip sağlıklılara uygulanan AM/PM vekil modeli, sağlıklılarda dağılım dışı kalan astım boyutlarından sinyal taşıyor. Sinyal boyutu çıkarılınca N5 p = 0.19 (tanılama). Yani N5, gerçek sinyal + bağlam durumunda (gerçekçi durum) C1'i yapısal olarak düşürür ve "ince bağlam" ile "gerçek sinyal"i ayıramaz. **KABUL (D-038)** (K15) |
| S12 | D-025: E2h-KLR (klasik koşullu logit) | Jeffreys-cezalı koşullu logit; klasik ikincil | Sentetik testte gerçek sinyalde klasik fit hata verdi (tabaka içi ayrışma) → C1 koşul 4 mekanik "✗" (K11). R `survival::coxph` (exact) tabanlı bağımsız uygulamayla 1e-4 içinde aynı (E13). **KABUL (D-038)** |
| K2 | CONFOUND 6 / D-025: C1 koşul 3 "gün-kümeli duyarlılıkta yön korunuyor"; planın ilk sürümü: gün-kümeli SE = Firth sandviç (CR1) | Koşul 3 = **gün-kümeli jackknife (CV3) p < 0.05 ve OR > 1**; CR1 yalnız betimsel; çalıştırma anında K2 geçerlilik kuralı (Bölüm 2, kalibrasyon) | Yön koşulu boştu (K2). Alper K2'yi "gün-kümeli test bu veri yapısında geçerliyse" koşuluyla kabul etti. SIM-002 (Bölüm 10): CR1 doğru belirtilmiş modelde bile ~%10 yanlış pozitif veriyor ve ölçülmemiş gün etkisini düzeltmiyor → koşul CR1 ile sağlanmıyor. CV3 üç senaryoda nominalde ya da altında. **KABUL (D-038)**; CR1 → CV3 değişikliği 16. turda, gerçek sonuçlardan önce, açıkça kaydedildi |

## 8. Çıktılar ve maliyet

- **Git (agrega, ID'siz):** `reports/context/EVAL-016_context.{md,json}` (EXP-020 G1 ve N5 bu raporun 4. bölümünde).
- **Drive:** `experiments/context_eval/eval016_participant_scores.csv` (katılımcı düzeyi ŝ).
- **Maliyet:** simüle Colab'da (2 CPU, 195 sentetik katılımcı, sabit ayarlar, CV3 dahil) EVAL-016 1539 s (~26 dk) sürdü [FACT, 16. tur]. Gerçek kohort ~1.7 kat büyük → Colab CPU'da ~40–80 dk [INFERENCE].
- **Sentetik testler** (`tests/test_eval016.py`):
  - E2 gerçek sinyal → T1 reddeder, Aşama-1 koşulları ✓;
  - E3 yalnız bağlam kodlayan skor → reddetmez, G1+, N5 bilgi eklemez, C2 pozitif;
  - E4 gürültü → reddetmez;
  - E8 kalibrasyon (60 veri seti, ret oranı ≤ 0.15);
  - E10 Firth = R brglm2 / profil cezalı olabilirlik;
  - E11 kısmi test bağlamsız sinyalde reddetmez;
  - E12 N5 yanlış alarmı (S11 gerekçesi); test kipi raporda işaretli; test ayarlarıyla gerçek kip reddedilir;
  - E13 cezalı koşullu logit = R survival;
  - E14 CR1 ve CV3 = statsmodels ML ile bağımsız hesap;
  - E15 kapı (sabit ayarlar) ve K2 fail-safe kuralı;
  - E16 kalibrasyon tanılaması CV3'ü ve K2 kuralını hesaplar;
  - E1, E5–E7, E9 bütünlük, ID, imza, eksik görev.

## 9. Tutarlılık kontrolü (15. tur; kodlama ve sentetik testler sırasında, gerçek veriden önce)

Plan; D-023, D-025, D-028, D-035, D-037 ve CONFOUND 4–6 (lens seti, C1 / C2 / C3) ile karşılaştırıldı. Bulgular:

| # | Bulgu | Çözüm | Durum |
|---|---|---|---|
| K1 | Plan L7'yi Aşama 2'ye koymuştu; CONFOUND L7'yi "zorunlu" sayıyor ve bu sapma tabloda yoktu | L7 Aşama 1'e alındı (S8) | Çözüldü (protokole dönüş) |
| K2 | C1 koşul 3 ("gün-kümeli duyarlılıkta yön korunuyor") **boş bir koşul**: kümeli SE nokta tahminini değiştirmez, yön hep ana fitle aynıdır | Koşul 3 = gün-kümeli **jackknife CV3** p < 0.05 ve OR > 1 (K16 nedeniyle CR1 değil); çalıştırma anında geçerlilik kuralı | **KABUL (D-038)** — Alper'in koşulu (geçerlilik teyidi) SIM-002 ile CV3 için sağlandı, CR1 için sağlanmadı |
| K3 | SIM-001 kalibrasyonu ML Wald için yapılmıştı; karar istatistiği LRT / PLR | Gerçek bağlamda kalibrasyon tanılaması eklendi (betimsel) | Çözüldü |
| K4 | Yarı-tam ayrışma: ML T1 fit'i ve bootstrap çöküyor; C1 koşul 2 mekanik olarak başarısız | Firth (S9); R brglm2 ve R'de bağımsız profil optimizasyonuyla doğrulandı (E10) | **KABUL (D-038)** |
| K5 | `mlconfound` kısmi testi (`linear`) bağlamsız güçlü skorda yanlış pozitif | Etiket-içi tam permütasyon testi (S10) | **KABUL (D-038)** |
| K6 | C2'nin C'sinde yaş olacak mı belirsiz (D-025 eğilim skoru yaşı içeriyordu) | Birincil C = saat + tarih (kayıt bağlamı); yaş bir kişi özelliği ve ses yaşı meşru biçimde kodlar → yaşlı sürüm yalnız rapor | **KABUL (D-038)** |
| K7 | N3'ün "etkiyi yeniden üretmiyor" tanımı yok | Öneri Bölüm 3'te; Aşama 2'den önce kesinleşir | Aşama 2'ye kadar açık |
| K8 | N5'te sağlıklıların vekil skoru 25 modelin ortalaması, hastalarınki 5 tekrarın OOF ortalaması (asimetri) | Belgelendi; N5 yalnız G1+ backbone'larda bilgili | Bilinen kısıt |
| K9 | L1 tasarımları (E1h, E2, E3 …) planda sayılmamıştı | E1, E1h, E2, E2h, E3, E3h + CONFOUND 2.3 yöntemiyle referanslar | Çözüldü |
| K10 | Aşama 2 tetikleyicisi (Holm-T1 pozitif), diğer Aşama-1 koşulları sağlanmasa bile çalışır | Değiştirilmedi: N3, C1 düşse bile mekanizma hakkında bilgi verir | Kabul edilen kural |
| K11 | E2h-KLR klasik koşullu logit; tabaka içinde ayrışmada fit hata veriyor → C1 koşul 4 mekanik olarak ✗ (sentetik testte gerçekleşti) | Jeffreys-cezalı koşullu logit (S12) | **KABUL (D-038)** |
| K12 | Gerçek `participant_context.csv`'de `in_time_overlap` ve `recording_date` sütunları [NEEDS VERIFICATION] | Script assert eder ve açık mesajla durur | Çalıştırmada doğrulanır |
| K13 | N3 "yalnız arka plan" = en düşük enerjili kareler; konuşma kayıtlarında bu kareler **nefes seslerini** de içerir ve astımla fizyolojik olarak ilişkili olabilir → N3'ün etkiyi "yeniden üretmesi" bağlam kanıtı olmayabilir | Aşama 2 tasarımında kare seçimi nefes / ses dışı gürültüyü ayıracak biçimde yazılacak (ör. sessizlik + spektral düzlük ölçütü); belirsizlik raporda | Aşama 2'den önce |
| K14 | Onaylanmamış uygulama ayrıntılarıyla gerçek veride yanlışlıkla çalıştırma riski | `PROTOCOL` sözlüğü (sürüm + ayarlar) rapora yazılır; `status != "KABUL"` iken script gerçek veri kipinde çalışmayı reddeder (testler `--allow-unapproved`) | Çözüldü; K17 ile genişletildi |
| K16 | (16. tur, K2 teyidi sırasında) Planın gün-kümeli SE'si (Firth sandviç, CR1) gerçek bağlam yapısında **anti-konservatif**: doğru belirtilmiş modelde p < 0.05 oranı ~0.10; ölçülmemiş gün etkisini düzeltmiyor (SIM-002). Neden [INFERENCE]: 65 günün 53'ü tek etiketli, bilgi az sayıda günden geliyor; az ve dengesiz kümede CR1 varyansı küçümser (bilinen sonuç) | CV3 jackknife (her gün dışarıda bırakılarak yeniden fit, t(G−1)); CR1 betimsel kalır, raporda görünür | **KABUL (D-038)**; olumsuz sonuç korunuyor (SIM-002) |
| K18 | (16. tur) Bilgi-matrisli Firth adımları yarı-tam ayrışmalı, çok düz bir yüzeyde yakınsamıyor (sentetik testte bir jackknife alt örneklemi; 3000 adımda bitmedi; R brglm2 6105 iterasyon gerektirdi). Jackknife fit'i "başarısız" sayılınca K2 kuralı sentetik testte yanlışlıkla "geçersiz" çıktı | Yakınsamazsa Newton-CG cilası (U tam gradyan, Hessian merkezi farklarla); tahminci aynı; R brglm2 ile 6 ondalığa kadar aynı (E17) | Çözüldü (D-038 madde 8) |
| K17 | Analiz ayarları (B, permütasyon, simülasyon sayısı, tohum, backbone listesi) komut satırından değiştirilebiliyordu; sonuçtan sonra ayar değiştirme riski | `FIXED` sözlüğü; gerçek veri kipinde herhangi bir sapma çalışmayı durdurur; `--allow-unapproved` raporlara "karar için kullanılamaz" yazar (E12, E15) | Çözüldü (D-038) |
| K15 | N5 tasarım kusuru (S11): gerçek sinyal + bağlam birlikteyken vekil skor astım sinyalini taşır | N5 betimsel; G1 ve C2 bağlam kullanımını göstermeye devam eder; "ince bağlam" sorusu için gün-gruplu CV (Aşama 2) kalır | **KABUL (D-038)** |

## 10. K2 geçerlilik çalışması — SIM-002 (16. tur; gerçek bağlam yapısı, yapay skor; ses sonucu YOK)

**Neden:** Alper K2'yi "gün-kümeli test mevcut veri yapısında geçerli biçimde uygulanabiliyorsa" koşuluyla kabul etti. Bu bölüm o teyidin kaydıdır (D-038).

**Veri:**
- Faz 0'da üretilen `participant_context.csv`'nin özel yerel kopyası ve EXP-016 split'i (git'e girmez).
- T1 kohortu 339 (282 / 57); 65 kayıt günü (53 tek etiketli, 12 karışık); sabah sağlıklı 12; öğleden sonra hasta 86; geç dönem hasta 185. EXP-003 ve CONFOUND'daki sayılarla aynı. [FACT]
- **Hiçbir ses / model skoru kullanılmadı;** EVAL-016 sonuçlarına bakılmadı.

**Yöntem:**
- SIM-001 senaryoları: S1 düzgün bağlam (doğru belirtilmiş model), S2 basamak, S3 ölçülmemiş gün etkisi. `skor = γ · astım + kestirme(bağlam) + N(0, 1)`.
- γ = 0'da 1000, γ > 0'da 400 simülasyon; her simülasyonda T1-PLR, CR1 ve CV3 (aynı veri).
- Script: `scripts/simulate_k2_validity.py` (tohum 2026) → `reports/context/SIM-002_k2_validity.{md,json}`.

**Yanlış pozitif ve güç** (iki yönlü p < 0.05 oranı [Wilson %95]):

| senaryo | T1-PLR | CR1 sandviç | **CV3 jackknife** |
|---|---|---|---|
| S1 düzgün, γ = 0 | 0.039 [0.029, 0.053] | **0.090** [0.074, 0.109] | 0.027 [0.019, 0.039] |
| S2 basamak, γ = 0 | 0.056 [0.043, 0.072] | **0.103** [0.086, 0.123] | 0.026 [0.018, 0.038] |
| S3 gün etkisi, γ = 0 | **0.138** [0.118, 0.161] | **0.144** [0.124, 0.167] | 0.044 [0.033, 0.059] |
| S1, γ = 0.5 (aynı-bağlam AUC ≈ 0.64) | 0.432 | 0.530 | 0.282 |
| S1, γ = 0.75 (AUC ≈ 0.70) | 0.823 | 0.858 | 0.578 |
| S3, γ = 0.75 | 0.525 | 0.525 | 0.242 |
| S1, γ = 1.0 (AUC ≈ 0.76) | 0.965 | 0.970 | 0.808 |
| S1, γ = 1.25 (AUC ≈ 0.81) | 0.995 | 0.998 | 0.923 |

**C1 biçiminde** (tek yönlü: p < α ve OR > 1; α = 0.0083, 6 backbone için Holm'un en sıkı eşiği):

| senaryo | koşul 1 | koşul 1 ∧ CR1 | koşul 1 ∧ CV3 |
|---|---|---|---|
| S1, γ = 0 | 0.004 | 0.004 | 0.004 |
| S3, γ = 0 | 0.015 | 0.015 | 0.012 |
| S1, γ = 0.5 | 0.177 | 0.177 | 0.150 |
| S1, γ = 0.75 | 0.517 | 0.517 | 0.435 |
| S1, γ = 1.0 (AUC ≈ 0.76) | 0.875 | 0.875 | 0.765 |
| S1, γ = 1.25 (AUC ≈ 0.81) | 0.980 | 0.980 | 0.913 |
| S3, γ = 0.75 | 0.330 | 0.328 | 0.212 |

α = 0.05 eşiğinde (Holm'un en gevşek adımı): S3, γ = 0 → koşul 1 0.075, ∧ CV3 0.017; S1, γ = 0.75 → 0.823, ∧ CV3 0.557.

**Yorum:**
- **CR1 geçersiz** [FACT, simülasyon]: doğru belirtilmiş modelde bile p < 0.05 oranı ~0.09 (nominalin ~2 katı); gün etkisini düzeltmiyor (0.144). K2'nin planlanan biçimi Alper'in koşulunu sağlamıyor. Bu olumsuz sonuç korunuyor; CR1 raporda betimsel kalıyor.
- **CV3 geçerli** [FACT, simülasyon]: üç senaryoda da nominalde ya da altında; 6000 jackknife fit'inin hiçbiri başarısız değil.
- **Koruma sınırlı** [FACT, simülasyon]: Holm'un en sıkı eşiğinde koşul 3, gün etkisi yanlış pozitifini yalnız 0.015 → 0.012'ye indiriyor; α = 0.05 eşiğinde 0.075 → 0.017. Gün etkisine karşı asıl koruma Aşama 2 (N3, gün-gruplu CV).
- **Güç maliyeti** [FACT, simülasyon]: Holm düzeyinde aynı-bağlam AUC ≈ 0.70'te 0.52 → 0.44, ≈ 0.76'da 0.88 → 0.77. %80 güç için gereken aynı-bağlam AUC'si ≈ 0.74'ten ≈ 0.77'ye çıkıyor [INFERENCE: ara değer]. Olumsuz bir C1, bu güç sınırıyla okunur.
- CR1'in neden küçümsediği [INFERENCE]: 65 günün 53'ü tek etiketli; ŝ katsayısı hakkındaki bilgi az sayıda günden geliyor; az ve dengesiz kümede sandviç varyansı aşağı yanlıdır.

**Sınırlar:**
- Skor yapay, tek boyutlu, normal gürültülü; gerçek ŝ farklı dağılabilir.
- Yerel kopya ile Drive dosyasının bayt bayt aynı olduğu doğrulanmadı [NEEDS VERIFICATION]; çalıştırma anındaki kalibrasyon tanılaması (K2 kuralı) aynı kontrolü Drive dosyasıyla tekrarlar.
- Gün etkisi S3'te normal ve etiketten bağımsız; gerçek gün etkileri (oda, cihaz, operatör) başka biçimde olabilir.

**Ön koşular (aynı tur; script'leri repoda değil, sayıları korunuyor):**
- Gerçek bağlam yapısında 2000 simülasyonluk PLR / CR1 koşusu: S1'de CR1 0.10, CR1 + t(G−1) 0.094; S3'te CR1 0.12. SIM-002 ile tutarlı.
- Kohort agregalarını taklit eden **sentetik** bir kohortla pilot: S2 basamak senaryosunda PLR bile ~0.20 yanlış pozitif verdi. Neden: sentetik saatler 12'de keskin bir basamakla üretilmişti. Gerçek saat dağılımında bu olmadı (SIM-002 S2: 0.056). Sentetik yapının sınırlılığıdır; bu yüzden geçerlilik kararı gerçek bağlam yapısına dayandırıldı.
