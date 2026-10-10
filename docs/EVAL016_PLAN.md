# EVAL-016 — Dondurulmuş aile için bağlam değerlendirmesi: analiz planı

**Durum:** D-025 (içerik değişmeden) ve D-037 **KABUL** (2026-10-10, 15. tur, Alper). Kod (`scripts/evaluate_context.py`), sentetik testler (`tests/test_eval016.py`, E1–E11) ve `notebooks/22_context_eval_frozen.ipynb` hazır. **Gerçek veride ÇALIŞTIRILMADI.**
**Gerçek veride çalıştırmadan önce onay bekleyenler:** kodlama ve sentetik testler sırasında bulunan uygulanabilirlik sorunları ve önerilen çözümleri (S9–S12, K2; Bölüm 9). Script'te `PROTOCOL["status"]` "KABUL" olmadan gerçek veride çalışmayı reddeder.
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
  - **Fit: Firth cezalı olabilirlik** (S9; onay bekliyor). Klasik ML fit ve ML LRT yalnız karşılaştırma için raporlanır.
- **Karar istatistiği:** ŝ için **cezalı olabilirlik oranı testi** (PLR, 1 sd, χ²₁; Heinze & Schemper 2002), iki yönlü p.
  - Sıfır modeli: ŝ katsayısı 0'a sabit, ceza yine tam tasarımın bilgi matrisiyle (logistf ile aynı tanım).
  - Firth Wald p ayrıca raporlanır.
- **Etki büyüklüğü:** SD başına OR.
  - %95 CI: katılımcı bootstrap, B = 2000, persentil, her örneklemde T1 modeli (Firth) yeniden fit edilir.
  - Yakınsamayan örneklemler sayılır ve raporlanır.
- **Önceden belirlenmiş duyarlılıklar:**
  1. gün-kümeli (`recording_date`) dayanıklı SE ile Wald p ve OR yönü (Firth skor katkılarıyla sandviç; G/(G−1) düzeltmesi);
  2. `cr(·, df=6)` bağlam modeli.
- **Betimsel:** ikinci düzey CV ΔAUC (bağlam vs bağlam + ŝ; tabakalı 5-fold × 20). Karar için kullanılmaz (SIM-001: güçsüz).
- **Aile:** 6 backbone, Holm, α = 0.05.
- **Kararlılık (L8 parçası):** T1 her tekrarın ŝ'siyle ayrı ayrı da hesaplanır; LR istatistiğinin tekrarlar arası aralığı raporlanır.

### L4 — E2h-KLR
- **E2h** = `in_time_overlap` ve saati bilinen katılımcılar. Tabaka = ⌊start_hour⌋. Yalnız iki etiketi de içeren tabakalar.
- **Koşullu lojistik regresyon:** `y ~ ŝ + age + start_hour + days`, z-standardize. ŝ için OR (Wald %95 CI).
  - **Fit: Jeffreys-cezalı (Firth tipi) koşullu logit** (Heinze & Puhr 2010; S12, onay bekliyor). Tabaka içinde ayrışmada klasik fit hata veriyor. Klasik ML koşullu logit yalnız karşılaştırma için.
- **Ham E2h AUC'si** aynı alt kümede, yeniden hesaplanan üç referans çizgisiyle birlikte raporlanır: yalnız yaş, yalnız bağlam, yaş + bağlam (OOF, CONFOUND 2.3 yöntemi). 0.5'e karşı test edilmez.

### L3 — Spisak tam testi ve kısmi test (C2)
- **Tam test** (destekleyici), H0: ŝ ⊥ Y | C → T1'in parametrik olmayan teyidi.
  - `mlconfound` 0.21.3, `cond_dist_method='linear'`, `num_perms=1000`, `cat_y=True`, `cat_yhat=False`, `cat_c=False`, `random_state=0`.
  - Paket yalnız tek boyutlu C kabul ediyor [FROM OFFICIAL DOCS: paket docstring'i].
  - Bu yüzden C = çapraz-fit edilmiş bağlam eğilim skoru logit p̂(Y | yaş, cr(saat, 4), cr(tarih, 4)): 5-fold tabakalı, katılımcı düzeyi, Firth fit. Bu yedek yol CONFOUND 8'de önceden yazılmıştı.
  - Y ikili olduğu için Y | C modeli lojistiktir; `linear` ayarı bu testi etkilemez [FROM OFFICIAL DOCS].
- **Kısmi test** (C2'nin ölçütü), H0: ŝ ⊥ C | Y → model bağlamı kullanıyor mu? (S10; onay bekliyor)
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

### Kalibrasyon tanılaması (K3; betimsel)
- Gerçek etiket ve gerçek bağlam, sinyalsiz yapay skor (SIM-001 S1 / S2 / S3, γ = 0, 500 simülasyon).
- T1-PLR'nin bu kohorttaki yanlış pozitif oranı. Karar kuralını değiştirmez; SIM-001 Wald ile yapılmıştı, bu kontrol karar istatistiğinin (PLR) kendisi içindir.

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

## 4. Karar kuralları (değişmedi; CONFOUND 6'dan)

- **C1** (bir backbone için) — hepsi gerekir:
  1. Holm-düzeltilmiş T1 LR p < 0.05 ve OR > 1;
  2. OR bootstrap CI'ı 1'i dışlıyor;
  3. gün-kümeli duyarlılıkta yön korunuyor;
  4. E2h-KLR OR'u > 1 (CI şart değil);
  5. cr df 4 ve df 6 aynı yön;
  6. N3 etkiyi yeniden üretmiyor;
  7. G1+ ise N5 anlamlı değil — **S11 önerisi: yalnız betimsel, C1'e dahil değil** (onay bekliyor; Bölüm 9 K15).
- **C1 kimse için sağlanmazsa:** "Bu veriyle bağlamdan bağımsız bir ses sinyali gösterilemedi". Güç sınırıyla birlikte raporlanır (SIM-001: %80 güç için aynı-bağlam AUC'si ≈ 0.69 α 0.05'te, ≈ 0.74 Holm düzeyinde; bu değerler iyimser).
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
- T1, **ölçülmüş** bağlamı ayarlar. Ölçülmemiş, güne özgü bağlamda yanlış pozitif ~%12 (p < 0.05) / ~%3 (p < 0.01) olabilir (SIM-001 S3). Gün-kümeli duyarlılık bunu düzeltmez, yalnız işaret eder.
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
| S9 | D-025: T1 "olabilirlik oranı testi" (klasik ML) | Firth cezalı olabilirlik + PLR (tahmin, test, bootstrap, gün-kümeli sandviç); ML LRT ikincil | Sentetik testte yarı-tam ayrışmada ML fit'i yakınsamadı, 200 bootstrap örneğinin 200'ü başarısız oldu → C1 koşul 2 mekanik olarak "✗" çıktı (gerçek sinyale rağmen). Gerçek veride aynı yapı var (son sağlıklı kayıttan sonra yalnız hasta; bağlam AUC 0.93; güçlü ŝ). **Onay bekliyor** (K4) |
| S10 | D-025: Spisak kısmi testi (`mlconfound`) C2'nin ölçütü | Etiket-içi tam koşullu permütasyon testi; C = saat + tarih (yaşlı sürüm ayrıca); mlconfound kısmi yalnız bilgi | Sentetik testte `mlconfound` kısmi testi (`linear`) bağlamı hiç kullanmayan güçlü bir skoru p < 0.001 ile "bağlam kullanıyor" diye işaretledi (yanlış pozitif). Y ikili olduğu için tam permütasyon testi model varsayımı gerektirmez. **Onay bekliyor** (K5, K6) |
| S11 | CONFOUND 6 / D-025 madde 5: C1 için "N5 bilgiliyse anlamlı değil" | N5 raporlanır ama C1'de veto değil | Sentetik testte (E12) hem gerçek astım sinyali hem bağlam kodlayan bir gömmede N5 p < 0.001 çıktı: hastalarda eğitilip sağlıklılara uygulanan AM/PM vekil modeli, sağlıklılarda dağılım dışı kalan astım boyutlarından sinyal taşıyor. Sinyal boyutu çıkarılınca N5 p = 0.19 (tanılama). Yani N5, gerçek sinyal + bağlam durumunda (gerçekçi durum) C1'i yapısal olarak düşürür ve "ince bağlam" ile "gerçek sinyal"i ayıramaz. **Onay bekliyor** (K15) |
| S12 | D-025: E2h-KLR (klasik koşullu logit) | Jeffreys-cezalı koşullu logit; klasik ikincil | Sentetik testte gerçek sinyalde klasik fit hata verdi (tabaka içi ayrışma) → C1 koşul 4 mekanik "✗" (K11). R `survival::coxph` (exact) tabanlı bağımsız uygulamayla 1e-4 içinde aynı (E13). **Onay bekliyor** |

## 8. Çıktılar ve maliyet

- **Git (agrega, ID'siz):** `reports/context/EVAL-016_context.{md,json}` (EXP-020 G1 ve N5 bu raporun 4. bölümünde).
- **Drive:** `experiments/context_eval/eval016_participant_scores.csv` (katılımcı düzeyi ŝ).
- **Maliyet:** simüle Colab'da (2 CPU, 195 sentetik katılımcı, varsayılan ayarlar) ölçülen süre notebook 22'nin PR açıklamasında. Gerçek kohort ~1.7 kat büyük [INFERENCE].
- **Sentetik testler** (`tests/test_eval016.py`):
  - E2 gerçek sinyal → T1 reddeder, Aşama-1 koşulları ✓;
  - E3 yalnız bağlam kodlayan skor → reddetmez, G1+, N5 bilgi eklemez, C2 pozitif;
  - E4 gürültü → reddetmez;
  - E8 kalibrasyon (60 veri seti, ret oranı ≤ 0.15);
  - E10 Firth = R brglm2 / profil cezalı olabilirlik;
  - E11 kısmi test bağlamsız sinyalde reddetmez;
  - E1, E5–E7, E9 bütünlük, ID, imza, eksik görev.

## 9. Tutarlılık kontrolü (15. tur; kodlama ve sentetik testler sırasında, gerçek veriden önce)

Plan; D-023, D-025, D-028, D-035, D-037 ve CONFOUND 4–6 (lens seti, C1 / C2 / C3) ile karşılaştırıldı. Bulgular:

| # | Bulgu | Çözüm | Durum |
|---|---|---|---|
| K1 | Plan L7'yi Aşama 2'ye koymuştu; CONFOUND L7'yi "zorunlu" sayıyor ve bu sapma tabloda yoktu | L7 Aşama 1'e alındı (S8) | Çözüldü (protokole dönüş) |
| K2 | C1 koşul 3 ("gün-kümeli duyarlılıkta yön korunuyor") **boş bir koşul**: kümeli SE nokta tahminini değiştirmez, yön hep ana fitle aynıdır | Kabul edilen kural olduğu gibi uygulanıyor; gün-kümeli p raporda | **Karar gerekli:** koşulu "gün-kümeli p < 0.05 (ve OR > 1)" olarak düzeltmek (önerim; daha katı, D-025'in amacıyla uyumlu) ya da olduğu gibi bırakmak |
| K3 | SIM-001 kalibrasyonu ML Wald için yapılmıştı; karar istatistiği LRT / PLR | Gerçek bağlamda kalibrasyon tanılaması eklendi (betimsel) | Çözüldü |
| K4 | Yarı-tam ayrışma: ML T1 fit'i ve bootstrap çöküyor; C1 koşul 2 mekanik olarak başarısız | Firth (S9); R brglm2 ve R'de bağımsız profil optimizasyonuyla doğrulandı (E10) | **Onay gerekli** |
| K5 | `mlconfound` kısmi testi (`linear`) bağlamsız güçlü skorda yanlış pozitif | Etiket-içi tam permütasyon testi (S10) | **Onay gerekli** |
| K6 | C2'nin C'sinde yaş olacak mı belirsiz (D-025 eğilim skoru yaşı içeriyordu) | Birincil C = saat + tarih (kayıt bağlamı); yaş bir kişi özelliği ve ses yaşı meşru biçimde kodlar → yaşlı sürüm yalnız rapor | S10 ile birlikte onay |
| K7 | N3'ün "etkiyi yeniden üretmiyor" tanımı yok | Öneri Bölüm 3'te; Aşama 2'den önce kesinleşir | Aşama 2'ye kadar açık |
| K8 | N5'te sağlıklıların vekil skoru 25 modelin ortalaması, hastalarınki 5 tekrarın OOF ortalaması (asimetri) | Belgelendi; N5 yalnız G1+ backbone'larda bilgili | Bilinen kısıt |
| K9 | L1 tasarımları (E1h, E2, E3 …) planda sayılmamıştı | E1, E1h, E2, E2h, E3, E3h + CONFOUND 2.3 yöntemiyle referanslar | Çözüldü |
| K10 | Aşama 2 tetikleyicisi (Holm-T1 pozitif), diğer Aşama-1 koşulları sağlanmasa bile çalışır | Değiştirilmedi: N3, C1 düşse bile mekanizma hakkında bilgi verir | Kabul edilen kural |
| K11 | E2h-KLR klasik koşullu logit; tabaka içinde ayrışmada fit hata veriyor → C1 koşul 4 mekanik olarak ✗ (sentetik testte gerçekleşti) | Jeffreys-cezalı koşullu logit (S12) | **Onay gerekli** |
| K12 | Gerçek `participant_context.csv`'de `in_time_overlap` ve `recording_date` sütunları [NEEDS VERIFICATION] | Script assert eder ve açık mesajla durur | Çalıştırmada doğrulanır |
| K13 | N3 "yalnız arka plan" = en düşük enerjili kareler; konuşma kayıtlarında bu kareler **nefes seslerini** de içerir ve astımla fizyolojik olarak ilişkili olabilir → N3'ün etkiyi "yeniden üretmesi" bağlam kanıtı olmayabilir | Aşama 2 tasarımında kare seçimi nefes / ses dışı gürültüyü ayıracak biçimde yazılacak (ör. sessizlik + spektral düzlük ölçütü); belirsizlik raporda | Aşama 2'den önce |
| K14 | Onaylanmamış uygulama ayrıntılarıyla gerçek veride yanlışlıkla çalıştırma riski | `PROTOCOL` sözlüğü (sürüm + ayarlar) rapora yazılır; `status != "KABUL"` iken script gerçek veri kipinde çalışmayı reddeder (testler `--allow-unapproved`) | Çözüldü |
| K15 | N5 tasarım kusuru (S11): gerçek sinyal + bağlam birlikteyken vekil skor astım sinyalini taşır | N5 betimsel; G1 ve C2 bağlam kullanımını göstermeye devam eder; "ince bağlam" sorusu için gün-gruplu CV (Aşama 2) kalır | **Onay gerekli** |

