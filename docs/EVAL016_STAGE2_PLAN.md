# EVAL-016 Aşama 2 — tasarım (N3 ve gün-gruplu CV) ve post hoc duyarlılıklar

**Durum:**
- Bölüm 1–4 (Aşama 2 tasarımı): **GERİ ÇEKİLDİ (D-039 → D-041, 18. tur); hiçbiri uygulanmadı.** Gün-gruplu değerlendirme Faz 3'e taşındı; N3 yapılmıyor. Belge karar geçmişi için korunuyor.
- Bölüm 5 (post hoc duyarlılıklar PH1–PH3): **KABUL (D-040, 2026-10-11)**. Kod: `scripts/posthoc_eval016.py`, `notebooks/23_eval016_posthoc.ipynb`.

**Yazıldığı an:** EVAL-016 Aşama 1 raporu (`reports/context/EVAL-016_context.md`) görüldükten **sonra**, Aşama 2'nin hiçbir sonucu hesaplanmadan **önce**. Bu yüzden belge bütünüyle **"Aşama 1 sonrası"** etiketlidir. Aşama 1 sonucuna bakılarak eklenen her öğe ayrıca işaretlendi (†).

---

## 0. Aşama 1'den bu tasarıma ne taşındı

| Aşama 1 bulgusu | Aşama 2 için anlamı |
|---|---|
| Holm-T1 pozitif: cnn10, cnn14, beats, wavlm_base_plus, wavlm_large | Kabul edilen kural (K10): Aşama 2 bu beşinde yapılır |
| Aşama-1 koşullarının hepsi: cnn10, cnn14, beats, wavlm_large | C1 yalnız bu dördü için kurulabilir; wavlm_base_plus için Aşama 2 yalnız mekanizma bilgisi verir |
| Gün düzeyi kümelenme: CV3 / model SE oranı SSL ve MFCC'de 1.45–1.59, CNN'lerde 1.03–1.04 | Skorlarda ölçülmüş bağlamın açıklamadığı bir gün yapısı var. İki aday mekanizma: (a) ortam (oda, cihaz, seans) → **N3**; (b) eğitimde gün "parmak izi" öğrenilmesi → **gün-gruplu CV** |
| G1 negatif (gömmeler hastalarda AM / PM'yi ayırmıyor); C2 gösterilemedi | Saat yolu zayıf; Aşama 2'nin odağı gün ve ortam yolu |

## 1. Kapsam ve sıra

1. **Kare seçimi ve dinleme denetimi** (CPU; N3'ün sonucu görülmeden).
2. **N3 gömme çıkarımı** (5 backbone; CPU'da yapılabilir, GPU gerekmez [INFERENCE: 339 × 4 s klip]).
3. **Problar** (CPU): N3 (5 backbone + backbone'suz spektral özet) ve gün-gruplu CV (5 backbone + 2 MFCC referansı).
4. **Aşama 2 raporu**, önceden yazılmış kurallarla (Bölüm 4).

Hiçbir adım bir öncekinin sonucuna göre yeniden ayarlanmaz; tek istisna Bölüm 2.3'teki dinleme denetimi kuralıdır (o da N3 sonucu görülmeden).

## 2. N3 — yalnız arka plan (C1 koşul 6)

### 2.1 Neyi sınar
Skorun etiketle ilişkisi, konuşma ve nefes **olmadan**, yalnız kayıt ortamından (oda gürültüsü, cihaz, kanal) üretilebiliyor mu? Üretilebiliyorsa, ŝ'nin bağlam ötesindeki ilişkisi ortam yoluyla açıklanabilir ve C1 kurulamaz.

### 2.2 Kare seçimi (K13 — nefesi dışarıda bırakacak biçimde; sayılar sonuç görülmeden sabitlendi)
- Kaynak: harmonize önbellek (D-030), 16 kHz. Seçilen zaman maskeleri her backbone'un örnekleme hızına (PANNs 32 kHz) aynen taşınır.
- Kare 25 ms, adım 10 ms; kare enerjisi dB (RMS).
- Kayıt başına gürültü tabanı T = kare enerjilerinin 10. persentili.
- **Aday kare:** enerji ≤ T + 3 dB.
- **Koruma bandı:** enerjisi T + 15 dB'yi aşan herhangi bir kareye 150 ms'den yakın kareler atılır. Bu bant konuşma başlangıç ve bitişlerini ve nefesin büyük kısmını dışarıda bırakmak içindir [HYPOTHESIS: denetimle sınanır].
- **Koşu uzunluğu:** ardışık aday kareler ≥ 100 ms olmalı; daha kısa koşular atılır.
- **Katılımcı düzeyinde havuz** (sapma, N3-a): arka plan oda, cihaz ve seans düzeyindedir, görev düzeyinde değildir. Katılımcının 7 kaydındaki koşular görev sırasıyla birleştirilir (5 ms kosinüs geçişlerle).
  - ≥ 4.0 s varsa ilk 4.0 s kullanılır.
  - 2.0–4.0 s arasıysa klip 4.0 s'ye döşenir (tekrarlanır). Döşemenin yarattığı yapay periyodiklik bütün katılımcılarda aynıdır; etiketle ilişkili değildir [INFERENCE].
  - < 2.0 s ise katılımcı N3'ten dışlanır.
  - Dışlananlar ve klip süreleri **etikete göre** raporlanır, çünkü farklı dışlanma kendi başına bir bulgudur.
- Böylece bütün klipler aynı uzunluktadır; uzunluk farkı (ve PANNs'te sıfır dolgu oranı) etikete sızamaz.

### 2.3 Dinleme denetimi (N3 sonucu görülmeden; D-020'nin "oda mı, nefes mi?" kuralı)
- Sabit tohumla 40 katılımcının klibi seçilir: 20 astım, 20 sağlıklı. Klipler karışık sırayla, etiket ve kimlik gösterilmeden Drive'a yazılır (git'e değil).
- Alper her klibi dinleyip işaretler: oda / cihaz gürültüsü · nefes · konuşma artığı · diğer.
- **Kabul ölçütü:** nefes + konuşma artığı ≤ %10 (≤ 4 / 40).
  - Aşılırsa eşikler **bir kez** sıkılaştırılır: koruma bandı 300 ms ve aday eşiği T + 2 dB. Yeni bir 40'lık örneklemle yeniden dinlenir.
  - İkinci denetim de geçmezse N3, bu veride "nefesten ayrılamadı" diye raporlanır. C1 koşul 6 o zaman "değerlendirilemedi" olur ve C1 kurulamaz.
- Etiketler ancak bütün puanlamadan sonra açılır; nefes oranı etikete göre raporlanır.

### 2.4 Temsil ve prob
- **N3-gömme (C1 koşul 6):**
  - Her backbone, 4.0 s'lik arka plan klibinden EXP-016 ile aynı birincil temsili çıkarır (PANNs fc1; SSL katman 1..L ortalaması); tek pencere.
  - Prob EXP-016 ile aynıdır: LR, aynı C ızgarası, aynı dış ve iç fold'lar, 5 tekrar.
  - Fark: 7 görev yerine tek girdi var, füzyon yok (sapma N3-b).
  - ŝ_N3 = 5 tekrarın dış-test logit ortalaması.
- **N3-spektral (D-020; backbone'dan bağımsız, destekleyici):** Aynı klipten 64 bant log-mel ortalaması ve SD'si, spektral düzlük ve düzey (dB) çıkarılır. Prob ve ŝ_N3spec aynı yoldan elde edilir.
- Kohort: T1 kohortu ∩ N3'e uygun katılımcılar. T1 ayrıca bu alt kohortta ŝ ile yeniden hesaplanır, karşılaştırma aynı kişilerde yapılsın diye.

### 2.5 "N3 etkiyi yeniden üretiyor" tanımı (K7)
- **Yeniden üretiyor:** ŝ_N3 ile T1 (aynı bağlam matrisi, Firth PLR) p < 0.05 ve OR > 1. Holm yok, çünkü bu bir çürütme kontrolü ve katı tutuluyor.
- **C1 koşul 6 sağlanır:** yeniden üretmiyor **ve** dinleme denetimi geçti.
- **Yeniden üretiyorsa:** D-020'ye göre otomatik olarak "sızıntı" denmez. C1 "kurulamadı — ortam yolu açıklayabilir" yazılır.
- **Betimsel:** ŝ ve ŝ_N3 birlikte T1'de; ŝ'nin ŝ_N3'e göre ayarlı PLR'si ve OR'u.

## 3. Gün-gruplu CV — gün "parmak izi" kontrolü

### 3.1 Neyi sınar †
- 65 kayıt gününün 53'ü tek etiketli. Katılımcı düzeyinde CV'de bir test katılımcısının aynı gün kaydedilmiş "gün eşleri" eğitimdedir ve neredeyse her zaman aynı etiketi taşır.
- Prob güne özgü akustik izi öğrenirse, test katılımcısını bu izden tanır. T1'in tarih spline'ı bunu ayarlayamaz; bu SIM-001 S3'ün eğitim yoluyla oluşan biçimidir.
- CV3 bu yapıyı **çıkarımda** hesaba katar. Gün-gruplu CV ise onu **kaynağında** kapatır.

### 3.2 Uygulama
- **Yeni split dosyaları:** `splits_day/outer_r{r}.csv`, 5 tekrar, aynı sütun biçimi. Hem dış hem iç fold'lar `recording_date`'e göre gruplu, etikete göre tabakalı (StratifiedGroupKFold, sabit tohumlar).
  - Bir günün bütün katılımcıları aynı role düşer (assert).
  - Gün ve etiket dağılımı split manifestinde raporlanır.
- **Prob:** EXP-016'nın aynısı (`run_probes.py`, yalnız split klasörü farklı); ayrı çıktı klasörü.
- **Kollar:** 5 backbone + mfcc_lr + mfcc_mlp (referans).
- **Analiz:**
  - ŝ_DG → T1 (aynı bağlam matrisi, Firth PLR), bootstrap OR, CV3.
  - Ayrıca katılımcı düzeyi ŝ ile ŝ_DG'nin T1 OR'larının farkı: eşleştirilmiş katılımcı bootstrap'ı, betimsel.

### 3.3 Karar
- **Öneri †** — C1'e kısıtlayıcı bir koşul eklemek:
  - **Koşul 8:** ŝ_DG ile T1-PLR p < 0.05 ve OR > 1. Holm yok, çünkü backbone Aşama 1'de zaten Holm'u geçti.
  - Aşama 1 sonucundan sonra önerildiği için açıkça işaretlenir. C1'i yalnız zorlaştırabilir.
- **Alternatif:** gün-gruplu CV betimsel kalır ve C1 kuralı değişmez. O durumda "C1 sağlandı" ifadesinin yanına gün-gruplu sonuç zorunlu olarak yazılır.

## 4. Aşama 2 kararları (önceden yazıldı)

| backbone | Aşama 1 | Aşama 2 sonunda C1 |
|---|---|---|
| cnn10, cnn14, beats, wavlm_large | hepsi ✓ | koşul 6 ✓ (ve önerilen koşul 8 ✓) → **C1 sağlandı**; aksi hâlde "kurulamadı" + hangi koşul |
| wavlm_base_plus | koşul 3 ✗ | C1 sağlanmadı (değişmez); N3 ve gün-gruplu CV mekanizma için raporlanır |

"C1 sağlandı" ifadesinin anlamı ve sınırı: o backbone'un skoru, **ölçülen saat ve tarih, yaş, gün düzeyi kümelenme, ortam (arka plan) ve gün parmak izi yollarıyla açıklanamayan** bir etiket ilişkisi taşıyor. Bu ifade şunları **içermez**:
- prosedür yolunun (bronkodilatör, S13) dışlanması;
- klinik ayırt edicilik;
- dış genelleme.

## 5. Post hoc duyarlılıklar (D-040; KABUL, 2026-10-11) †

Aşama 1 sonucundan sonra önerildi. **C1 / C2 kararlarını değiştirmez.** Tanımlar çalıştırmadan önce sabitlendi.

| # | Ne | Neden |
|---|---|---|
| PH0 | Kaydedilmiş ŝ'lerle birincil T1'i yeniden hesapla; Aşama 1 raporuyla \|Δ LR\| < 1e-4 olmalı (değilse durur) | Girdi bütünlüğü |
| PH1 | T1 bağlamında yaş doğrusal yerine cr(yaş, 4) | Hastalar ortalama 7.6 yaş büyük; skorlar etiket içinde yaşı güçlü kodluyor (L3 yaşlı p < 0.001). Doğrusal ayar kalıntı bırakabilir |
| PH2 | T1 bağlamı + cinsiyet + sigara (şu an / bırakmış / hiç) | RQ9 (D-006). Cinsiyet dengeli; sigara sağlıklılarda daha sık |
| PH3 | E2h cezalı koşullu logitin aynı amaç fonksiyonuyla yeniden fit'i: 3 başlangıç (0, sabit tohumlu rastgele, sınırlıysa klasik ML), BFGS + Nelder–Mead; yakınsama = yerel minimum denetimi + en az iki başlangıcın uyuşması | K19: 3 kolda "yakınsamadı" bayrağı. Yöntem R `survival` referansıyla doğrulandı (E18) |

Her PH'de raporlananlar: Firth PLR p, Holm (6, yalnız karşılaştırma için), SD başına OR, CV3 p.

**Yorum kuralı:**
- Aşama-1 koşullarını sağlayan bir backbone, PH1 ya da PH2'de Holm p ≥ 0.05 ya da CV3 p ≥ 0.05 çıkarsa **"yaş / demografi ayarına duyarlı"** diye işaretlenir. Bu işaret o backbone'un bütün yorumlarında yazılır; C1 durumu değişmez.
- PH3'te OR'un yönü değişirse K19 "karar etkilendi" diye yeniden açılır ve bir karar gerekir.

## 6. Maliyet ve yeniden üretilebilirlik

- **Kare seçimi:** 2393 kayıt, CPU, dakikalar. **N3 çıkarımı:** 339 klip × 5 backbone, CPU'da dakikalar [INFERENCE].
- **Problar:** N3 hafif (tek girdi). Gün-gruplu CV, EXP-016 kadar CPU ister (7 kol × 5 tekrar; saatler).
- **Kontrol noktaları:** problar `run_probes.py`'nin parça kayıtlarını (kol × tekrar) kullanır. Bağlantı koparsa kaldığı yerden devam eder.
- **Kod ve test:** her script sentetik, doğrusu bilinen testlerle (yerleştirilmiş arka plan etkisi, yerleştirilmiş gün parmak izi) ve simüle Colab'da sınanır. Gerçek veride çalıştırma kapısı (protokol KABUL, sabit ayarlar) EVAL-016 Aşama 1'deki gibi.
- **GPU:** gerekmez [INFERENCE]. CPU süresi 3 saati aşacaksa önce bildiririm.
