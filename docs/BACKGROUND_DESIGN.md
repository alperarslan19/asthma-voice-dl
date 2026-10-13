# Arka plan (ortam sesi) — Adım 0 tasarımı (EXP-032; D-046)

**Durum:** ÖNERİLDİ (2026-10-13). Yaklaşım Alper tarafından onaylandı ("onaylıyorum", 13 Ekim). Parametreler 0b'nin
kalite kapısından sonra, 0c'den ÖNCE dondurulur (D-046 → KABUL).

**Bağlam:** Kayıt bağlamı (gün, dönem, saat) etiketi neredeyse belirliyor (yalnız bağlam AUC 0.93–0.94). Açık kalan
yollardan biri **ortam** (G2): kayıt odasının akustik arka planı. Alper'in önerisi: kendi sesine astımlı / sağlıklı
kayıtlardan alınmış arka planlar ekleyip modellerin skorunun kayıp kaymadığına bakmak (arka plan değiştirme testi). Bu
belge o testin **ön adımını** (Adım 0) tanımlar. Değiştirme testi (Adım 1–5) ayrı bir tasarımla açılacak.

---

## 1. Sorular

| Adım | Soru | Astım etiketine bakar mı? |
|---|---|---|
| 0a | Kayıtlardan arka plan çıkarılabiliyor mu? Kişi başına kaç saniye; gruplara göre dengeli mi? | Yalnız sayım (envanter dengesi) |
| 0b | Çıkan şey gerçekten arka plan mı (konuşma sızıntısı yok mu)? | Hayır (cinsiyet ve dinleme) |
| 0c | Arka plan tek başına etiketi / dönemi ayırıyor mu? | Evet — yalnız D-046 KABUL'den sonra |

0c'nin cevabı değiştirme testinin anlamını belirler: arka planda etiket bilgisi yoksa modellerin onu kullanması
beklenmez; varsa değiştirme testi "kullanıyorlar mı?" sorusunu sorar.

---

## 2. Veri ve varsayımlar

- Girdi: 2 393 kayıt / 342 katılımcı (önbellek kohortu); kayıt haritası; önbellek dizini (kırpma sınırları, kazanç);
  `participant_context.csv` (etiket, cinsiyet, kayıt tarihi, saat, yaş); gün-gruplu split'ler (`splits_day`, 3 × 5).
- Kenar sessizliği çok az: baştan kırpılan medyan 0.05 s, sondan 0 s [FACT, `reports/audio_cache`]. Arka plan çoğunlukla
  kelime tekrarları arasındaki iç duraklamalardan gelir. "aaa"da neredeyse yok.
- Bir katılımcının 7 kaydı aynı oturumda, aynı odada yapıldı [INFERENCE: aynı gün tarihli] → arka plan katılımcı
  düzeyinde 7 kayıttan birleştirilir.
- **Model görünümü** [DECISION]: her şey önbellekle aynı zincirden geçer (48 kHz → 32 kHz + 11 kHz alçak geçiren →
  önbellekteki tepe kazancı) ve yalnız önbelleğin kırpma sınırları içinde alınır. Böylece arka plan, modellerin
  duraklamalarda gördüğü şeyin aynısıdır. Kazanç uyumu her kayıtta kontrol edilir.

---

## 3. Üç arka plan türü

**KARAR:** Birincil A1, ikincil A2 ve B.
**NEDEN:** A1 en tutucu "ortam" tanımı. A2 − A1 farkı nefes / oda olaylarının katkısını, B ise yalnız durağan "oda
renginin" yetip yetmediğini gösterir.
**ALTERNATİFLER:** Sinir ağıyla ayırma (DeepFilterNet artığı) — konuşma ve model artefaktı sızma riski; yalnız A'nın
envanteri yetersiz kalırsa. Elle (Audition) döngüleme — ölçeklenmez, etiketlere kör değil; ilk 10 bağışçı için betikle
karşılaştırma amaçlı kullanılabilir.
**RİSK:** Aşağıdaki tablo ve bölüm 6.
**BİLİMSEL SONUÇ:** Bilgi bulunursa hangi bileşenden geldiği ayrılabilir.
**UYGULAMA:** `scripts/background_extract.py`.

| Tür | Tanım | Parametre [DECISION] |
|---|---|---|
| **A1 oda tonu** | Enerjiyle etkin HER kareden ≥ 150 ms uzak VE taban + 3 dB'nin altı | etkin kare: ses denetimi / önbellek kuralı (tepe − 35 dB; SNR > 20 dB ise en az taban + 10 dB); taban = kare dB p10; parça ≥ 60 ms; katılımcı toplamı ≥ 2.0 s |
| **A2 tüm sessizlik** | SESLİ (periyodik) konuşmadan ≥ 150 ms uzak bütün kareler — nefes ve oda olayları dahil | sesli kare = etkin VE özilinti tepe ≥ 0.45 (F0 60–400 Hz); ünsüzler sesli karelerin 150 ms yakınında kalır; parça ≥ 100 ms; toplam ≥ 3.0 s |
| **B durağan** | Her geçerli kayıtta (100 Hz–8 kHz log-ortalamaya göre) en sessiz %10 STFT karesi (64 ms) → ortalama güç spektrumu; kayıtların eşit ağırlıklı ortalaması; bu spektrumla gürültü sentezi | Kısa aralara da sığar; ani olayları ve dokuyu taşımaz |

Ortak kurallar:
- SNR (p95 − p10) ≤ 20 dB olan kayıtlardan A1 / A2 / B alınmaz: orada "taban" konuşmanın kendisi olabilir (ör. "aaa").
- Sayısal sıfır kareleri (< −100 dBFS; kodlayıcı dolgusu) tabana ve arka plana girmez.
- Kırpma sınırlarının içinde 50 ms pay.
- İz: parçalar rastgele sırayla (katılımcı ve türe bağlı sabit tohum), 10 ms eşit güçlü çapraz geçişle uç uca, 10 s'ye
  döngülenir. Seviye korunur (ilişkisiz gürültüde cos² + sin² = 1).

Denenip bırakılan (olumsuz sonuç, korunur): B için frekans başına kantil tahmini (minimum istatistik ailesi; Martin 2001
fikri). Sentetik testte %60 konuşma doluluğunda +3–5 dB sapma verdi: kantil, gürültü-yalnız karelerin payı < %100
olunca yukarı kayıyor. Özyineli yumuşatma (α 0.8) güçlü konuşmanın kuyruğunu 0.4 s'lik aralara taşıdı. Yerine "en
sessiz kareler" seçildi (sapma ≤ 0.5 dB; test U4).

---

## 4. Özellikler

Her katılımcı × tür için 10 s'lik iz:
- **ltas_abs:** 1/3 oktav bant düzeyleri, 125 Hz – 8 kHz (19 bant), model görünümü ölçeğinde dB → seviye + renk.
- **ltas_shape:** aynı bantlar, satır ortalaması çıkarılmış → yalnız renk.
- **mfcc44:** projedeki MFCC-44 tarifi (cache modu; 22.05 kHz).

Derin gömmeler (PANNs / SSL) bu adımda yok: GPU ister. 0c sonucu gerektirirse eklenir.

---

## 5. Kalite kapıları (0b) ve 0c

| Kapı | Ölçüt | Eşik [DECISION] | Kalırsa |
|---|---|---|---|
| G-a envanter dengesi | Dahil olma oranı farkı, sağlıklı vs astım | ≤ 10 puan (yalnız bayrak) | Raporlanır; 0c'de duyarlılık |
| **G-b konuşma sızıntısı** | Arka plandan cinsiyet AUC (gün-gruplu, her özellik seti) | ≤ max(0.60, bağlam referansı + 0.05) | Koruma payı / eşikler ayarlanır, 0a–0b yeniden |
| G-c tutarlılık | Katılımcı başına medyan \|A1 − B\| (bantlar) | Genel medyan ≤ 3 dB | B veya A1 tanımı incelenir |
| G-d kör dinleme | "Konuşma kalıntısı ≥ 1" oranı | A1 ≤ %10, A2 ≤ %20, B ≤ %10 | Koruma payı ayarlanır |

- Pozitif kontrol: konuşma MFCC'sinden cinsiyet yüksek AUC vermeli; vermezse G-b yorumlanmaz.
- Cinsiyetin kayıt günüyle zayıf ilişkisi olabilir; bağlam referansı (saat + tarih) bu yüzden eşikte.
- Parametre değişiklikleri yalnız 0a–0b sırasında, gerekçesiyle kaydedilir. Sonra **dondurulur** (D-046 KABUL). 0c
  sonuçlarına göre parametre değiştirilmez.

**0c (bilgi testi):**
- **Birincil:** A1 | ltas_abs → astım / sağlıklı. Gün-gruplu CV havuz AUC (3 tekrar ortalaması) ve tekrar-0'ın
  katılımcı bootstrap %95 GA'sı.
- **Okuma (önceden yazılı):** GA'nın alt sınırı > 0.5 → "arka planda etiket bilgisi var"; AUC ≥ 0.70 → "güçlü". Bağlam
  referansıyla (0.93–0.94) betimsel karşılaştırma.
- **İkincil (betimsel):** diğer 8 kol; astımlılarda dönem (erken / geç); A1 skorunun bağlam ötesi ilişkisi (T1).
  Çoklu karşılaştırma düzeltmesi yok; ikinciller hipotez üretir.

| 0c sonucu | Değiştirme testi için anlamı |
|---|---|
| A1'de bilgi var | Durağan oda sesi bir kısayol olabilir → değiştirme testinin A1 kolu öncelikli |
| Yalnız A2'de | Nefes / oda olayları; hastaya ait sinyal olabilir → yorum dikkatli |
| Yalnız dönemde, etikette zayıf | Ortam zamanla değişmiş; etiketle ilişkisi dönem üzerinden |
| Hiçbirinde | Eklenen arka plan yolu zayıf; bağlam etkisi başka yollardan (yankı, kanal, SFT günü, konuşma) |

---

## 6. Riskler

- **Nefes / konuşma ayrımı:** A2'de nefes var (amaçlı); hastalık sinyali olabilir. A1 bunu dışlar.
- **Seçim yanlılığı:** Düşük SNR veya kısa sessizlik nedeniyle dışarıda kalanlar gruba / döneme göre raporlanır (G-a).
- **Döngü yapısı:** Az parçalı katılımcıda iz tekrar eder; dinlemede "döngü hissi" puanlanır. Özellikler zaman
  ortalaması olduğu için etkisi küçük [INFERENCE].
- **Yankı / kanal yolu:** Bu adım yalnız eklenen arka planı ölçer; odanın yankısı ve mikrofon tepkisi sesin içindedir,
  burada sınanmaz.
- **Gizlilik:** Arka plan izleri katılımcı düzeyi veridir (Drive). Raporlar agrega; ID taraması otomatik.

---

## 7. Uygulama ve çalıştırma

- Kod: `scripts/background_extract.py` (`extract`, `probe`, `listening`).
- Test: `tests/test_background.py`. Birim testleri: maskeler, sayısal sessizlik, düşük SNR, B sapması, sentez, iz
  oluşturma, gün-gruplu CV. Uçtan uca: sahte veriyle extract → probe sex → 0c kapısı (ÖNERİLDİ reddedilir, KABUL çalışır).
- Notebook: `notebooks/27_background_step0.ipynb` (CPU). Bölüm 0–4: 0a + 0b; kör dinleme; Bölüm 5: 0c.
- Çıktılar: Drive `experiments/background_v1/` (katılımcı düzeyi); repo `reports/background/BG-0a…0c` (agrega).

Kaynaklar: Martin 2001, IEEE TSAP 9(5):504–512 (minimum istatistik; burada denenip bırakıldı) · Gerkmann & Hendriks
2012, IEEE TASLP (MMSE tabanlı gürültü gücü tahmini; alternatif olarak not edildi) · McDermott & Simoncelli 2011,
Neuron (doku sentezi; gerekli görülmedi).
