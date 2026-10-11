# EVAL-016 — Dondurulmuş aile için sonuç bildirimi (sabitlendi)

**Durum:** D-041 ile önerildi (2026-10-11, 18. tur). Kaynaklar:
- `reports/context/EVAL-016_context.md` (Aşama 1, önceden belirlenmiş);
- `reports/context/EVAL-016_posthoc.md` (PH1–PH3, post hoc);
- `reports/context/SIM-002_k2_validity.md`.

Bu belge **yeni analiz içermez**. Elimizdeki sonuçların neyi söylediğini, neyi söylemediğini ve sonucu değiştirebilecek eksikleri sabitler.

---

## 1. Desteklenen iddia

> Dondurulmuş, önceden eğitilmiş ses temsillerinden üç tanesinden (PANNs CNN10, BEATs, WavLM Large) elde edilen katılımcı skoru, bu kohortta **ölçülen kayıt saati, kayıt tarihi ve yaşın ötesinde** astım / sağlıklı etiketiyle ilişkilidir.
>
> Bu ilişki şu ayarlardan sonra da anlamlı kalıyor:
> - gün düzeyi kümelenme için geçerli bir çıkarım (gün-kümeli jackknife);
> - yaşın esnek ayarı;
> - cinsiyet ve sigara;
> - zaman penceresi içinde aynı saat dilimindeki karşılaştırma (E2h).
>
> Etki büyüklüğü ŝ'nin 1 SD'si başına OR ≈ 3–5, aralıklar geniş. Bu kohortta bağlam tek başına AUC ≈ 0.93 verdiği için ayırt edicilikteki artış ölçülemeyecek kadar küçük (ΔAUC +0.006–0.009 ± 0.010).

| | cnn10 | beats | wavlm_large | cnn14 | wavlm_base_plus | cnn14_16k | MFCC (referans) |
|---|---|---|---|---|---|---|---|
| T1 (Holm) | 0.002 | <0.001 | <0.001 | 0.012 | 0.006 | 0.145 | ham 0.009 / 0.022 |
| Gün-kümeli CV3 | 0.003 | 0.020 | 0.017 | 0.015 | **0.068** | 0.192 | **0.10 / 0.15** |
| Aşama-1 koşullarının hepsi | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | — |
| PH1 / PH2 (post hoc) | dayanıklı | dayanıklı (yaş spline'da CV3 0.045, sınırda) | dayanıklı | **duyarlı** (yaş spline'da CV3 0.067) | — | — | CV3 > 0.05 |
| **İddiadaki yeri** | **evet** | **evet** | **evet** | sınırda: yalnız birincil analizde | hayır | hayır | hayır (yalnız T1) |

[FACT: önceden belirlenmiş analiz + işaretli post hoc duyarlılık]

## 2. Desteklenmeyen iddialar

1. **"Ses astımı tanıyor" / klinik ayırt edicilik.**
   - Vaka-kontrol tasarımı, tek merkez, sağlıklı n = 57 (sabah 12), geç dönemde hiç sağlıklı yok, dış doğrulama yok.
   - Artımlı AUC sıfırdan ayırt edilemiyor.
2. **"Sinyal bağlamdan tamamen bağımsız" (protokoldeki C1).**
   - C1 önceden yazılmış kurala göre **kurulmadı**: N3 (arka plan) yapılmadı.
   - Ölçülmemiş ortam ve prosedür yolları dışlanmadı (Bölüm 3).
3. **"Bir backbone diğerinden iyi."** Bu analiz backbone'ları karşılaştırmak için tasarlanmadı. Örneğin beats'in OR'u en yüksek ama gün kümelenmesine de en duyarlı olanı (SE oranı 1.59).
4. **"Model bağlamı kullanmıyor."**
   - C2 hiçbir backbone'da **gösterilemedi**; bu, "yok" demek değil.
   - Betimsel işaretler var: sağlıklılarda sabah skoru daha yüksek (beats'te CI 0'ı dışlıyor, n = 12); SSL ve MFCC skorlarında gün kümelenmesi var.
5. **Fine-tune modellerine genelleme.** Bütün bu sonuçlar dondurulmuş problar içindir (C3 ayrı değerlendirilir).

## 3. Sonucu değiştirebilecek eksikler — ve ne yapacağımız

| # | Eksik | Sonucu değiştirebilir mi? | Karar (D-041) |
|---|---|---|---|
| G1 | **Eğitimde gün "parmak izi".** 65 günün 53'ü tek etiketli; katılımcı düzeyi CV'de test katılımcısının gün eşleri eğitimde | **Evet.** Sinyal gün-gruplu CV'de kaybolursa iddia "gün / seans artefaktı olabilir"e döner. Fine-tune modelleri bu yola daha açık [INFERENCE] | Ayrı bir Aşama 2 olarak yapılmaz. **Faz 3'ün değerlendirme tasarımına taşınır:** dondurulmuş problar, Faz 3'ün baz çizgisi olarak gün-gruplu split'lerde yeniden koşulur. Sonuç, Faz 3 ile birlikte bu belgeye eklenir |
| G2 | **Ortam / arka plan yolu (N3).** Gün düzeyindeki ortam G1 ile büyük ölçüde örtüşüyor; katılımcı düzeyindeki ortam farkı örtüşmüyor | Kısmen. Ama N3, nefes belirsizliği (K13) yüzünden kendisi de kesin cevap vermiyor | **Yapılmaz.** Sınırlılık olarak yazılır. C1 bu yüzden "kurulmadı" kalır |
| G3 | **Prosedür yolu** (bronkodilatör / spirometri sonrası kayıt, S13) | Evet, ama veriyle değil, bilgiyle | Analiz yok; veri ekibinin cevabı beklenir. Cevap gelene kadar her sunumda sınırlılık olarak yazılır |
| G4 | Ölçülmemiş diğer karıştırıcılar (cihaz değişimi, operatör, hastalık şiddeti ile yaşın etkileşimi …) | Bilinmiyor | Geriye dönük çözülemez; sınırlılık |

**Durma kuralı:** Dondurulmuş aile için başka bağlam analizi eklenmez. Bu belge yalnız G1'in sonucu ve G3'ün cevabı geldiğinde güncellenir.

## 4. Sunumda kullanılacak cümle (taslak)

> Önceden kaydedilmiş bir analiz planında, dondurulmuş önceden eğitilmiş ses temsillerinden üçünün katılımcı skoru, ölçülen kayıt saati, tarihi ve yaşın ötesinde astım etiketiyle ilişkili bulundu (Holm düzeltmeli p ≤ 0.002; gün-kümeli jackknife p ≤ 0.02). Bu ilişki, yaşın esnek ayarına ve cinsiyet ile sigaraya karşı dayanıklıydı (post hoc). Ancak kohortta kayıt bağlamı etiketi neredeyse belirlediğinden (bağlam AUC 0.93) ayırt edicilikteki artış ölçülemedi. Arka plan / ortam yolu, prosedür yolu ve dış geçerlilik dışlanmadı. Bulgu klinik bir test olarak yorumlanmamalıdır.
