# Colab + Drive + GitHub çalışma düzeni

Bu belge "nerede ne durur, her oturum nasıl başlar, kod nasıl kaydedilir, eğitim yarıda kalırsa ne olur" sorularının cevabıdır. Bir kez okunur, sonra kontrol listesi olarak kullanılır.

---

## 1. Büyük resim: üç yer, üç görev

```
GitHub (private repo)            Google Drive                         Colab (geçici makine)
─────────────────────            ─────────────────────────            ──────────────────────────
KOD ve KARARLAR                  VERİ ve ÇIKTILAR                     ÇALIŞTIRMA
 scripts/, src/, configs/         data_raw/   (salt okunur)            /content/asthma-voice-dl  ← repo klonu
 notebooks/                       data_derived/                        /content/audio/           ← zip'ten açılmış sesler
 DECISIONS.md, EXPERIMENTS.md     models/     (indirilen ağırlıklar)   GPU (T4)
 reports/ (yalnız agrega)         experiments/EXP-XXX/                 Oturum kapanınca HER ŞEY silinir
 results/registry.csv
```

**Altın kural:** Colab makinesi geçicidir. Saklanması gereken her şey ya **git'e** (kod, küçük metin) ya **Drive'a** (veri, model, log, checkpoint) yazılır.

**Neden repo Drive'ın içinde değil?** Drive, Colab'a bir ağ diski gibi bağlanır; git'in binlerce küçük dosya işlemi orada yavaş ve bozulmaya açıktır. Repo her oturumda `/content`'e klonlanır (birkaç saniye).

**Neden ses dosyaları zip?** Drive'dan binlerce küçük dosyayı tek tek okumak çok yavaştır. Tek bir zip'i yerel diske kopyalayıp açmak hızlıdır ve zip'in sha256'sı "hangi veriyle çalıştım?" sorusunun cevabıdır.

---

## 2. Bir kerelik kurulum

### 2.1 GitHub
1. github.com → **New repository** → ad: `asthma-voice-dl` → **Private** → Create.
2. Bu iskeleti yükle: repo sayfasında **Add file → Upload files**, açtığın zip'in *içindeki* dosya ve klasörleri sürükle → **Commit changes**.
3. Colab'ın repo'ya yazabilmesi için bir anahtar (token) oluştur: GitHub → Settings → Developer settings → **Fine-grained personal access tokens** → Generate new token → *Repository access: Only select repositories → asthma-voice-dl* → *Permissions → Contents: Read and write* → Generate. Çıkan metni kopyala (bir daha gösterilmez).
4. Colab'da sol menüdeki **anahtar simgesi (Secrets)** → *Add new secret* → Name: `GITHUB_TOKEN`, Value: token → *Notebook access* anahtarını aç. Token asla notebook hücresine yazılmaz.

### 2.2 Drive klasörleri
`MyDrive/asthma-voice/` altında:
```
data_raw/          soundData.zip, clinical_data.csv, astim-tarama_Data Report_20260314.xlsx
                   → yükledikten sonra BİR DAHA DEĞİŞTİRME (yeni dışa aktarım gelirse yeni klasör: data_raw_v2/)
data_derived/      participants.csv, audio_inventory.csv, recording_map.csv, participant_context.csv,
                   splits/ (outer_r0…4.csv), audio_cache_v1/ (audio_32k.f32, audio_16k.f32, index.csv, cache_info.json)
                   → önbellek bir kez üretilir; eğitim oturumlarında Drive'dan YEREL diske kopyalanıp np.memmap ile okunur
models/            indirilen pretrained ağırlıklar (+ sha256)
experiments/       EXP-XXX/ klasörleri
```

### 2.3 Ses klasörünü zip'le ve yükle
- Dosyalar `101001_1.m4a` … `101344_7.m4a` biçiminde (veri ekibi; 344 × 7 = 2 408 dosya). Yeniden adlandırma ya da format dönüştürme **yapma**: denetim ham dosyaları görmeli.
- Ham dosyalar iPhone'un GPS konum etiketini taşıyabilir → kimseyle paylaşma; Drive klasörünü paylaşıma açma.
- Mac: `soundData` klasörüne sağ tık → *Compress*. Windows: sağ tık → *Send to → Compressed (zipped) folder*.
- `soundData.zip`'i `data_raw/`'a yükle. Notebook ilk çalıştığında zip'in sha256'sını hesaplayıp yazar; bu değer veri sürümünün parçasıdır.

---

## 3. Her oturumun başı (her notebook'un ilk hücreleri)

Her notebook aynı "oturum kurulumu" bloğuyla başlar (bkz. `notebooks/01_data_audit.ipynb`):
1. **GPU / ortam kontrolü** — hangi GPU, Python ve kütüphane sürümleri.
2. **Drive bağla** — `drive.mount('/content/drive')`.
3. **Repo'yu klonla ya da güncelle** — token Secrets'tan okunur.
4. **Paketleri kur** — `requirements-colab.txt`.
5. **Veriyi yerel diske kopyala ve hash'ini doğrula.**

Colab'ın "Runtime → Restart session" sonrası 1–5 yeniden çalıştırılır.

---

## 4. Notebook düzeni

**Kural:** Mantık `.py` dosyalarında (`scripts/`, ileride `src/`), notebook'lar yalnızca "kur → çalıştır → göster". Neden: `.py` dosyaları git'te okunabilir fark (diff) verir, test edilebilir ve aynı kod her notebook'ta aynıdır.

| Dosya | Amaç | Faz |
|---|---|---|
| `01_data_audit.ipynb` | Klinik + ses denetimi, eşleme, dinleme kontrolü | 0 |
| `02_splits.ipynb` | Fold dosyalarını üret, kesişim = 0 kontrolleri | 1 |
| `03_audio_cache.ipynb` | 16 / 32 kHz deterministik önbellek | 1 |
| `10_mfcc_baseline.ipynb` | MFCC yeniden üretim + ortak protokol | 1 |
| `20_smoke_tests.ipynb` | Her backbone için yükleme, şekil, 1 adım eğitim, GPU bellek | 2 |
| `21_frozen_embeddings.ipynb` | Gömme çıkarımı + lineer prob | 2 |
| `30_finetune.ipynb` | Fine-tune (checkpoint/resume ile) | 3 |
| `90_analysis.ipynb` | Karşılaştırmalar, confounder ve alt grup analizleri | 3 |

Adlandırma: iki haneli sıra numarası + kısa İngilizce ad. Deney klasörleri `EXP-NNN_kisa-ad` (ör. `EXP-004_frozen-probe`).

---

## 5. Git ile çalışma (Colab'dan)

```bash
cd /content/asthma-voice-dl
git config user.name  "Alper"
git config user.email "<GitHub e-postan>"
git status                      # neyin değiştiğine HER ZAMAN önce bak
git add scripts/ reports/ DECISIONS.md EXPERIMENTS.md results/registry.csv
git commit -m "audit: ses denetimi ve eşleme raporu"
git push
```

**Commit öncesi kontrol listesi**
- [ ] `git status`'ta `.wav/.m4a/.csv(katılımcı düzeyi)/.xlsx/.pt` yok (`.gitignore` bunları dışlar ama yine de bak).
- [ ] Notebook çıktıları temizlendi (*Edit → Clear all outputs*) — çıktılarda katılımcı satırları olabilir.
- [ ] Commit mesajı deney ID'sini içeriyor (`EXP-004: ...`) ya da türünü (`audit:`, `docs:`, `fix:`).
- [ ] Deneyin `git_commit` alanı registry'ye yazıldı (deney kodu çalışmadan **önce** commit et ki sonuç bir commit'e bağlansın).

---

## 6. Kütüphane kararları

| Paket | Ne için | Sürüm politikası |
|---|---|---|
| numpy, pandas, scipy, scikit-learn, matplotlib | Temel | Colab'dakiler; her deney `pip freeze`'i kaydeder |
| ffmpeg / ffprobe (sistem) | Her ses formatını çözme ve inceleme | Colab'da kurulu |
| soundfile, openpyxl, tabulate | WAV okuma/yazma, XLSX, md tablolar | `requirements-colab.txt` |
| torch, torchaudio | Derin öğrenme, resample | **Colab'dakini kullan** — yeniden kurmak CUDA uyumsuzluğu yaratabilir |
| torchlibrosa | PANNs'in STFT/log-Mel katmanları | Faz 2'de eklenecek [NEEDS VERIFICATION: Colab numpy sürümüyle uyum] |
| PANNs model kodu | CNN10 / CNN14 sınıfları | Resmi repodan `third_party/panns/`'a kopyalanır, kaynak commit hash'i not edilir |
| BEATs model kodu | BEATs | Resmi repodan `third_party/beats/`'e kopyalanır |
| transformers | WavLM | Faz 2'de; sürüm kaydedilir |
| librosa, imbalanced-learn, xgboost, catboost | MFCC baseline yeniden üretimi (makale: librosa 0.10.1, sklearn 1.5.0, imblearn 0.12.0) | Faz 1'de; birebir sürüm Colab numpy'sıyla çakışırsa farkı not et |
| statsmodels | İstatistiksel testler | Faz 1'de |

Paketleri faz faz ekliyoruz: kullanmadığımız bir kütüphaneyi bugünden kurmak, kurulum hatalarıyla odak kaybı demek.

---

## 7. Eğitim fazları için checkpoint ve devam kuralları (şimdiden sabit)

**Her deney klasörü** (`experiments/EXP-NNN_ad/`):
```
config.yaml            tüm ayarlar (model, SR, split dosyası+hash, hiperparametreler, seed)
env.txt                pip freeze + GPU adı + git commit
fold_r{r}_k{k}/
    train_log.csv      epoch, step, train_loss, val_loss, val_auc, val_balacc, lr, elapsed_s  (her satır flush)
    latest.pt          model + optimizer + scheduler + scaler + RNG durumları + epoch + en iyi skor
    best.pt            en iyi val skorundaki model ağırlıkları
    oof_predictions_best.csv  katılımcı, kayıt, segment düzeyinde dış-test tahminleri (en iyi val checkpoint'i)
    oof_predictions_last.csv  aynısı, son epoch'un modeliyle (fine-tune ve sıfırdan eğitimde; D-028)
    DONE               fold bitince yazılan boş dosya
```

**Kurallar**
1. `latest.pt` her epoch sonunda **atomik** yazılır: önce `latest.pt.tmp`, sonra yeniden adlandırma. Yazım sırasında oturum koparsa eski dosya sağlam kalır.
2. Script başlarken `DONE` olan fold'ları atlar; `latest.pt` olan fold'u kaldığı yerden sürdürür. Yani "Run all" yeniden basmak güvenlidir.
3. Log, epoch başına bir satır ve hemen `flush` — oturum koparsa kayıp en fazla bir epoch.
4. **Drive kotası:** ücretsiz 15 GB. CNN14'ün tam durumu (ağırlık + Adam) ≈ 1 GB. Fold bittiğinde `latest.pt` silinir, `best.pt` fp16 olarak tutulur (~160 MB), tahminler (best ve last) her zaman tutulur. Son epoch tahminleri, `latest.pt` silinmeden **önce** yazılır. WavLM Large için daha da önemli. [INFERENCE — gerçek boyutlar smoke test'te ölçülecek]
5. **Smoke test olmadan tam eğitim yok:** checkpoint yükleme (`strict=True`), 1 batch ileri geçiş + şekil assert'leri, 1 eğitim adımı (kayıp sonlu mu?), 1 validation adımı, GPU bellek tepe değeri, kaydet → yeniden yükle → aynı çıktı.

---

## 8. Günlük rutin

**Deneyden önce:** EXPERIMENTS.md'ye soru + tasarım + beklenti → config yaz → commit → registry'de satır (`status=running`).
**Deneyden sonra:** sonuç tablosu + yorum + kısıtlar (olumsuz sonuçlar dahil) → registry satırını güncelle → yeni bir karar alındıysa DECISIONS.md → commit.
**Haftalık:** registry.csv'yi Google Sheets'e içe aktar (*File → Import → Upload → Replace current sheet*). Sheets'te düzenleme yapılmaz; tek kaynak CSV.

---

## 9. Sık yapılan hatalar

- `df.take`, `df.count`, `df.size` gibi adlar pandas'ta **metottur**: sütuna `df["take"]` diye eriş (ses denetimi script'inde gerçekten yakaladığımız bir hata).
- Colab "Restart session" → değişkenler gider ama `/content` dosyaları kalır; "Disconnect and delete runtime" → hepsi gider.
- Drive'a yazılan dosyalar bazen birkaç saniye gecikmeyle görünür; kritik yazımdan sonra `drive.flush_and_unmount()` yalnız oturum sonunda.
- Hücreleri sırasız çalıştırmak → gizli durum. Şüphede: *Runtime → Restart and run all*.
