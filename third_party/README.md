# Üçüncü taraf model kodu (değiştirilmeden kopyalandı)

Bu klasördeki dosyalar resmi depolardan **olduğu gibi** kopyalandı. İçlerinde değişiklik yapılmaz. Uyum gereken her şey `scripts/backbones.py` içinde yapılır. Böylece "resmi kodu mu kullandık?" sorusunun cevabı bir sha256 karşılaştırmasıdır.

| Klasör | Kaynak | Commit | Dosyalar | Lisans |
|---|---|---|---|---|
| `panns/` | https://github.com/qiuqiangkong/audioset_tagging_cnn | `d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4` (2021-07-13) | `pytorch/models.py`, `pytorch/pytorch_utils.py`, `metadata/class_labels_indices.csv` | MIT (`LICENSE.MIT`, © 2018-2020 Qiuqiang Kong) |
| `beats/` | https://github.com/microsoft/unilm/tree/master/beats | `31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd` (depo HEAD, 2026-09-21; `beats/` klasörüne dokunan son commit) | `BEATs.py`, `backbone.py`, `modules.py` | MIT (`LICENSE`, © Microsoft) |

**sha256 (kopyalama anındaki):**
```
7f9af440395ace5160bbb51d654a0dc35fb887fbf5edecb12da61ff6efb306d9  panns/models.py
1464fcfbfc0fe4c55f690f6b39e1c80eeed5de1e7fd1b7fd30334d304de7dbe9  panns/pytorch_utils.py
cdd1049833c4b86127c2773ac0d14a2754b6a6d0d1798002ed5c66e699708429  panns/class_labels_indices.csv
27f289db7c56ce26f2ceb50d3719854b91b2dec1c2830d8b1dd8de1bbee19eeb  beats/BEATs.py
31c0378379a7e0f1d1069f9da444fb86890fe1ea078959a2dcd39640cdcadbaa  beats/backbone.py
edeb6b6cd6a784da749f932c3e0783c0bce556fc768d0d23a4d53d4b819eb424  beats/modules.py
```
`tests/test_phase2.py` bu değerleri kontrol eder (dosyalar yanlışlıkla değiştirilirse test düşer).

**Neden kopya, `pip install` değil?** PANNs'in pip paketi (`panns_inference`) yalnız CNN14'ü ve kendi ön işlemesini içeriyor; BEATs'in pip paketi yok. Resmi depodaki sınıfları birebir kullanmak, `load_state_dict(strict=True)` ile resmi ağırlıkların her anahtarının eşleştiğini doğrulamamızı sağlar.

**İçe aktarma:** Bu dosyalar kendi klasörlerini `sys.path`'te bekler (`from pytorch_utils import ...`, `from backbone import ...`). `scripts/backbones.py` ilgili klasörü yalnız içe aktarma sırasında yola ekler.

**Ağırlıklar burada değil.** Drive `models/` altında; kaynaklar ve sha256'lar `configs/model_input_contracts.yaml` ve SMK-001 raporunda.
