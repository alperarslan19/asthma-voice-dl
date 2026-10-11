# DIAG-001 — cnn14: sonlu olmayan logit teşhisi (otomatik üretildi; katılımcı ID'si içermez)

- Ortam: Tesla T4 · torch 2.11.0+cu130 · 1 epoch · 392 iç doğrulama kaydı · 131.3 s
- A önceden eğitilmiş, fp16: 0 / 392 kayıt sonlu değil
- B eğitim: {'completed': True, 'train_loss': [0.2461], 'skipped_steps': [1]}
- B ilk bozulan adım: None
- C eğitim sonrası sonlu olmayan parametre / tampon: 0 (türler: [])
- D eğitim sonrası: fp16 3 / 392, fp32 0 / 392 kayıt sonlu değil
- E fp16'da ilk sonlu olmayan modül: {'module': 'backbone.conv_block6.conv1', 'dtype': 'float16', 'finite': False, 'max_abs': 64512.0}; fp32'de 65504'ü aşan modüller: ['backbone.conv_block6.conv1', 'backbone.conv_block6.conv2', 'backbone.conv_block6.bn2']

**Okuma (D-043'te önceden yazılmış eşleme):** H2: durum sağlam, fp16 değerlendirme taşıyor → değerlendirme bütün kollarda fp32
