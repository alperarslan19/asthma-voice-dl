"""
Faz 3 — Uçtan uca fine-tuning (EXP-030; D-027, D-035, D-036, D-041, D-042; docs/PHASE3_DESIGN.md)

BİLİMSEL AMAÇ
    Önceden eğitilmiş ses modellerinin bütün katmanları (WavLM'in konvolüsyonel özellik kodlayıcısı hariç) astım /
    sağlıklı görevinde eğitilirse, katılımcı skoru dondurulmuş probun skoruna göre (1) daha iyi ayırıyor mu (E1) ve
    (2) ölçülen bağlamın ötesinde daha fazla bilgi taşıyor mu (C3)? Değerlendirme gün-gruplu split'lerde (D-041):
    aynı kayıt gününün bütün katılımcıları ya eğitimde ya testte; model "gün parmak izi"nden yararlanamaz.

VERİ / BİRİM
    Harmonize önbellek (D-030): her kayıt modelin SR'ında. Bir örnek = bir kayıt; 7 görevin hepsi tek modelde.
    Eğitim: her epoch'ta her kayıttan bir rastgele 4 s parça (tohumlu); 4 s'den kısa kayıt sağdan sıfırla doldurulur ve
    gerçek uzunluk modele verilir (BEATs / WavLM maskesi; PANNs'te maske yok — D-034).
    Değerlendirme: kaydın bütün 4 s / 2 s pencereleri (Faz 2 ile aynı) → kayıt logit'i = pencere logit'lerinin ortalaması
    → katılımcı skoru = kayıt logit'lerinin ortalaması (evaluate_phase3.py).

MODEL
    temsil: PANNs fc1 gömmesi (resmi 'embedding'); SSL'de 1..L katmanlarının zaman ortalamalarının öğrenilebilir
            softmax ağırlıklı toplamı (başlangıçta eşit ağırlık = Faz 2'nin birincil temsili).
    baş   : LayerNorm → Linear(d, 256) → ReLU → Dropout(0.3) → Linear(256, 1)  (Boll'un MLP başı; gizli boyut bizim seçimimiz)
    kayıp : sınıf-dengeli BCE (pos_weight = n_sağlıklı / n_astım, kayıt düzeyinde, eğitim kümesinden)
    eğitim: Adam (weight decay 1e-4), batch 16 (gerekirse birikimli), fp16 autocast + GradScaler, gradyan normu ≤ 1.0;
            augmentation, SpecAugment ve layerdrop KAPALI (D-016); dropout varsayılan.
    epoch : sabit; kolun kalibrasyon koşusunda (tekrar 0, fold 0'ın iç train / iç doğrulama ayrımı) iç doğrulama
            kaybının 3 epoch'luk hareketli ortalamasının en küçük olduğu epoch (D-027). Dış fold'larda erken durdurma ve
            checkpoint seçimi YOK; son epoch'un modeli test edilir.

KAYIT VE KURTARMA (proje kuralı)
    <out_root>/<arm>/r{r}_k{k}/ : latest.pt (her ckpt_every epoch'ta — CNN10 1, CNN14 5, SSL 2 — ve son epoch'ta; atomik: eğitilen parametreler, BatchNorm tamponları,
    optimizer, scaler, scheduler, RNG durumları, epoch), train_log.csv (epoch, kayıp, lr, adım, atlanan adım, süre, GPU GB),
    config.json (kol, tohum, imzalar). Fold bitince preds.csv + fold.json (atomik) yazılır; latest.pt silinir.
    Yeniden çalıştırma: bitmiş fold atlanır; yarım kalan fold latest.pt'den sürer. "En iyi doğrulama checkpoint'i" yoktur:
    tasarım gereği doğrulamaya göre seçim yapılmaz (D-027).

SIZINTI KONTROLLERİ (assert)
    train ∩ test katılımcı = ∅; train ve test kayıt günleri ayrık (gün-gruplu split); bir katılımcının bütün kayıtları
    aynı rolde; etiketler bağlam tablosundan; pos_weight yalnız eğitim kümesinden; test katılımcıları eğitime hiç girmez.

ÇALIŞTIRMA (Colab; notebooks/25_phase3_finetune.ipynb)
    python scripts/finetune.py calibrate --arm cnn10 ...      # bir kez, kol başına
    python scripts/finetune.py train --arm cnn10 --repeats 0,1,2 ...   # bütün fold'lar; biten atlanır
    --arm ssl: her fold'da D-035 seçimini (EXP-016DG d035_selection.csv) okur.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backbones as B  # noqa: E402
from smoke_test_backbones import HOP_S, WIN_S, Cache  # noqa: E402

# Faz 3 protokolü (docs/PHASE3_DESIGN.md; DECISIONS D-042). Kalibrasyon ve eğitim yalnız status == "KABUL" iken çalışır.
PHASE3 = {"version": "Faz 3 protokolü v1 (D-042)", "status": "ÖNERİLDİ", "splits": "splits_day (gün-gruplu, 3 tekrar × 5 fold)",
          "fine_tune": "uçtan uca (WavLM özellik kodlayıcısı donuk)", "epoch_rule": "kalibrasyon: iç doğrulama kaybı, 3'lü hareketli ort. min"}
COMMON = {"batch": 16, "lr_head": 1e-3, "weight_decay": 1e-4, "head_hidden": 256, "head_dropout": 0.3, "grad_clip": 1.0,
          "eval_batch": 16, "smooth": 3, "base_seed": 3000}
ARMS = {
    "cnn10":           {"backbone": "cnn10", "init": "pretrained", "lr_backbone": 1e-4, "warmup_frac": 0.0, "max_epochs": 40,
                        "micro_batch": 16, "ckpt_every": 1},
    "cnn10_scratch":   {"backbone": "cnn10", "init": "random", "lr_backbone": 1e-4, "warmup_frac": 0.0, "max_epochs": 40,
                        "micro_batch": 16, "ckpt_every": 1},
    "cnn14":           {"backbone": "cnn14", "init": "pretrained", "lr_backbone": 1e-4, "warmup_frac": 0.0, "max_epochs": 40,
                        "micro_batch": 16, "ckpt_every": 5},
    "beats":           {"backbone": "beats", "init": "pretrained", "lr_backbone": 2e-5, "warmup_frac": 0.1, "max_epochs": 20,
                        "micro_batch": 16, "ckpt_every": 2},
    "wavlm_base_plus": {"backbone": "wavlm_base_plus", "init": "pretrained", "lr_backbone": 2e-5, "warmup_frac": 0.1,
                        "max_epochs": 20, "micro_batch": 16, "ckpt_every": 2},
    "wavlm_large":     {"backbone": "wavlm_large", "init": "pretrained", "lr_backbone": 2e-5, "warmup_frac": 0.1,
                        "max_epochs": 20, "micro_batch": 8, "ckpt_every": 2},
}
SSL_CANDIDATES = ["beats", "wavlm_base_plus", "wavlm_large"]          # D-035
N_FOLDS = 5


# ----------------------------------------------------------------------------- yardımcılar
def sha256_short(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:16]


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_torch_save(obj, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def seed_for(r: int, k: int) -> int:
    """Kollar arasında AYNI: aynı fold'da bütün kollar aynı kayıt sırasını ve aynı kesim noktalarını görür (eşleştirme)."""
    return COMMON["base_seed"] + 100 * r + 10 * k


def protocol_gate(allow_unapproved: bool) -> None:
    if PHASE3["status"] != "KABUL" and not allow_unapproved:
        raise SystemExit(f"Faz 3 protokolü onaylanmadı ({PHASE3['version']}: {PHASE3['status']}; DECISIONS D-042). "
                         "Kalibrasyon / eğitim çalıştırılmadı. (SMK-002 bu kapıya tabi değildir.)")


# ----------------------------------------------------------------------------- veri
class FoldData:
    """Bir (tekrar, fold) için katılımcılar, etiketler ve önbellek satırları; rol ve sızıntı kontrolleri burada."""

    def __init__(self, splits_dir: Path, context: Path, cache: Cache, r: int, k: int, mode: str):
        assert mode in ("cv", "calibrate")
        S = pd.read_csv(splits_dir / f"outer_r{r}.csv")
        b = S[S.outer_fold == k].set_index("participant_id")
        C = pd.read_csv(context).set_index("participant_id")
        assert set(b.index) <= set(C.index), "split kohortu bağlam tablosunda yok"
        if mode == "cv":
            tr, te = b.index[b.role == "train"], b.index[b.role == "test"]
        else:                                              # kalibrasyon: dış test hiç kullanılmaz
            trb = b[b.role == "train"]
            tr, te = trb.index[trb.inner_fold != 0], trb.index[trb.inner_fold == 0]
        tr, te = set(tr), set(te)
        assert tr and te and not (tr & te), "train ∩ değerlendirme katılımcıları boş olmalı (SIZINTI)"
        day = C.recording_date.astype("string").fillna("nodate")
        day = day.where(day != "nodate", "nodate_" + day.index.astype(str))
        shared = set(day[list(tr)]) & set(day[list(te)])
        assert not shared, f"{len(shared)} kayıt günü hem eğitimde hem değerlendirmede (GÜN SIZINTISI; gün-gruplu split bekleniyordu)"
        idx = cache.idx.reset_index().rename(columns={"index": "row"})
        self.rows_train = idx[idx.participant_id.isin(tr)].row.to_numpy()
        self.rows_eval = idx[idx.participant_id.isin(te)].row.to_numpy()
        self.label = C.label.astype(int)
        self.pid_of_row = idx.set_index("row").participant_id
        self.slot_of_row = idx.set_index("row").slot
        assert tr <= set(idx.participant_id) and te <= set(idx.participant_id), "split katılımcısı önbellekte yok"
        assert not set(self.pid_of_row[self.rows_train]) & set(self.pid_of_row[self.rows_eval])
        y = self.label[self.pid_of_row[self.rows_train].to_numpy()].to_numpy()
        self.n_pos, self.n_neg = int(y.sum()), int((1 - y).sum())
        assert self.n_pos > 0 and self.n_neg > 0
        self.pos_weight = self.n_neg / self.n_pos           # sınıf-dengeli: iki sınıfın toplam ağırlığı eşit
        ye = self.label[self.pid_of_row[self.rows_eval].to_numpy()].to_numpy()
        self.summary = {"mode": mode, "repeat": r, "fold": k, "n_train_participants": len(tr), "n_eval_participants": len(te),
                        "n_train_recordings": int(len(self.rows_train)), "n_eval_recordings": int(len(self.rows_eval)),
                        "train_recordings_asthma_healthy": [self.n_pos, self.n_neg],
                        "eval_recordings_asthma_healthy": [int(ye.sum()), int((1 - ye).sum())],
                        "eval_participants_asthma_healthy": [int(self.label[list(te)].sum()), int((1 - self.label[list(te)]).sum())],
                        "pos_weight": round(self.pos_weight, 4), "n_days_train": len(set(day[list(tr)])),
                        "n_days_eval": len(set(day[list(te)]))}


def crop_batch(cache: Cache, rows, fracs, sr: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Her kayıttan bir 4 s parça. frac ∈ [0, 1) başlangıç noktasını belirler (SR'dan bağımsız → kollar aynı bölgeyi görür).
    Döner: wav (B, win) float32, lengths (B,) long (kısa kayıtta gerçek uzunluk, kalanı sıfır)."""
    win = int(WIN_S * sr)
    out = torch.zeros(len(rows), win, dtype=torch.float32)
    lengths = torch.empty(len(rows), dtype=torch.long)
    for i, (r, f) in enumerate(zip(rows, fracs)):
        x = cache.audio(int(r), sr)
        n = len(x)
        if n > win:
            s = int(f * (n - win + 1))
            out[i] = torch.from_numpy(x[s:s + win])
            lengths[i] = win
        else:
            out[i, :n] = torch.from_numpy(x)
            lengths[i] = n
    assert torch.isfinite(out).all()
    return out, lengths


def epoch_plan(rows: np.ndarray, seed: int, epoch: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng([seed, epoch])
    return rows[rng.permutation(len(rows))], rng.random(len(rows))


# ----------------------------------------------------------------------------- model
class FTModel(nn.Module):
    def __init__(self, bb: B.Backbone, hidden: int, dropout: float):
        super().__init__()
        self.bb = bb
        self.backbone = bb.model                          # parametreler bu modülün altında kayıtlı
        self.ssl = bb.family in ("beats", "wavlm")
        if self.ssl:
            self.layer_logits = nn.Parameter(torch.zeros(len(bb.primary_layers)))
        self.head = nn.Sequential(nn.LayerNorm(bb.dim), nn.Linear(bb.dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, 1))

    def features(self, wav: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
        if not self.ssl:
            return self.bb.model(wav)["embedding"]       # PANNs: resmi gömme; dolgu (varsa) girdiye dahil (D-034)
        _, valid = self.bb._run(wav, lengths)
        tdim = 0 if self.bb.family == "beats" else 1
        P = torch.stack([self.bb._masked_mean(self.bb._acts[i], valid, tdim) for i in self.bb.primary_layers], dim=1)
        assert P.shape == (wav.shape[0], len(self.bb.primary_layers), self.bb.dim), f"katman yığını {tuple(P.shape)}"
        w = torch.softmax(self.layer_logits.float(), 0).to(P.dtype)
        return (P * w[None, :, None]).sum(1)

    def forward(self, wav: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.features(wav, lengths).float()).squeeze(1)


def build_model(arm: str, model_dir, device: str, random_init_test: bool = False) -> FTModel:
    cfg = ARMS[arm]
    random_init = random_init_test or cfg["init"] == "random"
    bb = B.build_backbone(cfg["backbone"], model_dir=None if random_init else model_dir, device=device, random_init=random_init)
    bb.disable_augmentation()                            # D-016: SpecAugment / maskeleme kapalı
    if bb.family == "wavlm":
        bb.model.config.layerdrop = 0.0                  # bütün katmanlar her adımda çalışmalı (kancalar + tanım)
        for p in bb.model.feature_extractor.parameters():
            p.requires_grad = False                      # konvolüsyonel özellik kodlayıcısı donuk (yaygın uygulama)
    if bb.family == "beats":
        bb.model.encoder.layerdrop = 0.0
    m = FTModel(bb, COMMON["head_hidden"], COMMON["head_dropout"]).to(device)
    m.info = dict(bb.info)
    return m


def trainable_state(m: FTModel) -> dict:
    """Eğitilen parametreler + tamponlar (BatchNorm istatistikleri). Donuk parametreler checkpoint'e girmez."""
    keep = {n for n, p in m.named_parameters() if p.requires_grad} | {n for n, _ in m.named_buffers()}
    return {n: t.detach().cpu().clone() for n, t in m.state_dict().items() if n in keep}


def make_optimizer(m: FTModel, cfg: dict):
    head = list(m.head.parameters()) + ([m.layer_logits] if m.ssl else [])
    hid = {id(p) for p in head}
    body = [p for p in m.parameters() if p.requires_grad and id(p) not in hid]
    return torch.optim.Adam([{"params": body, "lr": cfg["lr_backbone"]}, {"params": head, "lr": COMMON["lr_head"]}],
                            weight_decay=COMMON["weight_decay"])


def make_scheduler(opt, total_steps: int, warmup_frac: float):
    w = int(round(warmup_frac * total_steps))
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / w) if w > 0 else 1.0)


@torch.no_grad()
def predict_recordings(m: FTModel, cache: Cache, rows, device: str, amp: bool) -> pd.DataFrame:
    """Bütün 4 s / 2 s pencereler (Faz 2 ile aynı); kısa kayıt tek parça, sıfır dolgulu + gerçek uzunluk (zeropad, D-034)."""
    m.eval()
    sr = m.bb.sr
    win, hop = int(WIN_S * sr), int(HOP_S * sr)
    out = []
    use_amp = amp and device.startswith("cuda")
    for r in rows:
        x = cache.audio(int(r), sr)
        ws = B.window_starts(len(x), win, hop)
        W = torch.zeros(len(ws), win)
        L = torch.empty(len(ws), dtype=torch.long)
        for i, (s, n) in enumerate(ws):
            W[i, :n] = torch.from_numpy(x[s:s + n])
            L[i] = n
        logits = []
        for i in range(0, len(ws), COMMON["eval_batch"]):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                lg = m(W[i:i + COMMON["eval_batch"]].to(device), L[i:i + COMMON["eval_batch"]].to(device))
            logits.append(lg.float().cpu())
        lg = torch.cat(logits)
        assert lg.shape == (len(ws),) and torch.isfinite(lg).all(), f"satır {r}: sonlu olmayan logit"
        out.append({"row": int(r), "n_windows": len(ws), "logit": float(lg.mean())})
    return pd.DataFrame(out)


def rng_state() -> dict:
    s = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def set_rng_state(s: dict) -> None:
    torch.set_rng_state(s["torch"])
    np.random.set_state(s["numpy"])
    random.setstate(s["python"])
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


# ----------------------------------------------------------------------------- eğitim döngüsü
def run_training(m: FTModel, fd: FoldData, cache: Cache, cfg: dict, epochs: int, seed: int, workdir: Path, device: str,
                 amp: bool = True, eval_each_epoch: bool = False, stop_after_epoch: int | None = None) -> dict:
    """epochs kadar eğitir (kaldığı yerden sürer). eval_each_epoch: kalibrasyon için değerlendirme kümesinde kayıp / AUC.
    Döner: {'log': DataFrame, 'done': bool}."""
    from sklearn.metrics import roc_auc_score
    workdir.mkdir(parents=True, exist_ok=True)
    use_amp = amp and device.startswith("cuda")
    batch, micro = COMMON["batch"], cfg["micro_batch"]
    assert batch % micro == 0
    assert batch // micro >= 1                  # birikim: batch = micro × (batch // micro)
    steps_per_epoch = math.ceil(len(fd.rows_train) / batch)
    opt = make_optimizer(m, cfg)
    sch = make_scheduler(opt, steps_per_epoch * epochs, cfg["warmup_frac"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    pos_w = torch.tensor(fd.pos_weight, device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    ck, logf = workdir / "latest.pt", workdir / "train_log.csv"
    start, log = 0, []
    B.set_determinism(seed)
    if ck.exists():
        st = torch.load(ck, map_location="cpu", weights_only=False)
        assert st["seed"] == seed and st["epochs_planned"] == epochs, "checkpoint başka bir yapılandırmaya ait"
        missing = m.load_state_dict(st["model"], strict=False)
        assert not missing.unexpected_keys, missing.unexpected_keys
        trainable = {n for n, p in m.named_parameters() if p.requires_grad}
        assert trainable <= set(st["model"]), "checkpoint'te eğitilen parametre eksik"
        opt.load_state_dict(st["opt"])
        sch.load_state_dict(st["sch"])
        scaler.load_state_dict(st["scaler"])
        set_rng_state(st["rng"])
        start = st["epoch"]
        log = pd.read_csv(logf).to_dict("records") if logf.exists() else []
        log = [r for r in log if r["epoch"] <= start]
        print(f"  devam: epoch {start}/{epochs} ({ck})", flush=True)
    t_fold = time.time()
    for ep in range(start, epochs):
        m.train()
        t0 = time.time()
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        order, fracs = epoch_plan(fd.rows_train, seed, ep)
        y_all = fd.label[fd.pid_of_row[order].to_numpy()].to_numpy().astype(np.float32)
        tot, n_seen, skipped, n_steps = 0.0, 0, 0, 0
        for i in range(0, len(order), batch):
            opt.zero_grad(set_to_none=True)
            b_rows, b_frac, b_y = order[i:i + batch], fracs[i:i + batch], y_all[i:i + batch]
            for j in range(0, len(b_rows), micro):
                wav, lengths = crop_batch(cache, b_rows[j:j + micro], b_frac[j:j + micro], m.bb.sr)
                yb = torch.from_numpy(b_y[j:j + micro]).to(device)
                assert wav.shape == (len(yb), int(WIN_S * m.bb.sr)) and wav.dtype == torch.float32
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    logit = m(wav.to(device), lengths.to(device))
                    loss = lossf(logit.float(), yb)
                assert torch.isfinite(loss) or use_amp, "sonlu olmayan kayıp (fp32)"
                scaler.scale(loss * len(yb) / len(b_rows)).backward()
                tot += float(loss.detach()) * len(yb)
                n_seen += len(yb)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], COMMON["grad_clip"])
            sc0 = float(scaler.get_scale()) if use_amp else 1.0
            scaler.step(opt)
            scaler.update()
            if use_amp and float(scaler.get_scale()) < sc0:
                skipped += 1
            sch.step()
            n_steps += 1
        assert n_seen == len(order) and n_steps == steps_per_epoch
        row = {"epoch": ep + 1, "train_loss": tot / n_seen, "lr_backbone": opt.param_groups[0]["lr"],
               "lr_head": opt.param_groups[1]["lr"], "steps": n_steps, "skipped_steps": skipped,
               "epoch_s": round(time.time() - t0, 1), "elapsed_s": round(time.time() - t_fold, 1),
               "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.startswith("cuda") else 0.0}
        assert math.isfinite(row["train_loss"]), f"epoch {ep + 1}: eğitim kaybı sonlu değil"
        assert skipped < n_steps, f"epoch {ep + 1}: bütün adımlar atlandı (fp16 taşması)"
        if eval_each_epoch:
            P = predict_recordings(m, cache, fd.rows_eval, device, amp)
            ye = fd.label[fd.pid_of_row[P.row].to_numpy()].to_numpy().astype(np.float32)
            lg = torch.tensor(P.logit.to_numpy(), dtype=torch.float32)
            row["val_loss"] = float(nn.BCEWithLogitsLoss(pos_weight=torch.tensor(fd.pos_weight))(lg, torch.from_numpy(ye)))
            part = P.assign(pid=fd.pid_of_row[P.row].to_numpy()).groupby("pid").logit.mean()
            row["val_auc_monitor_only"] = float(roc_auc_score(fd.label[part.index], part))
        log.append(row)
        print("  " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
        if (ep + 1) % cfg["ckpt_every"] == 0 or ep + 1 == epochs or (stop_after_epoch and ep + 1 == stop_after_epoch):
            atomic_torch_save({"model": trainable_state(m), "opt": opt.state_dict(), "sch": sch.state_dict(),
                               "scaler": scaler.state_dict(), "rng": rng_state(), "epoch": ep + 1, "seed": seed,
                               "epochs_planned": epochs}, ck)
            atomic_write_text(logf, pd.DataFrame(log).to_csv(index=False))
        if stop_after_epoch and ep + 1 == stop_after_epoch and ep + 1 < epochs:
            return {"log": pd.DataFrame(log), "done": False}
    atomic_write_text(logf, pd.DataFrame(log).to_csv(index=False))
    return {"log": pd.DataFrame(log), "done": True}


def choose_epochs(val_loss: list[float], smooth: int) -> int:
    """D-027 / D-042: iç doğrulama kaybının `smooth` epoch'luk ortalanmış hareketli ortalamasının en küçük olduğu epoch
    (1'den sayılır). Kenarlarda pencere kırpılır. Eşitlikte küçük epoch."""
    v = np.asarray(val_loss, float)
    assert len(v) >= 1 and np.isfinite(v).all()
    h = smooth // 2
    ma = np.array([v[max(0, i - h): i + h + 1].mean() for i in range(len(v))])
    return int(np.argmin(ma)) + 1


# ----------------------------------------------------------------------------- komutlar
def signatures(args, cfg_arm: dict, m: FTModel) -> dict:
    return {"splits": {f"outer_r{r}.csv": sha256_short(args.splits_dir / f"outer_r{r}.csv")
                       for r in range(3) if (args.splits_dir / f"outer_r{r}.csv").exists()},
            "context": sha256_short(args.context), "cache_index": sha256_short(args.cache_dir / "index.csv"),
            "checkpoint_sha256": m.info.get("checkpoint_sha256", "random_init"), "code": sha256_short(Path(__file__)),
            "protocol": dict(PHASE3), "common": dict(COMMON), "arm_cfg": dict(cfg_arm)}


def resolve_backbone_arm(arm: str, r: int, k: int, selection: Path | None) -> str:
    if arm != "ssl":
        return arm
    assert selection is not None and selection.exists(), "--arm ssl için D-035 seçim dosyası gerekli (EXP-016DG d035_selection.csv)"
    sel = pd.read_csv(selection)
    hit = sel[(sel.repeat == r) & (sel.fold == k)]
    assert len(hit) == 1 and hit.selected.iloc[0] in SSL_CANDIDATES, f"r{r} k{k}: D-035 seçimi yok"
    return str(hit.selected.iloc[0])


def cmd_calibrate(args) -> None:
    protocol_gate(args.allow_unapproved)
    cfg = ARMS[args.arm]
    out = args.out_root / args.arm / "calibration"
    res_path = out / "calibration.json"
    if res_path.exists():
        print(f"{args.arm}: kalibrasyon zaten var → {json.loads(res_path.read_text())['epochs']} epoch")
        return
    cache = Cache(args.cache_dir)
    fd = FoldData(args.splits_dir, args.context, cache, 0, 0, "calibrate")
    print(f"{args.arm} kalibrasyon: {fd.summary}", flush=True)
    seed = seed_for(0, 0)
    B.set_determinism(seed)                              # rastgele başlatılan kısımlar (sıfırdan kol, testler) yeniden üretilebilir
    m = build_model(args.arm, args.model_dir, args.device, args.random_init_test)
    E = args.max_epochs or cfg["max_epochs"]
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out / "config.json", json.dumps({"arm": args.arm, "mode": "calibrate", "max_epochs": E, "seed": seed,
                                                       "data": fd.summary, "sig": signatures(args, cfg, m)}, indent=1, default=str))
    res = run_training(m, fd, cache, cfg, E, seed, out, args.device, eval_each_epoch=True, stop_after_epoch=args.stop_after_epoch)
    if not res["done"]:
        print("durduruldu (test); yeniden çalıştırınca sürer")
        return
    log = res["log"]
    e_star = choose_epochs(log.val_loss.tolist(), COMMON["smooth"])
    at_edge = e_star == E
    result = {"arm": args.arm, "epochs": e_star, "max_epochs": E, "at_max_edge": at_edge, "rule": PHASE3["epoch_rule"],
              "val_loss": log.val_loss.round(5).tolist(), "train_loss": log.train_loss.round(5).tolist(),
              "val_auc_monitor_only": log.val_auc_monitor_only.round(4).tolist(), "data": fd.summary,
              "epoch_s_mean": float(log.epoch_s.mean()), "gpu_peak_gb_max": float(log.gpu_peak_gb.max())}
    atomic_write_text(res_path, json.dumps(result, indent=1))
    (out / "latest.pt").unlink(missing_ok=True)
    print(f"{args.arm}: sabit epoch = {e_star}" + (" (UYARI: üst sınırda — kayıp hâlâ düşüyor olabilir)" if at_edge else ""))


def cmd_train(args) -> None:
    protocol_gate(args.allow_unapproved)
    cache = Cache(args.cache_dir)
    repeats = [int(x) for x in args.repeats.split(",")]
    folds = [int(x) for x in args.folds.split(",")]
    for r in repeats:
        for k in folds:
            arm_bb = resolve_backbone_arm(args.arm, r, k, args.selection)
            cfg = ARMS[arm_bb]
            cal = args.out_root / arm_bb / "calibration" / "calibration.json"
            assert cal.exists(), f"{arm_bb}: kalibrasyon yok ({cal}); önce `finetune.py calibrate --arm {arm_bb}`"
            E = json.loads(cal.read_text())["epochs"]
            wd = args.out_root / args.arm / f"r{r}_k{k}"
            if (wd / "fold.json").exists():
                print(f"{args.arm} r{r} k{k}: bitmiş, atlandı")
                continue
            fd = FoldData(args.splits_dir, args.context, cache, r, k, "cv")
            print(f"{args.arm} r{r} k{k} ({arm_bb}, {E} epoch): {fd.summary}", flush=True)
            seed = seed_for(r, k)
            B.set_determinism(seed)
            m = build_model(arm_bb, args.model_dir, args.device, args.random_init_test)
            wd.mkdir(parents=True, exist_ok=True)
            sig = signatures(args, cfg, m)
            if (wd / "config.json").exists():
                old = json.loads((wd / "config.json").read_text())
                assert old["sig"] == json.loads(json.dumps(sig, default=str)) and old["epochs"] == E, \
                    f"{wd}: yarım kalan fold farklı bir yapılandırmaya ait (imza / epoch); klasörü inceleyin"
            else:
                atomic_write_text(wd / "config.json", json.dumps({"arm": args.arm, "backbone_arm": arm_bb, "repeat": r, "fold": k,
                                                                 "epochs": E, "seed": seed, "data": fd.summary, "sig": sig},
                                                                indent=1, default=str))
            t0 = time.time()
            res = run_training(m, fd, cache, cfg, E, seed, wd, args.device, stop_after_epoch=args.stop_after_epoch)
            if not res["done"]:
                print("durduruldu (test); yeniden çalıştırınca sürer")
                return
            P = predict_recordings(m, cache, fd.rows_eval, args.device, amp=True)
            P["participant_id"] = fd.pid_of_row[P.row].to_numpy()
            P["slot"] = fd.slot_of_row[P.row].to_numpy()
            P["label"] = fd.label[P.participant_id].to_numpy()
            P = P.assign(arm=args.arm, backbone_arm=arm_bb, repeat=r, fold=k)
            assert set(P.participant_id) == set(fd.pid_of_row[fd.rows_eval]), "bütün test katılımcıları tahmin almalı"
            atomic_write_text(wd / "preds.csv", P.to_csv(index=False))
            log = res["log"]
            atomic_write_text(wd / "fold.json", json.dumps({
                "arm": args.arm, "backbone_arm": arm_bb, "repeat": r, "fold": k, "epochs": E, "seed": seed, "data": fd.summary,
                "final_train_loss": float(log.train_loss.iloc[-1]), "skipped_steps": int(log.skipped_steps.sum()),
                "train_s": float(log.epoch_s.sum()), "total_s": round(time.time() - t0, 1),
                "gpu_peak_gb_max": float(log.gpu_peak_gb.max()), "sig": sig}, indent=1, default=str))
            if args.keep_final:
                atomic_torch_save({"model": trainable_state(m), "epoch": E, "seed": seed}, wd / "final_trainable.pt")
            (wd / "latest.pt").unlink(missing_ok=True)
            print(f"{args.arm} r{r} k{k}: bitti ({time.time() - t0:.0f} s)", flush=True)
            del m
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("calibrate", "train"):
        p = sub.add_parser(name)
        p.add_argument("--arm", required=True, choices=list(ARMS) + ["ssl"])
        p.add_argument("--splits-dir", required=True, type=Path, help="splits_day/ (gün-gruplu)")
        p.add_argument("--context", required=True, type=Path)
        p.add_argument("--cache-dir", required=True, type=Path)
        p.add_argument("--model-dir", type=Path)
        p.add_argument("--out-root", required=True, type=Path, help="Drive: experiments/phase3/finetune")
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--allow-unapproved", action="store_true", help="yalnız testler")
        p.add_argument("--random-init-test", action="store_true", help="yalnız testler: küçük rastgele modeller")
        p.add_argument("--stop-after-epoch", type=int, help="yalnız testler: kesinti benzetimi")
        if name == "calibrate":
            p.add_argument("--max-epochs", type=int, help="yalnız testler (varsayılan: kol yapılandırması)")
        else:
            p.add_argument("--repeats", default="0,1,2")
            p.add_argument("--folds", default="0,1,2,3,4")
            p.add_argument("--selection", type=Path, help="--arm ssl: EXP-016DG d035_selection.csv")
            p.add_argument("--keep-final", action="store_true", help="fold sonunda eğitilen ağırlıkları sakla (Drive alanı)")
    args = ap.parse_args()
    if args.cmd == "calibrate":
        assert args.arm != "ssl", "kalibrasyon somut bir kol için yapılır (beats / wavlm_base_plus / wavlm_large)"
        cmd_calibrate(args)
    else:
        cmd_train(args)


if __name__ == "__main__":
    main()
