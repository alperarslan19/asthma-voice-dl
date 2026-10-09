"""
SMK-001 — Faz 2 smoke test'i: her backbone için yükleme, sözleşme, şekil, determinizm, işlevsel kontrol ve
Faz 3 bellek / hız ölçümü (D-034 madde 10; docs/PHASE2_DESIGN.md Bölüm 5.1; docs/COLAB_WORKFLOW.md Bölüm 7.5)

BİLİMSEL AMAÇ
    Tam çıkarımdan ÖNCE, her modelin (1) resmi ağırlığı eksiksiz yüklediğini, (2) resmi girdi biçimini aldığını,
    (3) eval modunda deterministik ve batch'ten bağımsız olduğunu (V2: katılımcılar arası bilgi akışı yok),
    (4) anlamlı çıktı verdiğini göstermek. "Kod çalıştı" yetmez: S9 (PANNs AudioSet "Speech" sınıfı) ve S10
    (aynı kişinin kayıtları birbirine daha benzer) temsillerin çökmediğini ve frontend'in doğru olduğunu sınar.
    Ayrıca Faz 3 için eğitim modunda bellek ve adım süresi ölçülür (S12) — fine-tune bütçesi buna göre yazılacak.

KONTROLLER (her biri PASS / FAIL / INFO; tek FAIL → çıkış kodu 1, çıkarım başlatılmaz)
    S1 checkpoint sha256 (models/MANIFEST.json ile)   S2 sözleşme (yükleyicide assert)   S3 strict yükleme
    S4 parametre sayısı   S5 ileri geçiş: şekil/dtype/sonluluk   S6 eval determinizmi   S7 batch-değişmezlik
    S8 son katman kancası = resmi çıktı   S9 PANNs: kelime kayıtlarında "Speech" ilk 3'te
    S10 aynı-kişi benzerliği (farklı görevler)   S11 kaydet → yükle → aynı çıktı   S12 Faz 3 eğitim adımı   S13 hız
    S14 dolgu yolu: tam uzunluk + lengths = dolgusuz çıktı (PASS ölçütü); kısa parça sıfır dolgulu vs dolgusuz benzerliği (INFO)

İKİ AYRI KULLANIM (karıştırılmamalı)
    Gerçek checkpoint'lerle (Colab, GPU): rapor = SMK-001_smoke.{json,md}. Gömme çıkarımının ÖN KOŞULU budur.
    --random-init (yalnız birim testi; ağırlıksız küçük modeller): rapor = UNITTEST_random_init_smoke.{json,md}.
        Bu, kod yollarının çalıştığını gösterir; modellerin doğru yüklendiğini / anlamlı çıktı verdiğini GÖSTERMEZ.
        S1, S9, S10 bu kipte yalnız INFO'dur. SMK-001 yerine geçmez.

GİRDİLER
    --cache-dir   audio_cache_v1 (index.csv, audio_32k.f32, audio_16k.f32)
    --model-dir   Drive models/ (checkpoint'ler; MANIFEST.json burada tutulur)
ÇIKTI
    <report_dir>/SMK-001_smoke.{json,md}   (gerçek checkpoint; yalnız agrega; katılımcı ID'si içermez)
    <report_dir>/UNITTEST_random_init_smoke.{json,md}   (--random-init)
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backbones as B  # noqa: E402

WIN_S, HOP_S = 4.0, 2.0
SPEECH_IDX = 0          # AudioSet class_labels_indices.csv: 0 = "Speech"


# ----------------------------------------------------------------------------- önbellek okuma
class Cache:
    def __init__(self, cache_dir: Path):
        self.dir = Path(cache_dir)
        self.idx = pd.read_csv(self.dir / "index.csv")
        need = {"participant_id", "slot", "offset_32k", "n_32k", "offset_16k", "n_16k"}
        assert need <= set(self.idx.columns), f"index.csv eksik sütun: {need - set(self.idx.columns)}"
        self._mm = {}

    def audio(self, row: int, sr: int) -> np.ndarray:
        k = f"{sr // 1000}k"
        if k not in self._mm:
            self._mm[k] = np.memmap(self.dir / f"audio_{k}.f32", dtype="<f4", mode="r")
        r = self.idx.iloc[row]
        o, n = int(r[f"offset_{k}"]), int(r[f"n_{k}"])
        x = np.asarray(self._mm[k][o:o + n], dtype=np.float32)
        assert len(x) == n and np.isfinite(x).all()
        return x

    def windows(self, row: int, sr: int) -> torch.Tensor:
        x = self.audio(row, sr)
        ws = B.window_starts(len(x), int(WIN_S * sr), int(HOP_S * sr))
        return torch.from_numpy(np.stack([x[s:s + n] for s, n in ws]))


def pick_rows(cache: Cache, n_part: int, slots=None) -> list[int]:
    """Deterministik örnek: 4 s'den uzun kayıtları olan ilk n_part katılımcı (ID sırası), istenen slotlar."""
    idx = cache.idx.reset_index().rename(columns={"index": "row"})
    long_ok = idx.groupby("participant_id").n_16k.min() >= 4 * 16000
    full = idx.groupby("participant_id").slot.nunique() == 7
    pids = sorted(set(long_ok[long_ok].index) & set(full[full].index))[:n_part]
    sel = idx[idx.participant_id.isin(pids)]
    if slots is not None:
        sel = sel[sel.slot.isin(slots)]
    return sel.sort_values(["participant_id", "slot"]).row.tolist()


# ----------------------------------------------------------------------------- kontroller
def rec_embedding(bb, cache, row) -> np.ndarray:
    return bb.embed(cache.windows(row, bb.sr)).mean(0).numpy()      # (n_store, dim)


def same_person_auc(bb, cache, rows) -> dict:
    """S10: farklı görev kayıt çiftlerinde, aynı kişi vs farklı kişi kosinüs benzerliği (AUC)."""
    E = np.stack([rec_embedding(bb, cache, r) for r in rows])[:, bb.primary_layers, :]   # (n, Lp, D)
    E = (E - E.mean(0)) / (E.std(0) + 1e-8)
    X = E.mean(1)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    meta = cache.idx.iloc[rows]
    pid, slot = meta.participant_id.to_numpy(), meta.slot.to_numpy()
    S = X @ X.T
    iu = np.triu_indices(len(rows), 1)
    diff_task = slot[iu[0]] != slot[iu[1]]
    same = (pid[iu[0]] == pid[iu[1]])[diff_task]
    sim = S[iu][diff_task]
    return {"auc_same_vs_diff_person": float(roc_auc_score(same, sim)), "mean_sim_same": float(sim[same].mean()),
            "mean_sim_diff": float(sim[~same].mean()), "n_pairs_same": int(same.sum()), "n_pairs_diff": int((~same).sum())}


def clone_backbone(bb):
    """S11: state_dict'i baytlara yaz → yeni örneğe strict yükle (Faz 3 checkpoint akışının özü)."""
    buf = io.BytesIO()
    torch.save(bb.model.state_dict(), buf)
    buf.seek(0)
    state = torch.load(buf, map_location="cpu", weights_only=True)
    if bb.family == "panns":
        fresh = type(bb.model)(**bb.contract["constructor"])
    elif bb.family == "beats":
        fresh = type(bb.model)(bb.model.cfg)
    else:
        fresh = type(bb.model)(copy.deepcopy(bb.model.config))
    fresh.load_state_dict(state, strict=True)
    args = (bb.name, bb.contract, fresh, bb.device, dict(bb.info))
    return type(bb)(*args, bb.fe) if bb.family == "wavlm" else type(bb)(*args)


def train_probe(bb, wav: torch.Tensor, amp: bool, grad_ckpt: bool, steps: int = 3) -> dict:
    """S12: augmentation kapalı (D-016), eğitim modu, ağırlıklı BCE yerine düz BCE (yalnız bellek/hız ölçümü)."""
    dev = bb.device
    bb.disable_augmentation()
    if grad_ckpt:
        bb.model.gradient_checkpointing_enable()
    bb.model.train()
    head = nn.Linear(bb.dim, 1).to(dev)
    params = [p for p in bb.model.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    use_amp = amp and dev.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    y = (torch.arange(len(wav)) % 2).float().to(dev)
    probe_p = next(p for p in bb.model.parameters() if p.requires_grad and p.ndim > 1)
    before = probe_p.detach().clone()
    if dev.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    times, losses, step_log = [], [], []
    for _ in range(steps):
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logit = head(bb.train_features(wav.to(dev)).float()).squeeze(1)
            loss = nn.functional.binary_cross_entropy_with_logits(logit, y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        grads_finite = all(torch.isfinite(p.grad).all().item() for p in params if p.grad is not None)
        scale_before = float(scaler.get_scale()) if use_amp else 1.0
        scaler.step(opt)
        scaler.update()
        step_log.append({"loss": float(loss.item()), "loss_finite": bool(torch.isfinite(loss).item()),
                         "grads_finite": grads_finite, "scale": scale_before,
                         "skipped": bool(use_amp and float(scaler.get_scale()) < scale_before)})
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        times.append(time.time() - t0)
        losses.append(float(loss.item()))
    out = {"batch": int(len(wav)), "seconds_per_window": round(wav.shape[1] / bb.sr, 2), "amp_fp16": use_amp,
           "grad_checkpointing": grad_ckpt, "loss_finite": bool(np.isfinite(losses).all()),
           # GradScaler ilk adımlarda taşan gradyanla adımı atlayabilir (normal); en az bir adım sonlu gradyanla atılmalı
           "grads_finite": any(s["grads_finite"] for s in step_log), "steps": step_log,
           "frontend_fp32": bool(bb.info.get("frontend_fp32_under_autocast", False)),
           "param_changed": bool((probe_p.detach() - before).abs().max().item() > 0),
           "step_s": round(float(np.mean(times[1:])) if len(times) > 1 else times[0], 3)}
    if dev.startswith("cuda"):
        out["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return out


def check_backbone(name, args, cache, manifest) -> dict:
    res = {"backbone": name, "checks": {}}
    C = res["checks"]

    def put(k, status, **v):
        C[k] = {"status": status, **v}

    dev = args.device
    t0 = time.time()
    bb = B.build_backbone(name, model_dir=args.model_dir, device=dev, random_init=args.random_init)
    res["info"] = {k: v for k, v in bb.info.items() if k not in {"cfg"}}
    res["load_s"] = round(time.time() - t0, 1)
    # S1 sha256
    if args.random_init:
        put("S1_sha256", "INFO", note="random_init (test)")
    else:
        sha = bb.info["checkpoint_sha256"]
        key = bb.contract["checkpoint_file"]
        prev = manifest.get(key)
        manifest[key] = sha
        put("S1_sha256", "PASS" if prev in (None, sha) else "FAIL", sha256=sha, previous=prev)
    put("S2_contract", "PASS", note="sözleşme alanları yükleyicide assert edildi", sr=bb.sr, n_store=bb.n_store, dim=bb.dim)
    put("S3_strict_load", "PASS" if not bb.info.get("missing_keys") else "FAIL",
        missing=len(bb.info.get("missing_keys", [])), unexpected=len(bb.info.get("unexpected_keys", [])))
    put("S4_params", "INFO", n_params=bb.n_params())

    rows_all = pick_rows(cache, args.n_participants)
    row0 = rows_all[0]
    w = cache.windows(row0, bb.sr)
    if dev.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    e1 = bb.embed(w)
    ok = e1.shape == (len(w), bb.n_store, bb.dim) and e1.dtype == torch.float32 and torch.isfinite(e1).all().item()
    put("S5_forward", "PASS" if ok else "FAIL", in_shape=list(w.shape), out_shape=list(e1.shape), dtype=str(e1.dtype),
        model_device=str(next(bb.model.parameters()).device),
        peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 1e9, 3) if dev.startswith("cuda") else None)
    e2 = bb.embed(w)
    d6 = float((e1 - e2).abs().max())
    put("S6_determinism", "PASS" if d6 < 1e-5 else "FAIL", max_abs_diff=d6)
    pid0 = cache.idx.participant_id.iloc[row0]
    other_rows = [r for r in rows_all if cache.idx.participant_id.iloc[r] != pid0][:3]     # başka katılımcılar
    other = torch.cat([cache.windows(r, bb.sr) for r in other_rows])
    eb = bb.embed(torch.cat([w, other]))[: len(w)]
    eo = bb.embed(w[:1])
    d7 = max(float((eb - e1).abs().max()), float((eo - e1[:1]).abs().max()))
    rel7 = d7 / (float(e1.abs().max()) + 1e-12)
    put("S7_batch_invariance", "PASS" if (d7 < 1e-4 or rel7 < 1e-4) else "FAIL", max_abs_diff=d7, rel=rel7)   # GPU'da batch boyutu algoritma seçimini değiştirebilir; sızıntı O(1) göreli fark üretirdi
    if bb.family == "panns":
        put("S8_hook_vs_official", "INFO", note="PANNs: resmi 'embedding' doğrudan kullanılıyor")
        word_rows = pick_rows(cache, args.n_participants, slots=[2, 3, 4, 5, 6, 7])
        P = np.stack([bb.clipwise(cache.windows(r, bb.sr)).mean(0).numpy() for r in word_rows])
        mean_p = P.mean(0)
        rank = int((mean_p > mean_p[SPEECH_IDX]).sum()) + 1
        labels = pd.read_csv(B.THIRD / "panns" / "class_labels_indices.csv")
        top = [labels.display_name.iloc[i] for i in np.argsort(-mean_p)[:5]]
        status = "INFO" if args.random_init else ("PASS" if rank <= 3 else "FAIL")
        put("S9_speech_class", status, speech_rank=rank, top5=top, n_recordings=len(word_rows))
    else:
        ours, off = bb.official_last(w)
        d8 = float((ours - off).abs().max())
        put("S8_hook_vs_official", "PASS" if d8 < 1e-4 else "FAIL", max_abs_diff=d8)
        put("S9_speech_class", "INFO", note="sınıflandırıcısı yok (öz-gözetimli)")
    s10 = same_person_auc(bb, cache, rows_all)
    put("S10_same_person", "INFO" if args.random_init else ("PASS" if s10["auc_same_vs_diff_person"] > 0.6 else "FAIL"), **s10)
    bb2 = clone_backbone(bb)
    d11 = float((bb2.embed(w) - e1).abs().max())
    put("S11_save_load", "PASS" if d11 < 1e-5 else "FAIL", max_abs_diff=d11)
    del bb2
    # S13 hız (eval)
    batch = torch.cat([cache.windows(r, bb.sr) for r in rows_all])[: args.speed_batch]   # hepsi 4 s (pick_rows)
    t1 = time.time()
    for _ in range(2):
        bb.embed(batch)
    wps = 2 * len(batch) / (time.time() - t1)
    put("S13_speed", "INFO", windows_per_s=round(wps, 1), est_minutes_12k_windows=round(12000 / wps / 60, 1))
    # S14 dolgu yolu (D-015 / D-034 madde 1): (a) tam uzunlukta lengths verilmesi sonucu değiştirmemeli;
    # (b) 2.5 s'lik parça: 4 s'ye sıfır dolgulu (resmi maske + maskeli ortalama; PANNs maskesiz) vs dolgusuz
    full_len = torch.full((len(w),), w.shape[1], dtype=torch.long)
    d14 = float((bb.embed(w, lengths=full_len) - e1).abs().max())
    n_short = int(2.5 * bb.sr)
    xs = w[:1, :n_short].clone()
    xp = torch.zeros(1, w.shape[1])
    xp[0, :n_short] = xs[0]
    e_np, e_zp = bb.embed(xs), bb.embed(xp, lengths=torch.tensor([n_short]))
    cos = torch.nn.functional.cosine_similarity(e_np[0], e_zp[0], dim=-1)
    rel14 = d14 / (float(e1.abs().max()) + 1e-12)
    put("S14_padding_path", "PASS" if (d14 < 1e-4 or rel14 < 1e-4) else "FAIL", full_length_max_abs_diff=d14,
        short_nopad_vs_zeropad_cos_min=round(float(cos.min()), 4), short_nopad_vs_zeropad_cos_mean=round(float(cos.mean()), 4),
        mask="yok (PANNs resmi API)" if bb.family == "panns" else "resmi maske + maskeli ortalama")
    # S12 Faz 3 eğitim adımı (en son: model ağırlıklarını değiştirir)
    if args.skip_train:
        put("S12_train_step", "INFO", note="atlandı")
    else:
        tb = torch.cat([cache.windows(r, bb.sr) for r in rows_all])[: args.train_batch]
        try:
            r12 = train_probe(bb, tb, amp=True, grad_ckpt=False)
            ok12 = r12["loss_finite"] and r12["grads_finite"] and r12["param_changed"]
            put("S12_train_step", "PASS" if ok12 else "FAIL", **r12)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            note = {"oom_without_grad_ckpt": True}
            if bb.family == "wavlm":
                bb.model.zero_grad(set_to_none=True)
                r12 = train_probe(bb, tb, amp=True, grad_ckpt=True)
                ok12 = r12["loss_finite"] and r12["grads_finite"] and r12["param_changed"]
                put("S12_train_step", "PASS" if ok12 else "FAIL", **r12, **note)
            else:
                put("S12_train_step", "FAIL", **note)
    res["status"] = "FAIL" if any(v["status"] == "FAIL" for v in C.values()) else "PASS"
    res["elapsed_s"] = round(time.time() - t0, 1)
    del bb
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    return res


def to_md(rep: dict) -> str:
    title = ("# BİRİM TESTİ (rastgele küçük modeller) — SMK-001 DEĞİL; modellerin doğruluğu hakkında bilgi vermez"
             if rep["random_init"] else "# SMK-001 — Faz 2 smoke test, GERÇEK checkpoint'ler (otomatik üretildi; katılımcı ID'si içermez)")
    L = [title, "",
         f"- Ortam: {rep['env']}", f"- Cihaz: {rep['device']} · random_init: {rep['random_init']}", "",
         "| backbone | durum | S3 strict | S5 şekil | S6 det. | S7 batch | S8 kanca | S9 Speech sırası | S10 aynı kişi AUC | S11 | S12 eğitim (GB, s/adım) | S13 pencere/s | S14 dolgu (tam=; kısa cos min) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rep["results"]:
        if "error" in r:
            L.append(f"| {r['backbone']} | **HATA** | {r['error'][:80]} |" + " |" * 9)
            continue
        c = r["checks"]
        s8, s12 = c["S8_hook_vs_official"], c.get("S12_train_step", {})
        s8v = f"{s8['max_abs_diff']:.1e}" if "max_abs_diff" in s8 else "—"
        s12v = f"{s12.get('peak_gpu_gb', '—')}, {s12.get('step_s', '—')}" + (" (grad ckpt)" if s12.get("grad_checkpointing") else "")
        L.append("| {} | **{}** | {} | {} | {:.1e} | {:.1e} | {} | {} | {:.2f} | {} | {} | {} |".format(
            r["backbone"], r["status"], c["S3_strict_load"]["status"], "×".join(map(str, c["S5_forward"]["out_shape"])),
            c["S6_determinism"]["max_abs_diff"], c["S7_batch_invariance"]["max_abs_diff"], s8v,
            c["S9_speech_class"].get("speech_rank", "—"), c["S10_same_person"]["auc_same_vs_diff_person"],
            c["S11_save_load"]["status"], s12v, c["S13_speed"]["windows_per_s"])
                 + f" {c['S14_padding_path']['status']}; {c['S14_padding_path']['short_nopad_vs_zeropad_cos_min']} |")
    L += ["", "Ayrıntılar (parametre sayısı, sha256, yükleme yöntemi, config) JSON'da."]
    if rep["random_init"]:
        L += ["", "**Bu bir birim testidir.** Rastgele küçük modeller kod yollarını sınar; resmi ağırlıkların doğruluğunu göstermez. "
              "SMK-001'in yerine geçmez."]
    else:
        L += ["", "**Kural (docs/PHASE2_DESIGN.md 1.1):** FAIL olan bir backbone varsa gömme çıkarımı ve EXP-016 başlatılmaz. "
              "Önce neden incelenir (JSON'daki değerler), düzeltilir, EXPERIMENTS.md'ye yazılır ve SMK-001 baştan çalıştırılır. "
              "Eşik gevşetilerek geçirme yapılmaz. Bu rapor birim testlerinden (UNITTEST_random_init_smoke) ayrıdır."]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--model-dir", type=Path)
    ap.add_argument("--backbones", default=",".join(B.BACKBONES))
    ap.add_argument("--report-dir", required=True, type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-participants", type=int, default=8)
    ap.add_argument("--train-batch", type=int, default=16)
    ap.add_argument("--speed-batch", type=int, default=32)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--random-init", action="store_true", help="yalnız birim testi: ağırlıksız küçük modeller")
    args = ap.parse_args()
    B.check_vendored()
    B.set_determinism()
    cache = Cache(args.cache_dir)
    man_path = (args.model_dir / "MANIFEST.json") if args.model_dir else None
    manifest = json.loads(man_path.read_text()) if man_path and man_path.exists() else {}
    results = []
    for name in args.backbones.split(","):
        print(f"== {name}", flush=True)
        try:
            r = check_backbone(name, args, cache, manifest)
        except Exception as e:
            traceback.print_exc()
            r = {"backbone": name, "status": "FAIL", "error": f"{type(e).__name__}: {e}"}
        results.append(r)
        print(f"   {r['status']}  " + ", ".join(f"{k}:{v['status']}" for k, v in r.get("checks", {}).items()), flush=True)
    if man_path and not args.random_init:
        man_path.write_text(json.dumps(manifest, indent=2))
    import transformers
    import torchaudio
    tag = "UNITTEST_random_init" if args.random_init else "SMK-001"
    rep = {"exp": tag, "date": time.strftime("%Y-%m-%d"), "device": args.device,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
           "random_init": args.random_init,
           "env": {"python": platform.python_version(), "torch": torch.__version__, "torchaudio": torchaudio.__version__,
                   "transformers": transformers.__version__, "numpy": np.__version__},
           "cache_index_sha256": B.sha256_file(args.cache_dir / "index.csv"), "results": results}
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / f"{tag}_smoke.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str))
    md = to_md(rep)
    (args.report_dir / f"{tag}_smoke.md").write_text(md)
    print(md)
    if any(r["status"] == "FAIL" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
