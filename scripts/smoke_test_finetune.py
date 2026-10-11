"""
Faz 3 — SMK-002: fine-tuning smoke testi (GPU; gerçek ağırlıklar, gerçek önbellek). Kalibrasyon ve eğitimden ÖNCE zorunlu.

Her kol için (cnn10, cnn10_scratch, cnn14, beats, wavlm_base_plus, wavlm_large):
    T1 split / sızıntı: gün-gruplu split'in (tekrar 0, fold 0) FoldData kapıları geçiyor (katılımcı ve gün ayrık)
    T2 batch: 4 s parçalar (B, win) float32; kısa kayıt dolgu + uzunluk; ileri geçiş (B,) sonlu logit (fp16 autocast)
    T3 eğitim: gerçek eğitim döngüsüyle (run_training) 2 küçük epoch (64 kayıt = 4 adım / epoch): kayıp sonlu,
       gövde ve baş parametreleri değişiyor, atlanan adım < adım; checkpoint yazılıyor
    T4 değerlendirme: 8 değerlendirme kaydının bütün pencereleri → sonlu kayıt logit'i (fp32 VE fp16; D-043)
    T5 kurtarma: checkpoint'ten kurulan yeni model aynı tahmini veriyor (fp32, en büyük fark < 1e-4)
    T6 bellek ve süre: tepe GPU belleği < 14.5 GB (T4); adım süresi → tam epoch ve fold süresi tahmini (bütçe)
Bir kol FAIL ise o kolun kalibrasyonu ve eğitimi başlatılmaz (notebook 25 durur). Eşik gevşetilerek geçirilmez.
Bu test protokol kapısına (D-042) tabi değildir: sonuç üretmez, yalnız hattı sınar.

ÇIKTI   reports/phase3/SMK-002_finetune.{md,json} (yalnız agrega; katılımcı ID'si yok)
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backbones as B  # noqa: E402
import finetune as F  # noqa: E402
from smoke_test_backbones import WIN_S, Cache  # noqa: E402

GPU_LIMIT_GB = 14.5


def smoke_arm(arm: str, args, cache: Cache) -> dict:
    res = {"arm": arm, "checks": {}}
    Ck = res["checks"]
    cfg = F.ARMS[arm]
    fd = F.FoldData(args.splits_dir, args.context, cache, 0, 0, "cv")
    Ck["T1_split"] = {"status": "PASS", **fd.summary}
    full_steps = math.ceil(len(fd.rows_train) / F.COMMON["batch"])
    n_eval_full = len(fd.rows_eval)
    rng = np.random.default_rng(0)
    small = copy.copy(fd)
    is_short = cache.idx.n_16k.to_numpy() < WIN_S * 16000
    tr_short = [r for r in fd.rows_train if is_short[r]][:4]
    small.rows_train = np.r_[tr_short, rng.choice([r for r in fd.rows_train if not is_short[r]], 64 - len(tr_short), replace=False)]
    small.rows_eval = rng.choice(fd.rows_eval, 8, replace=False)
    B.set_determinism(0)
    m = F.build_model(arm, args.model_dir, args.device, args.random_init_test)
    res["n_params"] = int(sum(p.numel() for p in m.parameters()))
    res["n_trainable"] = int(sum(p.numel() for p in m.parameters() if p.requires_grad))
    # T2
    wav, lengths = F.crop_batch(cache, small.rows_train[:cfg["micro_batch"]], np.full(cfg["micro_batch"], 0.5), m.bb.sr)
    assert wav.shape == (cfg["micro_batch"], int(WIN_S * m.bb.sr)) and wav.dtype == torch.float32
    m.train()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.device.startswith("cuda")):
        lg = m(wav.to(args.device), lengths.to(args.device))
    ok2 = lg.shape == (cfg["micro_batch"],) and bool(torch.isfinite(lg).all())
    Ck["T2_forward"] = {"status": "PASS" if ok2 else "FAIL", "wav": list(wav.shape), "n_short": int((lengths < wav.shape[1]).sum()),
                        "logit_shape": list(lg.shape), "device": str(lg.device)}
    # T3 + T4
    body0 = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad and not n.startswith("head")}
    head0 = {n: p.detach().clone() for n, p in m.named_parameters() if n.startswith("head")}
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        out = F.run_training(m, small, cache, dict(cfg, ckpt_every=1), 2, F.seed_for(0, 0), wd, args.device,
                             eval_each_epoch=True)
        log = out["log"]
        body_changed = sum(not torch.equal(body0[n], p.detach()) for n, p in m.named_parameters() if n in body0)
        head_changed = sum(not torch.equal(head0[n], p.detach()) for n, p in m.named_parameters() if n in head0)
        ok3 = (bool(np.isfinite(log.train_loss).all()) and body_changed > 0 and head_changed > 0
               and int(log.skipped_steps.max()) < int(log.steps.min()) and (wd / "latest.pt").exists())
        Ck["T3_train"] = {"status": "PASS" if ok3 else "FAIL", "train_loss": log.train_loss.round(4).tolist(),
                          "val_loss": log.val_loss.round(4).tolist(), "skipped_steps": log.skipped_steps.tolist(),
                          "body_params_changed": f"{body_changed}/{len(body0)}", "head_params_changed": f"{head_changed}/{len(head0)}"}
        t0 = time.time()
        P1 = F.predict_recordings(m, cache, small.rows_eval, args.device, amp=False)
        eval_s_per_rec = (time.time() - t0) / len(small.rows_eval)
        try:                                          # D-043: eğitimde kullanılan fp16 değerlendirme yolu da sınanır (CAL-001'de
            P16 = F.predict_recordings(m, cache, small.rows_eval, args.device, amp=True)   # CNN14 burada durdu)
            fp16_ok, fp16_diff = True, float(np.abs(P16.logit.to_numpy() - P1.logit.to_numpy()).max())
        except AssertionError as e:
            fp16_ok, fp16_diff = False, str(e)[:200]
        Ck["T4_eval"] = {"status": "PASS" if np.isfinite(P1.logit).all() and len(P1) == 8 and fp16_ok else "FAIL",
                         "windows_per_recording_mean": float(P1.n_windows.mean()), "fp16_eval_finite": fp16_ok,
                         "fp16_vs_fp32_max_abs_diff": fp16_diff}
        # T5
        st = torch.load(wd / "latest.pt", map_location="cpu", weights_only=False)
        B.set_determinism(0)
        m2 = F.build_model(arm, args.model_dir, args.device, args.random_init_test)
        info = m2.load_state_dict(st["model"], strict=False)
        P2 = F.predict_recordings(m2, cache, small.rows_eval, args.device, amp=False)
        d = float(np.abs(P1.logit.to_numpy() - P2.logit.to_numpy()).max())
        Ck["T5_resume"] = {"status": "PASS" if d < 1e-4 and not info.unexpected_keys else "FAIL", "max_abs_diff": d,
                           "checkpoint_mb": round((wd / "latest.pt").stat().st_size / 1e6, 1)}
        del m2
    # T6
    s_step = float(log.epoch_s.iloc[-1]) / int(log.steps.iloc[-1])           # 2. epoch (ısınma sonrası); değerlendirme hariç değil
    peak = float(log.gpu_peak_gb.max())
    E = cfg["max_epochs"]
    est_epoch_min = s_step * full_steps / 60
    est_fold_min = E * est_epoch_min + eval_s_per_rec * n_eval_full / 60
    ok6 = peak < GPU_LIMIT_GB or not args.device.startswith("cuda")
    Ck["T6_budget"] = {"status": "PASS" if ok6 else "FAIL", "gpu_peak_gb": peak, "s_per_step_incl_val8": round(s_step, 3),
                       "steps_per_epoch_full": full_steps, "est_epoch_min": round(est_epoch_min, 2),
                       "est_fold_min_at_max_epochs": round(est_fold_min, 1), "max_epochs": E, "micro_batch": cfg["micro_batch"]}
    res["status"] = "PASS" if all(v["status"] == "PASS" for v in Ck.values()) else "FAIL"
    del m
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="cnn10,cnn10_scratch,cnn14,beats,wavlm_base_plus,wavlm_large")
    ap.add_argument("--splits-dir", required=True, type=Path)
    ap.add_argument("--context", required=True, type=Path)
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--model-dir", type=Path)
    ap.add_argument("--report-dir", required=True, type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--random-init-test", action="store_true", help="yalnız birim testi")
    args = ap.parse_args()
    cache = Cache(args.cache_dir)
    env = {"torch": torch.__version__, "device": args.device, "random_init_test": args.random_init_test,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    results = []
    for arm in [a for a in args.arms.split(",") if a]:
        print(f"== {arm}", flush=True)
        try:
            r = smoke_arm(arm, args, cache)
        except Exception as e:                                               # FAIL de raporlanır (olumsuz sonuç korunur)
            r = {"arm": arm, "status": "FAIL", "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:], "checks": {}}
        print(json.dumps({k: v for k, v in r.items() if k != "trace"}, ensure_ascii=False, default=str)[:1500], flush=True)
        results.append(r)
    rep = {"exp": "SMK-002", "env": env, "protocol": F.PHASE3, "common": F.COMMON, "results": results,
           "code_sha256": F.sha256_short(Path(F.__file__)), "date": time.strftime("%Y-%m-%d")}
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "SMK-002_finetune.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str))
    L = ["# SMK-002 — Faz 3 fine-tuning smoke testi (otomatik üretildi; katılımcı ID'si içermez)", "",
         f"- Ortam: {env}", f"- Protokol: {F.PHASE3['version']} — {F.PHASE3['status']}", "",
         "| kol | durum | T2 ileri | T3 eğitim (kayıp e1→e2; değişen gövde) | T4 değ. | T5 kurtarma (fark; ckpt MB) | GPU tepe GB | s/adım | tahmini epoch / fold (dk, en çok epoch) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        c = r.get("checks", {})
        if r["status"] == "FAIL" and not c.get("T6_budget"):
            L.append(f"| {r['arm']} | **FAIL** | {r.get('error', '—')[:120]} | | | | | | |")
            continue
        t3, t5, t6 = c["T3_train"], c["T5_resume"], c["T6_budget"]
        L.append(f"| {r['arm']} | **{r['status']}** | {c['T2_forward']['status']} | {t3['status']} ({t3['train_loss'][0]:.3f}→{t3['train_loss'][-1]:.3f}; "
                 f"{t3['body_params_changed']}) | {c['T4_eval']['status']} | {t5['status']} ({t5['max_abs_diff']:.1e}; {t5['checkpoint_mb']}) | "
                 f"{t6['gpu_peak_gb']} | {t6['s_per_step_incl_val8']} | {t6['est_epoch_min']} / {t6['est_fold_min_at_max_epochs']} ({t6['max_epochs']}) |")
    L += ["", "Süre tahmini: 2. küçük epoch'un adım süresinden (8 kayıtlık izleme değerlendirmesi dahil) × tam epoch adım sayısı; "
          "fold = en çok epoch × epoch + test değerlendirmesi. Kalibrasyondan sonra gerçek epoch sayısıyla güncellenir. [INFERENCE]",
          "", "**Kural:** FAIL olan kolun kalibrasyonu ve eğitimi başlatılmaz; neden incelenir, düzeltilir, SMK-002 yeniden çalıştırılır."]
    (args.report_dir / "SMK-002_finetune.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    if any(r["status"] != "PASS" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
