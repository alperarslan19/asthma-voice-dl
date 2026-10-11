"""
DIAG-001 — Faz 3 kalibrasyonunda CNN14'ün "sonlu olmayan logit" hatasının teşhisi (GPU; ~5 dk). Sonuç üretmez.

GÖZLEM (2026-10-11, Colab T4): cnn14 kalibrasyonu 1. epoch'un iç doğrulamasında durdu ("satır 77: sonlu olmayan logit").
    Eğitim kaybı sonluydu (o assert önce çalışır). log-Mel ön işlemesi zaten fp32 (backbones._fp32_forward).
    SMK-002 T4 değerlendirmeyi fp32 yapmıştı → fp16 değerlendirme yolu gerçek CNN14 ağırlıklarıyla hiç sınanmamıştı.

HİPOTEZLER (birbirini dışlar; düzeltme hangisinin doğru olduğuna göre seçilir — eşleme D-043'te önceden yazıldı):
    H1  Eğitim durumu bozuldu: fp16 ileri geçişte bir aktivasyon taştı (> 65504). GradScaler o adımı atlar, ama BatchNorm
        koşu istatistikleri ileri geçişte zaten güncellenmiştir → tamponlar inf/NaN olur. Değerlendirme (eval modu,
        koşu istatistikleri) bu yüzden NaN verir. Belirti: eğitimden sonra sonlu olmayan tampon / parametre.
        → Düzeltme: CNN14 kolu fp32 eğitilir ve değerlendirilir (ARMS["cnn14"]["amp"] = False).
    H2  Durum sağlam, yalnız fp16 değerlendirme taşıyor (eval modunda koşu istatistikleriyle büyüyen aktivasyon).
        Belirti: tamponlar sonlu; aynı kayıt fp32'de sonlu, fp16'da değil.
        → Düzeltme: değerlendirme bütün kollarda fp32 (COMMON["eval_amp"] = False).
    H3  Başka bir şey (ör. önceden eğitilmiş ağırlıklarda zaten taşma, belirli kayıt). → Rapor incelenir, karar sonra.

NE YAPAR (aynı kalibrasyon ayrımı, aynı tohum, aynı eğitim döngüsü run_training):
    A  önceden eğitilmiş model: bütün iç doğrulama kayıtları fp16 → sonlu olmayan kayıt sayısı
    B  1 epoch eğitim; HER adımdan sonra parametre / tampon sonluluğu (ilk bozulan adım, o adımın kaybı ve ölçek)
    C  eğitim sonrası sonlu olmayan parametre / tampon adları
    D  bütün iç doğrulama kayıtları fp16 ve fp32 → sonlu olmayan kayıt sayıları
    E  ilk bozuk kayıtta modül modül çıktı: fp16'da ilk sonlu olmayan modül; fp32'de en büyük mutlak değerler (65504 sınırı)
ÇIKTI  reports/phase3/DIAG-001_<arm>_nonfinite.{md,json} (agrega; katılımcı ID'si yok) + ekrana özet
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backbones as B  # noqa: E402
import finetune as F  # noqa: E402
from smoke_test_backbones import HOP_S, WIN_S, Cache  # noqa: E402

FP16_MAX = 65504.0


def eval_rows(m, cache, rows, device, amp: bool) -> dict:
    """Her kayıt için: sonlu mu (bütün pencereler)."""
    bad = []
    for r in rows:
        try:
            F.predict_recordings(m, cache, [r], device, amp)
        except AssertionError:
            bad.append(int(r))
    return {"n_rows": len(rows), "n_nonfinite": len(bad), "first_bad_rows": bad[:5]}


def windows(cache, r, sr):
    x = cache.audio(int(r), sr)
    ws = B.window_starts(len(x), int(WIN_S * sr), int(HOP_S * sr))
    W = torch.zeros(len(ws), int(WIN_S * sr))
    L = torch.empty(len(ws), dtype=torch.long)
    for i, (s, n) in enumerate(ws):
        W[i, :n] = torch.from_numpy(x[s:s + n])
        L[i] = n
    return W[:F.COMMON["eval_batch"]], L[:F.COMMON["eval_batch"]]


@torch.no_grad()
def module_trace(m, W, L, device, amp: bool) -> list[dict]:
    """Yaprak modüllerin çıktısı: en büyük |değer| ve sonluluk (çağrı sırasıyla)."""
    out, hooks = [], []
    for name, mod in m.named_modules():
        if len(list(mod.children())) == 0:
            def h(_m, _i, o, name=name):
                t = o[0] if isinstance(o, (tuple, list)) else o
                if torch.is_tensor(t) and t.is_floating_point():
                    tf = t.detach().float()
                    fin = torch.isfinite(tf)
                    out.append({"module": name, "dtype": str(t.dtype).replace("torch.", ""), "finite": bool(fin.all()),
                                "max_abs": float(tf[fin].abs().max()) if fin.any() else float("nan")})
            hooks.append(mod.register_forward_hook(h))
    m.eval()
    try:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device.startswith("cuda")):
            m(W.to(device), L.to(device))
    finally:
        for hk in hooks:
            hk.remove()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="cnn14")
    ap.add_argument("--splits-dir", required=True, type=Path)
    ap.add_argument("--context", required=True, type=Path)
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--model-dir", type=Path)
    ap.add_argument("--report-dir", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-eval-rows", type=int, default=0, help="0 = bütün iç doğrulama kayıtları")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--random-init-test", action="store_true", help="yalnız birim testi")
    a = ap.parse_args()
    t0 = time.time()
    cache = Cache(a.cache_dir)
    cfg = dict(F.ARMS[a.arm], amp=True)                                   # hatanın oluştuğu yapılandırma: fp16
    fd = F.FoldData(a.splits_dir, a.context, cache, 0, 0, "calibrate")
    rows = fd.rows_eval if not a.max_eval_rows else fd.rows_eval[:a.max_eval_rows]
    seed = F.seed_for(0, 0)
    B.set_determinism(seed)
    m = F.build_model(a.arm, a.model_dir, a.device, a.random_init_test)
    R = {"exp": "DIAG-001", "arm": a.arm, "device": a.device, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
         "torch": torch.__version__, "epochs": a.epochs, "data": fd.summary}
    print("A: önceden eğitilmiş model, fp16 değerlendirme ...", flush=True)
    R["A_pretrained_fp16"] = eval_rows(m, cache, rows, a.device, True)
    print("  ", R["A_pretrained_fp16"], flush=True)
    first = {}

    def on_step(ep, step, loss, sc0, sc1):
        if not first:
            bad = F.nonfinite_state(m)
            if bad:
                first.update({"epoch": ep + 1, "step": step, "last_microbatch_loss": loss, "scale_before": sc0,
                              "scale_after": sc1, "n_bad": len(bad), "bad_first": bad[:10],
                              "bad_kinds": sorted({n.rsplit(".", 1)[-1] for n in bad})})
    print(f"B: {a.epochs} epoch eğitim (fp16, adım adım durum kontrolü) ...", flush=True)
    with tempfile.TemporaryDirectory() as td:
        try:
            out = F.run_training(m, fd, cache, cfg, a.epochs, seed, Path(td), a.device, amp=True, on_step=on_step)
            R["B_train"] = {"completed": True, "train_loss": out["log"].train_loss.round(4).tolist(),
                            "skipped_steps": out["log"].skipped_steps.tolist()}
        except AssertionError as e:                                         # beklenen: epoch sonu sonluluk kontrolü
            R["B_train"] = {"completed": False, "assert": str(e)[:400]}
    R["B_first_nonfinite_step"] = first or None
    print("  ", R["B_train"], "| ilk bozulma:", R["B_first_nonfinite_step"], flush=True)
    bad = F.nonfinite_state(m)
    R["C_nonfinite_state"] = {"n": len(bad), "names_first": bad[:15], "kinds": sorted({n.rsplit(".", 1)[-1] for n in bad})}
    print("C:", R["C_nonfinite_state"], flush=True)
    print("D: eğitim sonrası değerlendirme fp16 ve fp32 ...", flush=True)
    R["D_fp16"] = eval_rows(m, cache, rows, a.device, True)
    R["D_fp32"] = eval_rows(m, cache, rows, a.device, False)
    print("   fp16:", R["D_fp16"], "\n   fp32:", R["D_fp32"], flush=True)
    bad_row = (R["D_fp16"]["first_bad_rows"] or R["D_fp32"]["first_bad_rows"] or [None])[0]
    if bad_row is not None:
        W, L = windows(cache, bad_row, m.bb.sr)
        t16 = module_trace(m, W, L, a.device, True)
        t32 = module_trace(m, W, L, a.device, False)
        first16 = next((x for x in t16 if not x["finite"]), None)
        big32 = sorted([x for x in t32 if np.isfinite(x["max_abs"])], key=lambda x: -x["max_abs"])[:6]
        R["E_trace"] = {"row_is_first_eval_row": bool(bad_row == int(rows[0])), "fp16_first_nonfinite_module": first16,
                        "fp32_any_nonfinite": any(not x["finite"] for x in t32), "fp32_largest": big32,
                        "fp32_over_fp16_max": [x["module"] for x in t32 if np.isfinite(x["max_abs"]) and x["max_abs"] > FP16_MAX][:10]}
        print("E:", json.dumps(R["E_trace"], ensure_ascii=False), flush=True)
    # sınıflandırma (D-043'te önceden yazılmış eşleme)
    if R["C_nonfinite_state"]["n"] > 0 or R["B_first_nonfinite_step"]:
        verdict = "H1: eğitim durumu bozuldu (sonlu olmayan parametre / tampon) → CNN14 fp32 eğitim + değerlendirme"
    elif R["D_fp16"]["n_nonfinite"] > 0 and R["D_fp32"]["n_nonfinite"] == 0:
        verdict = "H2: durum sağlam, fp16 değerlendirme taşıyor → değerlendirme bütün kollarda fp32"
    elif R["D_fp16"]["n_nonfinite"] == 0 and R["D_fp32"]["n_nonfinite"] == 0:
        verdict = "yeniden üretilemedi (bu koşuda bozulma yok) → rapor incelenir; belirlenimcilik / epoch sayısı kontrol edilir"
    else:
        verdict = "H3: fp32'de de sonlu değil → rapor incelenir, karar sonra"
    R["verdict"] = verdict
    R["elapsed_s"] = round(time.time() - t0, 1)
    md = [f"# DIAG-001 — {a.arm}: sonlu olmayan logit teşhisi (otomatik üretildi; katılımcı ID'si içermez)", "",
          f"- Ortam: {R['gpu']} · torch {R['torch']} · {a.epochs} epoch · {len(rows)} iç doğrulama kaydı · {R['elapsed_s']} s",
          f"- A önceden eğitilmiş, fp16: {R['A_pretrained_fp16']['n_nonfinite']} / {len(rows)} kayıt sonlu değil",
          f"- B eğitim: {R['B_train']}",
          f"- B ilk bozulan adım: {R['B_first_nonfinite_step']}",
          f"- C eğitim sonrası sonlu olmayan parametre / tampon: {R['C_nonfinite_state']['n']} (türler: {R['C_nonfinite_state']['kinds']})",
          f"- D eğitim sonrası: fp16 {R['D_fp16']['n_nonfinite']} / {len(rows)}, fp32 {R['D_fp32']['n_nonfinite']} / {len(rows)} kayıt sonlu değil"]
    if "E_trace" in R:
        md.append(f"- E fp16'da ilk sonlu olmayan modül: {R['E_trace']['fp16_first_nonfinite_module']}; "
                  f"fp32'de 65504'ü aşan modüller: {R['E_trace']['fp32_over_fp16_max']}")
    md += ["", f"**Okuma (D-043'te önceden yazılmış eşleme):** {verdict}"]
    a.report_dir.mkdir(parents=True, exist_ok=True)
    (a.report_dir / f"DIAG-001_{a.arm}_nonfinite.md").write_text("\n".join(md) + "\n")
    (a.report_dir / f"DIAG-001_{a.arm}_nonfinite.json").write_text(json.dumps(R, indent=1, ensure_ascii=False, default=str))
    print("\n".join(md))


if __name__ == "__main__":
    main()
