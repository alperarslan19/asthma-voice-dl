"""
Faz 3 (scripts/finetune.py, make_splits --group-by-day, run_probes EXP-016DG) için sentetik testler.
Gerçek veri ve gerçek ağırlık GEREKMEZ; küçük rastgele modellerle (random_init) bütün kod yolları sınanır.
Bu testlerin geçmesi modellerin gerçek ağırlıklarla doğru eğitileceğini göstermez → GPU'da SMK-002.

    python tests/test_phase3.py

    F1  gün-gruplu split: hiçbir gün hem train hem testte (dış ve iç); her katılımcı tekrar başına bir kez testte; belirlenimci
    F2  FoldData sızıntı kapıları: gün paylaşan (katılımcı-gruplu) split reddedilir; kalibrasyon kipi dış testi hiç görmez
    F3  v1 epoch kuralı (choose_epochs; kayıt için): hareketli ortalama, kenar, eşitlik
    F4  protokol kapısı: status KABUL değilken kalibrasyon / eğitim reddedilir (koddaki statüden bağımsız sınanır);
        KABUL iken kapı geçer
    F5  kalibrasyon → sabit epoch; eğitim → preds.csv + fold.json; bitmiş fold yeniden çalıştırmada atlanır;
        bütün test katılımcıları tahmin alır, eğitim katılımcıları almaz
    F6  kesinti + devam: 1 epoch sonra durdurulan eğitim, kaldığı yerden sürünce kesintisiz eğitimle AYNI tahminleri verir (CPU)
    F7  dört aile de eğitilebiliyor (PANNs CNN10, BEATs, WavLM Base+ biçimi, WavLM Large biçimi): kayıp sonlu, parametreler
        değişiyor, kısa kayıt (dolgu + maske) yolu çalışıyor; donuk WavLM özellik kodlayıcısı değişmiyor
    F8  --arm ssl: fold başına D-035 seçimi okunuyor
    F9  öğrenilebilirlik: yerleştirilmiş sinyal (astımda 300 Hz ton) birkaç epoch'ta öğreniliyor (eğitim kaybı düşüyor)
    F10 SMK-002 CPU'da (küçük rastgele modeller) bütün kontrolleri geçiyor ve rapor yazıyor
    F11 G1'in mantığı, doğrusu bilinen veriyle: gömme YALNIZ kayıt gününün izini taşıyor (astım bilgisi yok; günlerin çoğu
        tek etiketli). Katılımcı-gruplu CV'de T1 yanlış pozitif veriyor (gün eşleri eğitimde); gün-gruplu CV'de vermiyor.
        Yani gün-gruplu kontrol, korumak istediği mekanizmayı gerçekten yakalıyor.
    F13 D-043 epoch kuralı (iç doğrulama AUC'si, 3'lü hareketli ort., 1-SE; Hanley–McNeil SE) ve ısınmanın toplam epoch'tan
        bağımsızlığı (1 epoch'luk eğitim = 20 epoch'luk kalibrasyonun ilk epoch'u); eğitim epoch'u eğrilerden hesaplanır
    F14 sonlu olmayan parametre / tampon yakalanır ve adıyla raporlanır; teşhis betiği (DIAG-001) CPU'da çalışır
    F12 --ckpt-root (Colab yerel disk): latest.pt yalnız orada, Drive klasöründe hiç yok; kesinti + devam aynı tahmini veriyor;
        yerel checkpoint kaybolursa (makine değişti) fold epoch 0'dan yeniden başlıyor ve yine aynı tahmini veriyor
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import finetune as F  # noqa: E402
from smoke_test_backbones import Cache  # noqa: E402

N_PART, N_DAYS = 30, 15


def run(*cmd, ok=True):
    r = subprocess.run([sys.executable, *map(str, cmd)], capture_output=True, text=True)
    if ok:
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r


def build_fixture(root: Path, seed: int = 0) -> dict:
    """30 katılımcı × 7 kayıt (2.5–6 s; bir kısmı 4 s'den kısa), 15 kayıt günü. Astımda 300 Hz ton (öğrenilebilir sinyal)."""
    rng = np.random.default_rng(seed)
    pid = np.arange(800001, 800001 + N_PART)
    label = np.r_[np.ones(18, int), np.zeros(12, int)]
    rng.shuffle(label)
    day = np.arange(N_PART) % N_DAYS
    C = pd.DataFrame({"participant_id": pid, "label": label, "age": rng.integers(20, 65, N_PART),
                      "recording_date": [str((pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(d))).date()) for d in day],
                      "start_hour": rng.uniform(8, 17, N_PART).round(2), "has_audio": True})
    C.to_csv(root / "ctx.csv", index=False)
    rows, a16, a32, o16, o32 = [], [], [], 0, 0
    for p, y in zip(pid, label):
        for s in range(1, 8):
            dur = float(rng.choice([2.5, 3.5, 5.0, 6.0]))
            n32 = int(dur * 32000)
            t = np.arange(n32) / 32000
            x = 0.05 * rng.normal(size=n32) + (0.2 * np.sin(2 * np.pi * 300 * t) if y else 0.0)
            x = x.astype(np.float32)
            x16 = x[::2].copy()
            rows.append({"participant_id": p, "slot": s, "offset_32k": o32, "n_32k": len(x), "offset_16k": o16, "n_16k": len(x16)})
            a32.append(x)
            a16.append(x16)
            o32 += len(x)
            o16 += len(x16)
    cache = root / "cache"
    cache.mkdir()
    pd.DataFrame(rows).to_csv(cache / "index.csv", index=False)
    np.concatenate(a32).astype("<f4").tofile(cache / "audio_32k.f32")
    np.concatenate(a16).astype("<f4").tofile(cache / "audio_16k.f32")
    pd.DataFrame([{"participant_id": p, "slot": s, "status": "OK"} for p in pid for s in range(1, 8)]).to_csv(root / "rm.csv", index=False)
    run(REPO / "scripts/make_splits.py", "--context", root / "ctx.csv", "--recording-map", root / "rm.csv",
        "--out-dir", root / "splits_day", "--report-dir", root / "splits_report", "--n-repeats", "3", "--group-by-day")
    run(REPO / "scripts/make_splits.py", "--context", root / "ctx.csv", "--recording-map", root / "rm.csv",
        "--out-dir", root / "splits_part", "--report-dir", root / "splits_report_p", "--n-repeats", "1")
    return {"ctx": root / "ctx.csv", "cache": cache, "splits": root / "splits_day", "splits_part": root / "splits_part"}


def ft_args(fx: dict, out: Path, *extra) -> list:
    return [REPO / "scripts/finetune.py", *extra[:1], "--splits-dir", fx["splits"], "--context", fx["ctx"], "--cache-dir", fx["cache"],
            "--out-root", out, "--device", "cpu", "--allow-unapproved", "--random-init-test", *extra[1:]]


def test_day_splits(fx) -> None:
    C = pd.read_csv(fx["ctx"]).set_index("participant_id")
    for r in range(3):
        S = pd.read_csv(fx["splits"] / f"outer_r{r}.csv")
        assert S[S.role == "test"].participant_id.is_unique and S[S.role == "test"].participant_id.nunique() == N_PART
        for k, b in S.groupby("outer_fold"):
            tr, te = b[b.role == "train"].participant_id, b[b.role == "test"].participant_id
            assert not set(C.loc[tr, "recording_date"]) & set(C.loc[te, "recording_date"]), f"r{r} k{k}: gün sızıntısı"
    man = json.loads((fx["splits"].parent / "splits_report" / "splits_manifest.json").read_text())
    assert "gün" in man["grouping"] and man["n_day_groups"] == N_DAYS
    print("  F1 geçti (gün-gruplu split: dış ve iç fold'larda gün sızıntısı yok)", flush=True)


def test_fold_gates(fx) -> None:
    cache = Cache(fx["cache"])
    try:
        F.FoldData(fx["splits_part"], fx["ctx"], cache, 0, 0, "cv")
    except AssertionError as e:
        assert "GÜN SIZINTISI" in str(e), e
    else:
        raise AssertionError("F2: katılımcı-gruplu split (gün paylaşan) kabul edildi")
    fd = F.FoldData(fx["splits"], fx["ctx"], cache, 0, 0, "calibrate")
    S = pd.read_csv(fx["splits"] / "outer_r0.csv")
    test_ids = set(S[(S.outer_fold == 0) & (S.role == "test")].participant_id)
    seen = set(fd.pid_of_row[np.r_[fd.rows_train, fd.rows_eval]])
    assert not seen & test_ids, "F2: kalibrasyon dış testi gördü"
    assert abs(fd.pos_weight - fd.n_neg / fd.n_pos) < 1e-12
    print(f"  F2 geçti (gün paylaşan split reddedildi; kalibrasyon dış testi görmüyor; {fd.summary['n_days_train']} / "
          f"{fd.summary['n_days_eval']} gün)", flush=True)


def test_choose_epochs() -> None:
    assert F.choose_epochs([5, 4, 3, 2, 1], 3) == 5                        # hâlâ düşüyor → kenar
    assert F.choose_epochs([1.0, 0.8, 0.6, 0.7, 0.9, 1.2], 3) == 3
    assert F.choose_epochs([1.0, 0.5, 2.0, 0.5, 0.5, 0.5], 3) == 5          # tek düşük nokta değil, plato
    assert F.choose_epochs([1.0, 1.0, 1.0], 3) == 1                         # eşitlikte küçük
    print("  F3 geçti (epoch kuralı)", flush=True)


def test_gate(fx, work: Path) -> None:
    saved = F.PHASE3["status"]
    try:
        F.PHASE3["status"] = "ÖNERİLDİ"
        try:
            F.protocol_gate(False)
        except SystemExit as e:
            assert "onaylanmadı" in str(e)
        else:
            raise AssertionError("F4: onaysız protokolde kapı geçti")
        F.protocol_gate(True)                                                # yalnız testler: --allow-unapproved
        F.PHASE3["status"] = "KABUL"
        F.protocol_gate(False)
    finally:
        F.PHASE3["status"] = saved
    if saved != "KABUL":                                                     # uçtan uca: komut satırı da reddediyor
        r = run(*[a for a in ft_args(fx, work / "gate", "calibrate", "--arm", "cnn10") if a != "--allow-unapproved"], ok=False)
        assert r.returncode != 0 and "onaylanmadı" in (r.stdout + r.stderr), r.stderr[-1000:]
        assert not (work / "gate").exists()
    print(f"  F4 geçti (kapı: ÖNERİLDİ → reddedildi, KABUL → geçti; koddaki statü {saved})", flush=True)


def test_calibrate_train_resume(fx, work: Path) -> None:
    out = work / "ft"
    run(*ft_args(fx, out, "calibrate", "--arm", "beats", "--max-epochs", "3"))
    cal = json.loads((out / "beats" / "calibration" / "calibration.json").read_text())
    assert 1 <= cal["epochs"] <= 3 and len(cal["val_loss"]) == 3 and not (out / "beats" / "calibration" / "latest.pt").exists()
    assert cal["epochs"] == F.select_epochs(cal)["epochs"] and cal["selection"]["epochs_v1_loss"] == F.choose_epochs(cal["val_loss"], 3)
    # eğitim, epoch'u kayıttaki sayıdan değil eğrilerden hesaplar (D-043): kayıttaki sayıyı boz → yine eğri kuralı
    cal_bad = dict(cal, epochs=99)
    (out / "beats" / "calibration" / "calibration.json").write_text(json.dumps(cal_bad))
    run(*ft_args(fx, out, "train", "--arm", "beats", "--repeats", "0", "--folds", "1"))
    wd = out / "beats" / "r0_k1"
    P0 = pd.read_csv(wd / "preds.csv")
    S = pd.read_csv(fx["splits"] / "outer_r0.csv")
    b = S[S.outer_fold == 1]
    assert set(P0.participant_id) == set(b[b.role == "test"].participant_id), "F5: test katılımcıları eksik / fazla"
    assert not set(P0.participant_id) & set(b[b.role == "train"].participant_id)
    assert np.isfinite(P0.logit).all() and (P0.n_windows >= 1).all() and not (wd / "latest.pt").exists()
    fj = json.loads((wd / "fold.json").read_text())
    assert fj["epochs"] == F.select_epochs(cal)["epochs"], "F5: eğitim epoch'u D-043 kuralından gelmedi"
    r2 = run(*ft_args(fx, out, "train", "--arm", "beats", "--repeats", "0", "--folds", "1"))
    assert "atlandı" in r2.stdout
    r3 = run(REPO / "scripts/finetune.py", "select", "--out-root", out, "--report-dir", work / "sel_rep")
    assert "beats" in r3.stdout and (work / "sel_rep" / "CAL-001_epoch_selection.md").exists()
    print(f"  F5 geçti (kalibrasyon → {fj['epochs']} epoch, eğrilerden; {len(P0)} kayıt tahmini; bitmiş fold atlandı; select raporu)",
          flush=True)
    # F6 / F12: devam testleri için sabit 3 epoch (yalnız testler: --force-epochs)
    FE = ("--force-epochs", "3")
    ref = work / "ft_ref"
    shutil.copytree(out / "beats" / "calibration", ref / "beats" / "calibration")
    run(*ft_args(fx, ref, "train", "--arm", "beats", "--repeats", "0", "--folds", "1", *FE))
    P = pd.read_csv(ref / "beats" / "r0_k1" / "preds.csv")
    out2 = work / "ft_resume"
    shutil.copytree(out / "beats" / "calibration", out2 / "beats" / "calibration")
    r = run(*ft_args(fx, out2, "train", "--arm", "beats", "--repeats", "0", "--folds", "1", *FE, "--stop-after-epoch", "1"))
    assert "durduruldu" in r.stdout and (out2 / "beats" / "r0_k1" / "latest.pt").exists()
    r = run(*ft_args(fx, out2, "train", "--arm", "beats", "--repeats", "0", "--folds", "1", *FE))
    assert "devam: epoch 1" in r.stdout
    P2 = pd.read_csv(out2 / "beats" / "r0_k1" / "preds.csv")
    d = np.abs(P.sort_values("row").logit.to_numpy() - P2.sort_values("row").logit.to_numpy()).max()
    assert d < 1e-5, f"F6: devam eden koşu farklı tahmin verdi (en büyük fark {d})"
    print(f"  F6 geçti (kesinti + devam = kesintisiz; en büyük fark {d:.1e})", flush=True)
    # F12: checkpoint ayrı (yerel) klasörde
    out3, ckr = work / "ft_local", work / "local_ckpt"
    shutil.copytree(out / "beats" / "calibration", out3 / "beats" / "calibration")
    a3 = ("--repeats", "0", "--folds", "1", "--ckpt-root", ckr, *FE)
    r = run(*ft_args(fx, out3, "train", "--arm", "beats", *a3, "--stop-after-epoch", "1"))
    assert (ckr / "beats" / "r0_k1" / "latest.pt").exists() and not list(out3.rglob("*.pt")), "F12: checkpoint Drive klasörüne yazıldı"
    r = run(*ft_args(fx, out3, "train", "--arm", "beats", *a3))
    assert "devam: epoch 1" in r.stdout and not (ckr / "beats" / "r0_k1" / "latest.pt").exists()
    P3 = pd.read_csv(out3 / "beats" / "r0_k1" / "preds.csv")
    d3 = np.abs(P.sort_values("row").logit.to_numpy() - P3.sort_values("row").logit.to_numpy()).max()
    # makine değişti: yerel checkpoint kayboldu, Drive'daki günlük duruyor
    out4, ckr4 = work / "ft_lost", work / "local_ckpt4"
    shutil.copytree(out / "beats" / "calibration", out4 / "beats" / "calibration")
    a4 = ("--repeats", "0", "--folds", "1", "--ckpt-root", ckr4, *FE)
    run(*ft_args(fx, out4, "train", "--arm", "beats", *a4, "--stop-after-epoch", "1"))
    shutil.rmtree(ckr4)
    r = run(*ft_args(fx, out4, "train", "--arm", "beats", *a4))
    assert "epoch 0'dan yeniden" in r.stdout and "devam:" not in r.stdout
    P4 = pd.read_csv(out4 / "beats" / "r0_k1" / "preds.csv")
    lg = pd.read_csv(out4 / "beats" / "r0_k1" / "train_log.csv")
    assert lg.epoch.tolist() == [1, 2, 3], "F12: yeniden başlayan fold'un günlüğü tutarsız"
    d4 = np.abs(P.sort_values("row").logit.to_numpy() - P4.sort_values("row").logit.to_numpy()).max()
    assert d3 < 1e-5 and d4 < 1e-5, f"F12: yerel checkpoint yolu farklı tahmin verdi ({d3}, {d4})"
    print(f"  F12 geçti (--ckpt-root: Drive klasöründe .pt yok; devam fark {d3:.1e}; checkpoint kaybında baştan fark {d4:.1e})",
          flush=True)
    # --force-epochs onaysız koşuda kullanılamaz
    r = run(*[a for a in ft_args(fx, work / "fx_no", "train", "--arm", "beats", *FE) if a != "--allow-unapproved"], ok=False)
    assert r.returncode != 0


def test_selection_rule_and_warmup() -> None:
    """F13: D-043 epoch kuralı (AUC 1-SE) ve ısınmanın toplam epoch'tan bağımsızlığı."""
    assert abs(F.hanley_se(0.8, 46, 10) - 0.0659) < 5e-4                  # Hanley & McNeil (1982) formülü, elle hesap
    cal = {"val_auc_monitor_only": [0.6, 0.7, 0.8, 0.85, 0.86, 0.86, 0.85], "val_loss": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
           "data": {"eval_participants_asthma_healthy": [46, 10]}}
    s = F.select_epochs(cal)
    assert s["auc_ma_argmax"] == 5 and s["epochs"] == 4 and s["epochs_v1_loss"] == 1, s   # MA: .65 .70 .78 .84 .86 .86 .86
    flat = dict(cal, val_auc_monitor_only=[0.70, 0.72, 0.71, 0.73, 0.72])
    assert F.select_epochs(flat)["epochs"] == 1                            # gürültü içinde düz eğri → en tutumlu seçim
    big = dict(cal, data={"eval_participants_asthma_healthy": [4000, 4000]})
    assert F.select_epochs(big)["epochs"] == 5                            # SE → 0: 1-SE kuralı argmax'a iner
    lrs = {}
    for total in (1, 20):                                                  # 95 adım / epoch, 2 epoch ısınma
        opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
        sch = F.make_scheduler(opt, 95, 2)
        seq = []
        for _ in range(95 * total):
            seq.append(opt.param_groups[0]["lr"]); opt.step(); sch.step()
        lrs[total] = seq
    assert lrs[1] == lrs[20][:95], "ısınma toplam epoch'a bağlı (v1 hatası)"
    assert abs(lrs[20][94] - 95 / 190) < 1e-9 and lrs[20][189] == 1.0 and lrs[20][-1] == 1.0
    assert all(c["warmup_epochs"] in (0, 2) for c in F.ARMS.values())
    print("  F13 geçti (AUC 1-SE kuralı; v1 kaydı; ısınma toplam epoch'tan bağımsız: 1 epoch'luk yol = 20 epoch'luğun başı)", flush=True)


def test_nonfinite(fx, work: Path) -> None:
    """F14: sonlu olmayan durum yakalanır ve açıklanır (bastırılmaz); teşhis betiği CPU'da rapor yazar."""
    cache = Cache(fx["cache"])
    fd = F.FoldData(fx["splits"], fx["ctx"], cache, 0, 0, "cv")
    torch.manual_seed(0)
    m = F.build_model("cnn10", None, "cpu", random_init_test=True)
    assert F.nonfinite_state(m) == []
    bn = next(n for n, b in m.named_buffers() if n.endswith("running_var"))
    dict(m.named_buffers())[bn].fill_(float("nan"))
    assert bn in F.nonfinite_state(m)
    try:
        F.predict_recordings(m, cache, fd.rows_eval[:2], "cpu", amp=False)
    except AssertionError as e:
        assert "sonlu olmayan logit" in str(e) and bn in str(e), e
    else:
        raise AssertionError("F14: bozuk tampon değerlendirmede yakalanmadı")
    rep = work / "diag"
    run(REPO / "scripts/diagnose_nonfinite.py", "--arm", "cnn10", "--splits-dir", fx["splits"], "--context", fx["ctx"],
        "--cache-dir", fx["cache"], "--report-dir", rep, "--device", "cpu", "--random-init-test", "--max-eval-rows", "6")
    R = json.loads((rep / "DIAG-001_cnn10_nonfinite.json").read_text())
    assert R["C_nonfinite_state"]["n"] == 0 and "yeniden üretilemedi" in R["verdict"], R["verdict"]
    print(f"  F14 geçti (bozuk BatchNorm tamponu adıyla raporlandı; teşhis betiği: {R['verdict'][:40]}…)", flush=True)


def test_families(fx) -> None:
    cache = Cache(fx["cache"])
    fd = F.FoldData(fx["splits"], fx["ctx"], cache, 0, 0, "cv")
    rows = fd.rows_train[:6]
    for arm in ["cnn10", "beats", "wavlm_base_plus", "wavlm_large"]:
        torch.manual_seed(0)
        m = F.build_model(arm, None, "cpu", random_init_test=True)
        m.train()
        frozen = {n: p.detach().clone() for n, p in m.named_parameters() if not p.requires_grad}
        body = [(n, p) for n, p in m.named_parameters() if p.requires_grad and not n.startswith("head")]
        before = {n: p.detach().clone() for n, p in body}
        opt = F.make_optimizer(m, F.ARMS[arm])
        short = [r for r in fd.rows_train if cache.idx.iloc[int(r)].n_16k < 4 * 16000][:2]
        long_ = [r for r in fd.rows_train if cache.idx.iloc[int(r)].n_16k >= 4 * 16000][:2]
        wav, lengths = F.crop_batch(cache, np.r_[short, long_], np.full(4, 0.3), m.bb.sr)
        assert (lengths < wav.shape[1]).sum() == len(short) and wav.dtype == torch.float32
        y = torch.tensor([1.0, 0.0, 1.0, 0.0])
        for _ in range(2):
            opt.zero_grad()
            logit = m(wav, lengths)
            assert logit.shape == (4,) and torch.isfinite(logit).all()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y)
            loss.backward()
            opt.step()
        changed = sum(not torch.equal(before[n], p) for n, p in body)
        assert changed > 0.5 * len(body), f"F7 {arm}: gövde parametrelerinin çoğu değişmedi ({changed}/{len(body)})"
        assert all(torch.equal(frozen[n], p) for n, p in m.named_parameters() if n in frozen), f"F7 {arm}: donuk parametre değişti"
        if arm.startswith("wavlm"):
            assert frozen and all("feature_extractor" in n for n in frozen), f"F7 {arm}: donuk küme beklenmedik"
        P = F.predict_recordings(m, cache, rows, "cpu", amp=False)
        assert len(P) == len(rows) and np.isfinite(P.logit).all()
        print(f"  F7 {arm}: geçti (gövdede değişen parametre {changed}/{len(body)}; donuk {len(frozen)})", flush=True)


def test_ssl_resolution(work: Path) -> None:
    sel = work / "sel.csv"
    pd.DataFrame([{"repeat": r, "fold": k, "selected": ["beats", "wavlm_large"][(r + k) % 2]} for r in range(3) for k in range(5)]).to_csv(sel, index=False)
    assert F.resolve_backbone_arm("ssl", 0, 0, sel) == "beats" and F.resolve_backbone_arm("ssl", 0, 1, sel) == "wavlm_large"
    assert F.resolve_backbone_arm("cnn10", 0, 1, sel) == "cnn10"
    print("  F8 geçti (D-035 fold başına seçim)", flush=True)


def test_learnable(fx) -> None:
    cache = Cache(fx["cache"])
    fd = F.FoldData(fx["splits"], fx["ctx"], cache, 0, 0, "cv")
    torch.manual_seed(0)
    m = F.build_model("cnn10", None, "cpu", random_init_test=True)
    cfg = dict(F.ARMS["cnn10"], ckpt_every=100)
    with tempfile.TemporaryDirectory() as td:
        res = F.run_training(m, fd, cache, cfg, 4, 1, Path(td), "cpu", amp=False)
    tl = res["log"].train_loss.to_numpy()
    assert tl[-1] < tl[0], f"F9: eğitim kaybı düşmedi {tl}"
    P = F.predict_recordings(m, cache, fd.rows_eval, "cpu", amp=False)
    P["y"] = fd.label[fd.pid_of_row[P.row].to_numpy()].to_numpy()
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(P.y, P.logit)
    print(f"  F9 geçti (CNN10 rastgele başlangıç, 4 epoch: kayıp {tl[0]:.3f} → {tl[-1]:.3f}; kayıt AUC {auc:.2f})", flush=True)


def test_smk002(fx, work: Path) -> None:
    rep = work / "smk"
    run(REPO / "scripts/smoke_test_finetune.py", "--arms", "beats,wavlm_large", "--splits-dir", fx["splits"], "--context", fx["ctx"],
        "--cache-dir", fx["cache"], "--report-dir", rep, "--device", "cpu", "--random-init-test")
    R = json.loads((rep / "SMK-002_finetune.json").read_text())
    assert all(r["status"] == "PASS" for r in R["results"]) and (rep / "SMK-002_finetune.md").exists()
    print("  F10 geçti (SMK-002 CPU: " + ", ".join(f"{r['arm']} {r['status']}" for r in R["results"]) + ")", flush=True)


def build_day_fingerprint_fixture(root: Path, seed: int = 4) -> dict:
    """240 katılımcı, 40 kayıt günü (32'si tek etiketli, 8'i karışık; etiket-gün eşleşmesi zamana göre rastgele).
    Gömme = kayıt gününe özgü rastgele vektör + gürültü; astım bilgisi YOK. MFCC saf gürültü."""
    rng = np.random.default_rng(seed)
    n_days = 40
    day_label = np.r_[np.ones(24), np.zeros(8), np.full(8, -1)]           # 24 yalnız hasta, 8 yalnız sağlıklı, 8 karışık
    rng.shuffle(day_label)
    rows = []
    pid = 900001
    for d in range(n_days):
        for _ in range(6):
            y = int(day_label[d]) if day_label[d] >= 0 else int(rng.random() < 0.5)
            rows.append({"participant_id": pid, "label": y, "age": float(rng.integers(20, 65)), "days": d * 5,
                         "start_hour": round(float(rng.uniform(8, 17)), 2)})
            pid += 1
    C = pd.DataFrame(rows)
    C["recording_date"] = [str((pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(x))).date()) for x in C.days]
    C["has_audio"] = True
    C["in_time_overlap"] = True
    C.to_csv(root / "ctx.csv", index=False)
    recs = [(p, s) for p in C.participant_id for s in range(1, 8)]
    pd.DataFrame([{"participant_id": p, "slot": s, "status": "OK"} for p, s in recs]).to_csv(root / "rm.csv", index=False)
    meta = pd.DataFrame([{"row": i, "participant_id": p, "slot": s, "dur_s": 10.0} for i, (p, s) in enumerate(recs)])
    Ci = C.set_index("participant_id")
    dvec = rng.normal(0, 1, (n_days * 5 + 1, 8))
    dd = Ci.loc[meta.participant_id, "days"].to_numpy()
    X = (1.5 * dvec[dd] + rng.normal(0, 1, (len(meta), 8)))[:, None, :].astype(np.float32)
    e = root / "emb" / "cnn10"
    e.mkdir(parents=True)
    np.save(e / "rec_win4.npy", X)
    meta.to_csv(e / "recordings.csv", index=False)
    (e / "info.json").write_text(json.dumps({"primary_layers": [0], "n_params": 1000, "random_init": False, "n_store": 1, "dim": 8,
                                             "short_policy": "zeropad"}))
    Fm = meta[["participant_id", "slot"]].copy()
    Fm["kept_s"] = 10.0
    for i in range(44):
        Fm[f"f{i:02d}"] = rng.normal(0, 1, len(Fm))
    Fm.to_csv(root / "mfcc.csv", index=False)
    return {"ctx": root / "ctx.csv"}


def test_g1_known_truth(work: Path) -> None:
    root = work / "g1"
    root.mkdir()
    fx = build_day_fingerprint_fixture(root)
    out = {}
    for kind, extra in [("part", []), ("day", ["--group-by-day"])]:
        sd = root / f"splits_{kind}"
        run(REPO / "scripts/make_splits.py", "--context", fx["ctx"], "--recording-map", root / "rm.csv", "--out-dir", sd,
            "--report-dir", root / f"sr_{kind}", "--n-repeats", "1", *extra)
        run(REPO / "scripts/run_probes.py", "--exp", "EXP-016DG", "--emb-root", root / "emb", "--backbones", "cnn10",
            "--mfcc-features", root / "mfcc.csv", "--context", fx["ctx"], "--splits-dir", sd, "--out-dir", root / f"probe_{kind}",
            "--report-dir", root / f"rp_{kind}", "--n-repeats", "1", "--n-jobs", "2")
        run(REPO / "scripts/evaluate_phase3.py", "frozen-dg", "--probe-dir", root / f"probe_{kind}", "--splits-dir", sd,
            "--context", fx["ctx"], "--report-dir", root / f"g1_{kind}", "--backbones", "cnn10", "--n-repeats", "1", "--n-boot", "100")
        out[kind] = json.loads((root / f"g1_{kind}" / "G1_frozen_daygrouped.json").read_text())["t1"]["cnn10"]
    assert out["part"]["p_lrt"] < 0.01 and out["part"]["or_sd"] > 1, f"F11: katılımcı-gruplu CV'de gün izi yanlış pozitif vermedi {out['part']}"
    assert out["day"]["p_lrt"] > 0.05, f"F11: gün-gruplu CV gün izini engellemedi {out['day']}"
    print(f"  F11 geçti (yalnız gün izi: katılımcı-gruplu T1 p {out['part']['p_lrt']:.1e} → gün-gruplu p {out['day']['p_lrt']:.2f})",
          flush=True)


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="test_phase3_"))
    try:
        fx = build_fixture(work)
        test_day_splits(fx)
        test_fold_gates(fx)
        test_choose_epochs()
        test_gate(fx, work)
        test_ssl_resolution(work)
        test_families(fx)
        test_calibrate_train_resume(fx, work)
        test_selection_rule_and_warmup()
        test_nonfinite(fx, work)
        test_learnable(fx)
        test_smk002(fx, work)
        test_g1_known_truth(work)
        print("Tüm Faz 3 kontrolleri geçti (F1–F14). Not: sentetik veri ve küçük rastgele modeller; gerçek eğitim SMK-002 ister.")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
