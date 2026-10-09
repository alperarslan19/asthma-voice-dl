"""
Faz 2 hattı için kendi kendine yeten test (gerçek veri ve gerçek ağırlık GEREKMEZ; torch, torchaudio,
torchlibrosa, transformers gerekir). Ağırlıksız küçük modellerle bütün kod yollarını sınar.

ÖNEMLİ: Bu testler gerçek checkpoint'lerle yapılan SMK-001'in yerine GEÇMEZ. Rastgele küçük modeller kodun doğru
çalıştığını gösterir; resmi ağırlıkların doğru yüklendiğini, resmi girdi biçimini aldığını ya da anlamlı çıktı verdiğini
göstermez. Bunlar yalnız notebooks/20_smoke_tests.ipynb'deki SMK-001'de (GPU, gerçek ağırlıklar) sınanır.

    python tests/test_phase2.py

Kontroller
    P1  third_party kopyaları değişmemiş (sha256); sözleşmedeki sınıflar resmi kodla örneklenebiliyor (Cnn14_16k ayrı sınıf)
    P2  window_starts: kapsama, sona hizalı son pencere, kısa kayıt tek parça; dolgu yolu: tam uzunlukta lengths = dolgusuz
    P3  backbone sarmalayıcıları: şekil, batch-değişmezlik, son kanca = resmi çıktı (WavLM stable / non-stable, BEATs);
        ara katman kancaları = resmi ara çıktılar (BEATs tgt_layer yolu, HF hidden_states)
    P4  smoke_test_backbones.py --random-init sahte önbellekte uçtan uca çalışır (CPU eğitim adımı, S14 dolgu dahil);
        rapor UNITTEST_random_init_smoke.* adıyla yazılır (SMK-001 adı KULLANILMAZ)
    P5  extract_embeddings.py: şekiller, pencere sayıları, kısa kayıt tek pencere, iki dolgu politikası çapraz tutarlı
        (zeropad'in alternatifi = nopad çalıştırmasının birincili, ve tersi), kaldığı yerden devam = aynı dosya
    P6  LayerAverage yalnız train istatistiğini kullanır; DeLong = O(mn) başvuru uygulaması (bağlarla); Holm doğru;
        hızlı bootstrap = rb.boot_ci / ais.paired; katman ortalaması gerçek boyutlarda (12×768, 24×1024) doğru şekilde;
        D-035 fold başına seçim: kurgulanmış örneklerde doğru, eşitlik kuralı, iç tablolarda dış test katılımcısı → hata
    P7  run_probes EXP-016: yerleştirilmiş sinyal AUC > 0.8, saf gürültü 0.35–0.65; her katılımcı her tekrarda tam bir
        kez test tahmini alır; onaylayıcı tablo, iç içe seçim, D-035 kuralı (sinyalli aday seçilir); rapor ID içermez;
        ikinci çalıştırma ara sonuçları kullanır; bir katılımcının bir görevi eksikken (101244 durumu) füzyon 6 görevle yapılır;
        her (kol, tekrar, fold) için dış test ve iç (eğitim) katılımcıları ayrık; D-035 seçimi dış test tahminleri
        bozulduğunda DEĞİŞMEZ (seçim yalnız iç doğrulamaya bağlı); ana raporda bağlam izleme göstergesi yok
    P7b EXP-016S (kısa kayıt dolgu politikası duyarlılığı) çalışır
    P7c META-016 ayrı meta veri raporu çalışır, ID içermez
    P8  EXP-017: sinyalin yerleştirildiği katmanlar diğerlerinden yüksek; EXP-016'nın tekrar-0 sonucu yeniden kullanılır
    P9  EXP-018 çalışır (tüm kayıt vs pencere karşılaştırması)
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tests"))
import backbones as B  # noqa: E402
import run_probes as RP  # noqa: E402
import test_mfcc_pipeline as tmp_mfcc  # noqa: E402  (sentetik kohort + split üretici)


def ids_in(txt: str, ids) -> set:
    """Metinde ayrı bir sayı olarak geçen 6 haneli ID'ler (0.8000123 gibi ondalıkların içindeki diziler sayılmaz)."""
    return {int(m) for m in re.findall(r"(?<![\d.])(\d{6})(?![\d.])", txt)} & set(ids)


def run(*cmd) -> subprocess.CompletedProcess:
    r = subprocess.run([sys.executable, *map(str, cmd)], capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout[-4000:] + r.stderr[-4000:])
    return r


# ----------------------------------------------------------------------------- P1–P3
def test_units() -> None:
    B.check_vendored()
    m = B._import_panns()
    for name in ["cnn10", "cnn14", "cnn14_16k"]:
        c = B.load_contract(name)
        assert hasattr(m, c["class"]), c["class"]
    assert B.load_contract("cnn14_16k")["class"] == "Cnn14_16k"
    # P2
    W, H = 64000, 32000
    for n in [64000, 64001, 96000, 164160, 200000]:
        ws = B.window_starts(n, W, H)
        cover = np.zeros(n, bool)
        for s, L in ws:
            assert L == W and 0 <= s and s + L <= n
            cover[s:s + L] = True
        assert cover.all(), f"P2: {n} örneklik kayıt tam kapsanmadı"
        assert ws[-1][0] + W == n, "P2: son pencere sona hizalı değil"
    assert B.window_starts(36480, W, H) == [(0, 36480)], "P2: kısa kayıt tek parça olmalı"
    # P3
    torch.manual_seed(0)
    for name in ["cnn10", "beats", "wavlm_base_plus", "wavlm_large"]:
        bb = B.build_backbone(name, random_init=True)
        w = torch.randn(3, 4 * bb.sr) * 0.1
        e = bb.embed(w)
        assert e.shape == (3, bb.n_store, bb.dim) and e.dtype == torch.float32
        assert (bb.embed(w[:1]) - e[:1]).abs().max() < 1e-4, f"P3 {name}: batch-değişmezlik"
        assert bb.embed(torch.randn(1, int(2.3 * bb.sr)) * 0.1).shape == (1, bb.n_store, bb.dim)
        full_len = torch.full((3,), 4 * bb.sr, dtype=torch.long)
        assert (bb.embed(w, lengths=full_len) - e).abs().max() < 1e-5, f"P2 {name}: tam uzunlukta lengths sonucu değiştirdi"
        if name in {"beats", "wavlm_large"}:          # maske dolguyu gerçekten dışlıyor mu? 4 s ve 6 s'ye dolgu aynı sonucu vermeli
            n_ = int(2.3 * bb.sr)
            x_ = torch.randn(1, n_) * 0.1
            p4, p6 = torch.zeros(1, 4 * bb.sr), torch.zeros(1, 6 * bb.sr)
            p4[0, :n_], p6[0, :n_] = x_[0], x_[0]
            e4, e6 = bb.embed(p4, lengths=torch.tensor([n_])), bb.embed(p6, lengths=torch.tensor([n_]))
            assert (e4 - e6).abs().max() < 1e-4, f"P2 {name}: dolgu uzunluğu maskeli gömmeyi değiştirdi"
            if name == "wavlm_large":                 # maske modele verildiği için zeropad ≈ nopad
                assert (e4 - bb.embed(x_)).abs().max() < 1e-4, "P2 wavlm_large: maskeli dolgu ≠ dolgusuz"
        bad = w.clone()
        try:
            bb.embed(bad, lengths=torch.tensor([bb.sr, 4 * bb.sr, 4 * bb.sr]))
            raise RuntimeError("P2: sıfır olmayan dolgu bölgesi kabul edildi")
        except AssertionError:
            pass
        if bb.family != "panns":
            a, b = bb.official_last(w)
            assert (a - b).abs().max() < 1e-5, f"P3 {name}: son kanca resmi çıktıdan farklı"
            assert bb.primary_layers == list(range(1, bb.n_store))
            with torch.no_grad():
                if bb.family == "beats":
                    m = bb.model
                    f = m.patch_embedding(m.preprocess(w).unsqueeze(1))
                    f = m.layer_norm(f.reshape(f.shape[0], f.shape[1], -1).transpose(1, 2))
                    f = m.post_extract_proj(f) if m.post_extract_proj is not None else f
                    _, lr = m.encoder.extract_features(f, None, tgt_layer=bb.L - 1)
                    official = torch.stack([x.mean(0) for x, _ in lr], 1)          # resmi ara katmanlar (T×B×C)
                    check = range(bb.n_store)
                else:
                    hs = bb.model(bb.preprocess(w)[0], output_hidden_states=True).hidden_states
                    official = torch.stack([h.mean(1) for h in hs], 1)
                    stable = bb.model.config.do_stable_layer_norm               # son öğenin anlamı sürüme bağlı
                    check = range(bb.n_store - 1 if stable else bb.n_store)
            for i in check:
                assert (e[:, i] - official[:, i]).abs().max() < 1e-5, f"P3 {name}: katman {i} kancası resmi ara çıktıdan farklı"
        if bb.family in {"panns", "beats"}:           # SMK-001: fp16 autocast'ta ön işleme fp32 kalmalı (en son: eğitim modu BN'yi değiştirir)
            tt = torch.arange(4 * bb.sr) / bb.sr
            xs = torch.stack([0.89 * torch.sin(2 * np.pi * (150 + 20 * i) * tt) * (tt < 3).float() for i in range(2)])
            bb.model.train()
            with torch.autocast("cpu", dtype=torch.float16):
                assert torch.isfinite(bb.train_features(xs)).all(), f"P3 {name}: fp16 autocast'ta sonlu olmayan çıktı"
            bb.model.eval()
    print("  P1–P3 geçti", flush=True)


# ----------------------------------------------------------------------------- sahte önbellek
def fake_cache(d: Path, n_part: int = 8) -> None:
    rng = np.random.default_rng(1)
    rows, a32, a16 = [], [], []
    o32 = o16 = 0
    for i in range(n_part):
        f0 = 100 + 15 * i
        for slot in range(1, 8):
            dur = 3.0 if (i == 0 and slot == 3) else 5.0 + 0.5 * (slot % 3)
            seg = {}
            for sr in (32000, 16000):
                t = np.arange(int(dur * sr)) / sr
                x = sum(np.sin(2 * np.pi * f0 * k * (1 + 0.05 * slot) * t) / k for k in range(1, 6))
                x = x + 0.05 * rng.standard_normal(len(t))
                seg[sr] = (0.9 * x / np.abs(x).max()).astype("<f4")
            rows.append({"participant_id": 900001 + i, "slot": slot, "offset_32k": o32, "n_32k": len(seg[32000]),
                         "offset_16k": o16, "n_16k": len(seg[16000])})
            o32 += len(seg[32000])
            o16 += len(seg[16000])
            a32.append(seg[32000])
            a16.append(seg[16000])
    d.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(d / "index.csv", index=False)
    np.concatenate(a32).tofile(d / "audio_32k.f32")
    np.concatenate(a16).tofile(d / "audio_16k.f32")


def test_smoke_and_extract(work: Path) -> None:
    cache = work / "cache"
    fake_cache(cache)
    rep = work / "smk"
    r = run(REPO / "scripts/smoke_test_backbones.py", "--cache-dir", cache, "--report-dir", rep, "--random-init",
            "--backbones", "cnn10,beats,wavlm_large", "--device", "cpu", "--n-participants", "4", "--train-batch", "2",
            "--speed-batch", "4")
    assert not (rep / "SMK-001_smoke.json").exists(), "P4: rastgele modeller SMK-001 adıyla rapor yazmamalı"
    j = json.loads((rep / "UNITTEST_random_init_smoke.json").read_text())
    assert all(x["status"] == "PASS" for x in j["results"]), r.stdout[-2000:]
    assert "S12_train_step" in j["results"][0]["checks"] and "S14_padding_path" in j["results"][0]["checks"]
    assert (rep / "UNITTEST_random_init_smoke.md").read_text().startswith("# BİRİM TESTİ")
    smk_txt = (rep / "UNITTEST_random_init_smoke.md").read_text() + (rep / "UNITTEST_random_init_smoke.json").read_text()
    assert not ids_in(smk_txt, range(900001, 900009)), "P4: raporda ID"
    print("  P4 geçti", flush=True)
    # P5
    out = work / "emb" / "cnn10"
    args = ["--backbone", "cnn10", "--random-init", "--cache-dir", cache, "--out-dir", out, "--device", "cpu",
            "--shard-size", "20", "--save-windows"]
    run(REPO / "scripts/extract_embeddings.py", *args)
    E = np.load(out / "rec_win4.npy")
    meta = pd.read_csv(out / "recordings.csv")
    info = json.loads((out / "info.json").read_text())
    assert E.shape == (56, 1, 512) and np.load(out / "rec_full.npy").shape == E.shape
    assert meta.loc[(meta.participant_id == 900001) & (meta.slot == 3), "n_windows"].item() == 1, "P5: kısa kayıt"
    assert (meta.n_windows[meta.dur_s >= 5.0] >= 2).all()
    assert info["n_short_lt4s"] == 1 and info["random_init"] is True
    W = np.load(out / "win4_windows.npy")
    assert len(W) == meta.n_windows.sum() and W.dtype == np.float16
    sha1 = B.sha256_file(out / "rec_win4.npy")
    r2 = run(REPO / "scripts/extract_embeddings.py", *args)
    assert "parça" not in r2.stdout, "P5: devam eden çalıştırma parçaları yeniden üretti"
    assert B.sha256_file(out / "rec_win4.npy") == sha1, "P5: devam eden çalıştırma farklı sonuç"
    assert info["short_policy"] == "zeropad" and meta.short.sum() == 1
    for name in ["cnn10", "beats", "wavlm_large"]:            # iki politika çapraz tutarlı (aynı ağırlık: random_init tohumu)
        outs = {}
        for pol in ["zeropad", "nopad"]:
            o = work / "emb_pol" / f"{name}_{pol}"
            run(REPO / "scripts/extract_embeddings.py", "--backbone", name, "--random-init", "--cache-dir", cache,
                "--out-dir", o, "--device", "cpu", "--views", "win4", "--short-policy", pol)
            outs[pol] = (np.load(o / "rec_win4.npy"), np.load(o / "short_alt_win4.npz"), pd.read_csv(o / "recordings.csv"))
        (Ez, Az, mz), (En, An, mn) = outs["zeropad"], outs["nopad"]
        srow = mz.row[mz.short].to_numpy()
        assert str(Az["policy"]) == "nopad" and str(An["policy"]) == "zeropad" and np.array_equal(Az["rows"], srow)
        assert np.allclose(Az["emb"], En[srow], atol=1e-5) and np.allclose(An["emb"], Ez[srow], atol=1e-5), f"P5 {name}: politikalar çapraz tutarsız"
        long_ = ~mz.short.to_numpy()
        assert np.allclose(Ez[long_], En[long_], atol=1e-5), f"P5 {name}: 4 s ve üstü kayıtlar politikadan etkilenmemeli"
    run(REPO / "scripts/extract_embeddings.py", "--backbone", "wavlm_base_plus", "--random-init", "--cache-dir", cache,
        "--out-dir", work / "emb" / "wavlm_base_plus", "--device", "cpu", "--views", "win4")
    assert np.load(work / "emb" / "wavlm_base_plus" / "rec_win4.npy").shape == (56, 3, 64)
    print("  P5 geçti", flush=True)


# ----------------------------------------------------------------------------- P6
def test_stats() -> None:
    rng = np.random.default_rng(0)
    A, Bm = rng.normal(0, 1, (50, 2 * 3)), rng.normal(5, 3, (40, 2 * 3))
    la = RP.LayerAverage(2, 3).fit(A)
    Z = Bm.reshape(40, 2, 3)
    exp = ((Z - A.reshape(50, 2, 3).mean(0)) / A.reshape(50, 2, 3).std(0)).mean(1)
    assert np.allclose(la.transform(Bm), exp), "P6: LayerAverage train istatistiği kullanmıyor"
    y = rng.integers(0, 2, 300)
    p1, p2 = y + rng.normal(0, 1, 300), y + rng.normal(0, 3, 300)
    d, z, p = RP.delong_paired(y, p1, p2)
    assert abs(d - (roc_auc_score(y, p1) - roc_auc_score(y, p2))) < 1e-10 and p < 0.01
    assert RP.delong_paired(y, p1, p1)[2] == 1.0
    import analyze_imbalance_selection as ais
    import run_mfcc_baselines as rb
    pp = 1 / (1 + np.exp(-p1))
    assert np.allclose(RP.boot_ci_fast(y, pp, n=300), rb.boot_ci(y, pp, n=300)), "P6: hızlı bootstrap CI farklı"
    assert np.allclose(RP.paired_fast(y, p1, p2, n=300), ais.paired(y, p1, p2, n=300)), "P6: hızlı eşleştirilmiş bootstrap farklı"
    # DeLong: O(mn) yapısal bileşen başvurusu, bağlı (yuvarlanmış) tahminlerle
    yy = rng.integers(0, 2, 120)
    q1, q2 = np.round(yy + rng.normal(0, 1, 120), 1), np.round(yy + rng.normal(0, 1.5, 120), 1)
    pos = yy == 1
    psi = lambda a, b: (a > b) + 0.5 * (a == b)
    V10 = np.vstack([[psi(x, q[~pos]).mean() for x in q[pos]] for q in (q1, q2)])
    V01 = np.vstack([[psi(q[pos], v).mean() for v in q[~pos]] for q in (q1, q2)])
    Sm = np.cov(V10) / pos.sum() + np.cov(V01) / (~pos).sum()
    z_ref = (V10[0].mean() - V10[1].mean()) / np.sqrt(Sm[0, 0] + Sm[1, 1] - 2 * Sm[0, 1])
    assert np.isclose(RP.delong_paired(yy, q1, q2)[1], z_ref), "P6: DeLong z başvurudan farklı"
    # Katman ortalaması gerçek boyutlarda: (N, L, D) → düzleştir → LayerAverage → (N, D)
    for L_, D_ in [(12, 768), (24, 1024)]:
        Xr = rng.normal(0, 1, (30, L_ * D_)).astype(np.float32)
        out_ = RP.LayerAverage(L_, D_).fit(Xr[:20]).transform(Xr[20:])
        assert out_.shape == (10, D_), f"P6: LayerAverage çıktısı {out_.shape}"
    # D-035 fold başına seçim (kurgulanmış örnek): 20 katılımcı, 5 fold, 2 görev
    pids = np.arange(1, 21)
    Cx = pd.DataFrame({"label": (pids % 2).astype(int)}, index=pd.Index(pids, name="participant_id"))
    Sx = pd.DataFrame([{"participant_id": q, "repeat": 0, "outer_fold": k, "role": "test" if q % 5 == k else "train",
                        "inner_fold": q % 4} for k in range(5) for q in pids])
    par = {"beats": 90e6, "wavlm_base_plus": 95e6, "wavlm_large": 316e6}

    def inner_tab(strength: dict) -> pd.DataFrame:
        rr = np.random.default_rng(1)
        rows_ = []
        for k in range(5):
            trn = [q for q in pids if q % 5 != k]
            for a, sd in strength.items():
                for task in ["aaa", "araba"]:
                    for q in trn:
                        sig_ = Cx.label[q] if sd is not None else 0
                        rows_.append({"participant_id": q, "task": task, "arm": a, "repeat": 0, "fold": k,
                                      "prob": sig_ + rr.normal(0, sd if sd is not None else 1)})
        return pd.DataFrame(rows_)
    sel = RP.select_per_fold(inner_tab({"beats": None, "wavlm_base_plus": None, "wavlm_large": 0.01}), Cx, {0: Sx},
                             ["beats", "wavlm_base_plus", "wavlm_large"], par)
    assert (sel.selected == "wavlm_large").all() and len(sel) == 5, sel
    tie = inner_tab({"beats": 0.01, "wavlm_base_plus": None, "wavlm_large": 0.01})      # ikisi de AUC 1.0 → eşit
    sel_t = RP.select_per_fold(tie, Cx, {0: Sx}, ["beats", "wavlm_base_plus", "wavlm_large"], par)
    assert (sel_t.selected == "beats").all(), "P6: eşitlikte en az parametreli seçilmeli"
    leak = tie.copy()
    leak.loc[0, "participant_id"] = [q for q in pids if q % 5 == int(leak.loc[0, "fold"])][0]   # dış test katılımcısı
    try:
        RP.select_per_fold(leak, Cx, {0: Sx}, ["beats", "wavlm_base_plus", "wavlm_large"], par)
        raise RuntimeError("P6: iç tabloda dış test katılımcısı yakalanmadı")
    except AssertionError:
        pass
    FTx = pd.DataFrame([{"arm": a, "task": "fusion", "repeat": 0, "fold": k, "auc": v + 0.01 * k}
                        for a, v in {"x": 0.7, "y": 0.8}.items() for k in range(5)])
    selx = pd.DataFrame({"repeat": 0, "fold": range(5), "selected": ["x", "x", "y", "y", "y"]})
    pe = RP.procedure_estimate(selx, FTx, ["x", "y"])
    assert np.isclose(pe["procedure_auc_mean"], np.mean([0.70, 0.71, 0.82, 0.83, 0.84])) and pe["best_fixed_arm_post_hoc"] == "y"
    h = RP.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert np.isclose(h["a"], 0.03) and np.isclose(h["c"], 0.06) and np.isclose(h["b"], 0.06)
    print("  P6 geçti", flush=True)


# ----------------------------------------------------------------------------- P7–P9
def synthetic_embeddings(root: Path, ctx: Path, rng) -> None:
    C = pd.read_csv(ctx).set_index("participant_id")
    pids = C.index.to_numpy()
    meta = pd.DataFrame([{"row": i, "participant_id": p, "slot": s, "dur_s": 3.0 if (p in SHORT_PIDS and s == 2) else 10.0}
                         for i, (p, s) in enumerate((p, s) for p in pids for s in range(1, 8))])
    y = C.loc[meta.participant_id, "label"].to_numpy()
    specs = {"cnn10": (1, 16, {0: 0.6}), "beats": (13, 8, {l: 1.5 for l in range(4, 9)}), "wavlm_base_plus": (3, 8, {})}
    for name, (L, D, sig) in specs.items():
        E = rng.normal(0, 1, (len(meta), L, D)).astype(np.float32)
        E[:, :, 1:] *= np.linspace(1, 20, L)[None, :, None]          # katmanlar arası ölçek farkı (LayerAverage sınaması)
        for l, s in sig.items():
            E[:, l, 0] += s * y
        d = root / name
        d.mkdir(parents=True)
        np.save(d / "rec_win4.npy", E)
        np.save(d / "rec_full.npy", E + rng.normal(0, 0.1, E.shape).astype(np.float32))
        meta.to_csv(d / "recordings.csv", index=False)
        prim = [0] if L == 1 else list(range(1, L))
        (d / "info.json").write_text(json.dumps({"primary_layers": prim, "n_params": 1000 * L, "random_init": True,
                                                 "n_store": L, "dim": D, "short_policy": "zeropad"}))


SHORT_PIDS = {800002, 800003, 800004}


def write_short_alt(d: Path, rng) -> None:
    meta = pd.read_csv(d / "recordings.csv")
    E = np.load(d / "rec_win4.npy")
    rows = meta.row[meta.dur_s < 4.0].to_numpy()
    np.savez(d / "short_alt_win4.npz", rows=rows, emb=E[rows] + rng.normal(0, 0.5, E[rows].shape).astype(np.float32),
             policy=np.array("nopad"))


def fake_index(path: Path, ctx: Path) -> None:
    C = pd.read_csv(ctx)
    rows = [{"participant_id": p, "slot": s, "n_16k": int((3.0 if (p in SHORT_PIDS and s == 2) else 10.0) * 16000)}
            for p in C.participant_id for s in range(1, 8) if not (p == 800001 and s == 4)]
    path.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(path / "index.csv", index=False)


def drop_slot(path: Path, pid: int, slot: int) -> None:
    f = pd.read_csv(path)
    f[~((f.participant_id == pid) & (f.slot == slot))].to_csv(path, index=False)


def test_probes(work: Path) -> None:
    ctx, fsig, _ = tmp_mfcc.synthetic_cohort(work, np.random.default_rng(0))
    emb = work / "emb_syn"
    synthetic_embeddings(emb, ctx, np.random.default_rng(1))
    # 101244 durumu: bir katılımcının görev 4 kaydı yok (bütün kollarda)
    drop_slot(fsig, 800001, 4)
    for d in emb.iterdir():
        meta = pd.read_csv(d / "recordings.csv")
        keep = ~((meta.participant_id == 800001) & (meta.slot == 4)).to_numpy()
        for v in ["win4", "full"]:
            np.save(d / f"rec_{v}.npy", np.load(d / f"rec_{v}.npy")[keep])
        meta[keep].assign(row=np.arange(keep.sum())).to_csv(d / "recordings.csv", index=False)
        write_short_alt(d, np.random.default_rng(2))
    reg = work / "registry.csv"
    pd.read_csv(REPO / "results/registry.csv", nrows=0).to_csv(reg, index=False)
    common = ["--emb-root", emb, "--context", ctx, "--splits-dir", work / "splits", "--out-dir", work / "fp",
              "--n-repeats", "2", "--n-jobs", "2", "--backbones", "cnn10,beats,wavlm_base_plus"]
    run(REPO / "scripts/run_probes.py", "--exp", "EXP-016", "--mfcc-features", fsig, "--report-dir", work / "rep",
        "--registry", reg, *common)
    rep = json.loads((work / "rep" / "EXP-016_frozen.json").read_text())
    S = pd.DataFrame(rep["summary"])
    fus = S[S.task == "fusion"].set_index("arm").auc_mean
    assert fus["beats"] > 0.8, f"P7: sinyal AUC {fus['beats']:.3f}"
    assert 0.35 < fus["wavlm_base_plus"] < 0.65, f"P7: gürültü AUC {fus['wavlm_base_plus']:.3f}"
    assert fus["mfcc_lr"] > 0.8 and "mfcc_mlp" in fus and "ref_age" in fus and "ref_context" in fus
    assert len(rep["confirmatory"]) == 3 and all("holm_p" in c for c in rep["confirmatory"])
    assert "context_monitors" not in rep, "P7: bağlam izleme ana rapora girmemeli (META-016)"
    sel0 = pd.read_csv(work / "fp" / "d035_selection.csv")
    assert len(sel0) == 10 and (sel0.selected == "beats").all(), sel0       # 2 tekrar × 5 fold; sinyalli aday
    assert "6 backbone (keşifsel)" in rep["procedure_estimates"]
    # dış test ve iç (eğitim) katılımcıları her (kol, tekrar, fold) için ayrık
    for f in (work / "fp" / "partial").glob("*_test.csv"):
        t, inn = pd.read_csv(f), pd.read_csv(str(f).replace("_test.csv", "_inner.csv"))
        for (r, k), g in t.groupby(["repeat", "fold"]):
            gi = inn[(inn.repeat == r) & (inn.fold == k)]
            assert not set(g.participant_id) & set(gi.participant_id), f"P7: {f.name} r{r} k{k} test ∩ iç ≠ ∅"
    for f in (work / "fp" / "partial").glob("*_test.csv"):
        t = pd.read_csv(f)
        for (r, task), g in t.groupby(["repeat", "task"]):
            n_exp = 159 if task == "ordu" else 160
            assert g.participant_id.is_unique and len(g) == n_exp, f"P7: {f.name} r{r} {task}: {len(g)}"
    npart = S.set_index(["arm", "task"]).n_participants
    assert npart[("beats", "fusion")] == 160 and npart[("beats", "ordu")] == 159, "P7: eksik görevli katılımcı füzyonda yok"
    txt = (work / "rep" / "EXP-016_frozen.md").read_text() + (work / "rep" / "EXP-016_frozen.json").read_text()
    found = ids_in(txt, range(800001, 800161))
    assert not found, f"P7: raporda katılımcı ID'si: {sorted(found)[:3]}"
    assert (pd.read_csv(reg).experiment_id == "EXP-016").sum() > 0
    # D-035 güvencesi: dış test tahminleri bozulsa da seçim değişmez (seçim yalnız iç doğrulamaya bağlı)
    for r in range(2):
        f = work / "fp" / "partial" / f"wavlm_base_plus_r{r}_test.csv"
        t = pd.read_csv(f)
        t["prob"] = pd.read_csv(ctx).set_index("participant_id").loc[t.participant_id, "label"].to_numpy()  # kusursuz test
        t.to_csv(f, index=False)
    r2 = run(REPO / "scripts/run_probes.py", "--exp", "EXP-016", "--mfcc-features", fsig, "--report-dir", work / "rep2", *common)
    assert r2.stdout.count("önceki sonuç kullanıldı") == 10, "P7: ara sonuçlar yeniden kullanılmadı"
    sel1 = pd.read_csv(work / "fp" / "d035_selection.csv")
    assert sel1.equals(sel0), "P7: D-035 seçimi dış test tahminlerinden etkilendi"
    rep2 = json.loads((work / "rep2" / "EXP-016_frozen.json").read_text())
    s2 = pd.DataFrame(rep2["summary"]).query("task == 'fusion'").set_index("arm").auc_mean
    assert s2["wavlm_base_plus"] > 0.99, "P7: bozulmuş test dosyaları kullanılmadı (sınama geçersiz)"
    print(f"  P7 geçti (sinyal {fus['beats']:.3f}, gürültü {fus['wavlm_base_plus']:.3f}, mfcc_lr {fus['mfcc_lr']:.3f})", flush=True)
    # P7b — EXP-016S
    run(REPO / "scripts/run_probes.py", "--exp", "EXP-016S", "--report-dir", work / "rep", *common)
    r16s = json.loads((work / "rep" / "EXP-016S_frozen.json").read_text())
    assert {c["a"] for c in r16s["exploratory_pairs"]} == {"cnn10@alt", "beats@alt", "wavlm_base_plus@alt"}
    print("  P7b geçti", flush=True)
    # P7c — META-016 (ayrı rapor)
    fake_index(work / "cache_syn", ctx)
    run(REPO / "scripts/report_metadata_monitor.py", "--probe-dir", work / "fp", "--cache-dir", work / "cache_syn",
        "--context", ctx, "--splits-dir", work / "splits", "--report-dir", work / "rep", "--n-repeats", "2")
    meta_rep = json.loads((work / "rep" / "META-016_metadata_monitor.json").read_text())
    assert meta_rep["n_short_recordings_total"] == 3 and meta_rep["short_exclusion_sensitivity"]["n_excluded_participants"] == 3
    assert set(meta_rep["context_monitors"]) >= {"beats", "mfcc_lr", "ref_context"}
    mtxt = (work / "rep" / "META-016_metadata_monitor.md").read_text() + (work / "rep" / "META-016_metadata_monitor.json").read_text()
    assert not ids_in(mtxt, range(800001, 800161)), "P7c: META raporunda ID"
    print("  P7c geçti", flush=True)
    # P8
    r3 = run(REPO / "scripts/run_probes.py", "--exp", "EXP-017", "--report-dir", work / "rep", *common)
    assert "beats r0: önceki sonuç kullanıldı" in r3.stdout
    curve = pd.DataFrame(json.loads((work / "rep" / "EXP-017_frozen.json").read_text())["layer_curves"]["beats"]).set_index("arm")
    sig = curve.loc[[f"beats@L{l}" for l in range(4, 9)], "fusion_auc_mean"]
    non = curve.loc[[f"beats@L{l}" for l in [0, 1, 2, 10, 11, 12]], "fusion_auc_mean"]
    assert sig.min() > non.max(), f"P8: katman eğrisi {curve.fusion_auc_mean.round(2).to_dict()}"
    print("  P8 geçti", flush=True)
    # P9
    run(REPO / "scripts/run_probes.py", "--exp", "EXP-018", "--report-dir", work / "rep", *common[:-1], "cnn10")
    r18 = json.loads((work / "rep" / "EXP-018_frozen.json").read_text())
    assert r18["exploratory_pairs"][0]["a"] == "cnn10@full"
    print("  P9 geçti", flush=True)


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="test_phase2_"))
    try:
        test_units()
        test_stats()
        test_smoke_and_extract(work)
        test_probes(work)
        print("Tüm kontroller geçti (P1–P9, P7b, P7c). Not: bu, gerçek checkpoint'lerle SMK-001 değildir.")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
