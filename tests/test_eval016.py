"""
EVAL-016 (scripts/evaluate_context.py) için sentetik, doğrusu bilinen testler. Gerçek veri GEREKMEZ; gerçek veride
hiçbir sonuç üretmez.

    python tests/test_eval016.py

Kurgu: kayıt bağlamı etiketi güçlü biçimde belirler (sağlıklılar çoğunlukla öğleden sonra ve erken dönemde; hastalar
çoğunlukla sabah, iki dönemde). Üç sentetik "backbone":
    cnn10            gömme, bağlamdan BAĞIMSIZ gerçek bir astım sinyali taşır      → T1 reddetmeli; C1 Aşama-1 koşulları ✓
    beats            gömme YALNIZ bağlamı (saat, tarih) kodlar, astım bilgisi yok → T1 reddetmemeli; G1+; N5 hesaplanır; C2 pozitif
    wavlm_base_plus  saf gürültü                                                   → T1 reddetmemeli; G1−
    wavlm_large      gerçek sinyal + bağlam (gerçekçi durum)                       → T1 reddetmeli; G1+; N5 YANLIŞ ALARM verir
                     (sağlıklılara uygulanan vekil model astım sinyalini taşır) → S11: N5 C1'de veto değil
Skorlar gerçek prob pipeline'ından gelir (run_probes.py EXP-016, 2 tekrar), yani partial/ dosyaları gerçek biçimdedir.

Kontroller
    E1  uçtan uca çalışır; kohort dışlamaları (saati eksik 2 katılımcı) sayılır
    E2  sinyal kolu: T1 p < 0.001, OR > 1, Holm p < 0.05, Aşama-1 koşulları ✓, Aşama 2 tetiklenir
    E3  yalnız-bağlam kolu: T1 p > 0.01; G1+; N5 hesaplanır ve bilgi eklemez (p > 0.01); C1 sağlanmadı; C2 pozitif
    E4  gürültü kolu: T1 p > 0.01; G1−; N5 yok
    E5  Holm p ≥ ham p; MFCC kolları referans (Holm ailesinde değil)
    E6  raporda katılımcı ID'si yok; Drive tablosu yazıldı
    E7  bağlam tablosu değişince (imza) ve ara sonuç eksikken çalışma durur
    E8  T1-LRT kalibrasyonu: sentetik bağlamda sinyalsiz (yalnız bağlam kodlayan) skorla ret oranı ≤ 0.15 (60 veri seti)
    E9  eksik görevli katılımcı (6 görev) ŝ alır
    E12 S11 gerekçesi: sinyal + bağlam kolunda N5 anlamlı (vekil skora astım sinyali sızar), ama N5 C1'de veto olmadığı için
        Aşama-1 koşulları sağlanır; test kipi raporda "karar için kullanılamaz" diye işaretlenir; --allow-unapproved olmadan
        test ayarlarıyla çalıştırma reddedilir (protokol onaysızsa "onay bekliyor", KABUL ise "sabitlenmiş ayarlardan sapma")
    E14 gün-kümeli varyanslar (Firth sandviç CR1 ve jackknife CV3) statsmodels ML ile bağımsız hesaplananla aynı
    E15 kapı birim testi (sabit ayarlar zorunlu; backbone sırası önemsiz) ve K2 fail-safe kuralı (K2 geçersizse C1 "kurulamadı")
    E16 kalibrasyon tanılaması CV3'ü hesaplar ve K2 geçerlilik kuralını değerlendirir; bütün kollarda CV3 p sonlu
    E17 yarı-tam ayrışmalı, çok düz bir yüzeyde Firth tahmini (Newton-CG cilası) R brglm2 ile aynı (brglm2 6105 iterasyon)
    E13 E2h için Jeffreys-cezalı koşullu logit = bağımsız R uygulaması (survival::coxph "exact" log-olabilirliği ve
        bilgi matrisi + optim); biri tabaka içinde tam ayrışmalı (klasik koşullu logit ıraksar)
    E11 C2'nin kısmi testi (etiket-içi permütasyon): bağlamsız sinyal skorunda p > 0.01 ve C2 "gösterilemedi";
        yalnız-bağlam skorunda p < 0.02 (100 permütasyonda en küçük p ≈ 0.0099)
    E10 Firth uygulaması bağımsız referansla aynı: katsayılar R brglm2 (type = "AS_mean") ile, cezalı olabilirlik oranı
        istatistikleri R'de genel amaçlı optimizasyonla (BFGS, ceza tam tasarımın bilgi matrisiyle) hesaplanan değerlerle
        eşleşir; biri yarı-tam ayrışmalı (klasik ML'nin yakınsamadığı) veri. Referanslar 2026-10-10'da R 4.3.3 /
        brglm2 0.9.2 ile üretildi.
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import evaluate_context as E  # noqa: E402

N1, N0 = 200, 60
BB = ["cnn10", "beats", "wavlm_base_plus", "wavlm_large"]
MISSING = (700003, 4)                    # bu katılımcının 4. görevi yok (6 görevle füzyon)
NO_HOUR = [700010, 700150]               # saati bilinmeyen iki katılımcı → T1 kohortundan dışlanır


def ids_in(txt: str, ids) -> set:
    return {int(m) for m in re.findall(r"(?<![\d.])(\d{6})(?![\d.])", txt)} & set(ids)


def run(*cmd, ok=True) -> subprocess.CompletedProcess:
    r = subprocess.run([sys.executable, *map(str, cmd)], capture_output=True, text=True)
    if ok:
        assert r.returncode == 0, (r.stdout[-4000:] + r.stderr[-4000:])
    return r


def cohort(rng) -> pd.DataFrame:
    n = N1 + N0
    y = np.r_[np.ones(N1, int), np.zeros(N0, int)]
    pid = np.arange(700001, 700001 + n)
    days = np.where(y == 1, rng.integers(0, 200, n), rng.integers(0, 81, n))
    am = np.where(y == 1, rng.random(n) < 0.7, rng.random(n) < 0.2)
    hour = np.where(am, rng.uniform(8, 12, n), rng.uniform(12, 17, n)).round(2)
    age = np.where(y == 1, rng.normal(50, 10, n), rng.normal(40, 10, n)).round()
    C = pd.DataFrame({"participant_id": pid, "label": y, "age": age, "start_hour": hour, "days": days,
                      "recording_date": [str((pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(d))).date()) for d in days],
                      "has_audio": True})
    C["in_time_overlap"] = C.days.between(C.days[C.label == 0].min(), C.days[C.label == 0].max())
    C.loc[C.participant_id.isin(NO_HOUR), "start_hour"] = np.nan
    return C


def write_inputs(work: Path, rng) -> dict:
    C = cohort(rng)
    ctx = work / "participant_context.csv"
    C.to_csv(ctx, index=False)
    rows = [(p, s) for p in C.participant_id for s in range(1, 8) if (p, s) != MISSING]
    rm = pd.DataFrame([{"participant_id": p, "slot": s, "status": "OK"} for p, s in rows]
                      + [{"participant_id": MISSING[0], "slot": MISSING[1], "status": "MISSING"}])
    rm.to_csv(work / "rm.csv", index=False)
    run(REPO / "scripts/make_splits.py", "--context", ctx, "--recording-map", work / "rm.csv", "--out-dir", work / "splits",
        "--report-dir", work / "splits_report", "--n-repeats", "2")
    Ci = C.set_index("participant_id")
    meta = pd.DataFrame([{"row": i, "participant_id": p, "slot": s, "dur_s": 10.0} for i, (p, s) in enumerate(rows)])
    yv = Ci.loc[meta.participant_id, "label"].to_numpy()
    h = Ci.loc[meta.participant_id, "start_hour"].fillna(Ci.start_hour.mean()).to_numpy()
    d = Ci.loc[meta.participant_id, "days"].to_numpy()
    zh, zd = (h - h.mean()) / h.std(), (d - d.mean()) / d.std()
    specs = {"cnn10": (1, 16), "beats": (13, 8), "wavlm_base_plus": (3, 8), "wavlm_large": (5, 8)}
    for name, (L, D) in specs.items():
        X = rng.normal(0, 1, (len(meta), L, D)).astype(np.float32)
        if name == "cnn10":
            X[:, :, 0] += 1.2 * yv[:, None]                   # bağlamdan bağımsız gerçek sinyal
        elif name == "beats":
            X[:, :, 0] += 1.5 * zh[:, None]                   # yalnız bağlam (saat) …
            X[:, :, 1] += 1.5 * zd[:, None]                   # … ve tarih
        elif name == "wavlm_large":
            X[:, :, 0] += 1.0 * yv[:, None]                   # gerçek sinyal …
            X[:, :, 1] += 1.5 * zh[:, None]                   # … + bağlam
            X[:, :, 2] += 1.5 * zd[:, None]
        dd = work / "emb" / name
        dd.mkdir(parents=True)
        np.save(dd / "rec_win4.npy", X)
        meta.to_csv(dd / "recordings.csv", index=False)
        prim = [0] if L == 1 else list(range(1, L))
        (dd / "info.json").write_text(json.dumps({"primary_layers": prim, "n_params": 1000 * L, "random_init": False,
                                                  "n_store": L, "dim": D, "short_policy": "zeropad"}))
    F = meta[["participant_id", "slot"]].copy()
    F["kept_s"] = 10.0
    for i in range(44):
        F[f"f{i:02d}"] = rng.normal(0, 1, len(F)) + (0.3 * yv if i == 0 else 0)
    F.to_csv(work / "mfcc.csv", index=False)
    chain = np.where(rng.random(len(meta)) < 0.3, "ffmpeg", "apple")
    pd.DataFrame({"participant_id": meta.participant_id, "slot": meta.slot, "chain": chain}).to_csv(work / "index.csv", index=False)
    run(REPO / "scripts/run_probes.py", "--exp", "EXP-016", "--emb-root", work / "emb", "--backbones", ",".join(BB),
        "--mfcc-features", work / "mfcc.csv", "--context", ctx, "--splits-dir", work / "splits", "--out-dir", work / "probe",
        "--report-dir", work / "probe_report", "--n-repeats", "2", "--n-jobs", "2")
    return {"ctx": ctx, "ids": list(C.participant_id)}


def eval_args(work: Path, ctx: Path, out: str = "out", rep: str = "rep") -> list:
    return [REPO / "scripts/evaluate_context.py", "--probe-dir", work / "probe", "--context", ctx, "--splits-dir", work / "splits",
            "--emb-root", work / "emb", "--cache-index", work / "index.csv", "--out-dir", work / out, "--report-dir", work / rep,
            "--backbones", ",".join(BB), "--n-repeats", "2", "--n-boot", "200", "--n-perms", "100", "--n-sim", "40",
            "--g1-repeats", "2", "--n-jobs", "2", "--allow-unapproved"]


def _ref_data(seed: int, sep: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 120
    x1, x2, g = rng.normal(size=n), rng.normal(size=n), (rng.random(n) < 0.4).astype(float)
    eta = -0.3 + 1.0 * x1 - 0.7 * x2 + 0.8 * g
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
    if sep:
        y[g == 1] = 1.0
    return pd.DataFrame({"y": y, "x1": x1.round(6), "x2": x2.round(6), "g": g})


REF = {  # (seed, ayrışma): (katsayılar [sabit, x1, x2, g], x1 / x2 / g için PLR χ²) — R brglm2 + R BFGS
    (11, False): ([-0.31272875, 1.03126063, -0.49284323, 0.49046200], [22.547815, 4.444739, 1.404072]),
    (12, True): ([0.03010580, 0.90922155, -0.80816145, 4.83153194], [10.425138, 9.903606, 39.765945]),
}


def test_firth_reference() -> None:
    for (seed, sep), (coef, chis) in REF.items():
        d = _ref_data(seed, sep)
        X = np.column_stack([np.ones(len(d)), d[["x1", "x2", "g"]].to_numpy(float)])
        f = E.firth_fit(X, d.y.to_numpy())
        assert f["converged"] and np.allclose(f["coef"], coef, rtol=1e-5, atol=1e-6), (f["coef"], coef)
        for j, c in zip([1, 2, 3], chis):
            plr = 2 * (f["pll"] - E.firth_fit(X, d.y.to_numpy(), fixed=j)["pll"])
            assert abs(plr - c) < 1e-3, (seed, j, plr, c)
        if sep:
            m, ok = E.logit_fit(d.y.to_numpy(), X)
            assert (not ok) or abs(m.params[-1]) > 10, "E10: ayrışmalı veride ML'nin patlaması bekleniyordu"
    print("  E10 geçti (Firth = brglm2 katsayıları; PLR = R profil cezalı olabilirlik)", flush=True)


def _cl_data(seed: int, strong: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 130
    g = rng.integers(0, 9, n)
    x2, x3, x4 = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    y = (rng.random(n) < 0.5).astype(int)
    s = (3.5 if strong else 0.6) * (y - 0.5) * 2 + rng.normal(size=n)
    if strong:
        s = np.where(y == 1, np.abs(s) + 0.5, -np.abs(s) - 0.5)
    return pd.DataFrame({"y": y, "s": s.round(6), "x2": x2.round(6), "x3": x3.round(6), "x4": x4.round(6), "g": g})


CL_REF = {  # R 4.3.3 survival 3.5.8: coxph(method = "exact") log-olabilirliği + bilgi matrisi, optim (Nelder–Mead → BFGS)
    (21, False): [1.221158, 0.140951, 0.085228, 0.018194],
    (22, True): [0.757581, -0.336329, 0.348173, 0.219132],
}


def test_clogit_reference() -> None:
    for (seed, strong), ref in CL_REF.items():
        d = _cl_data(seed, strong)
        r = E.clogit_firth(d.y.to_numpy(), d[["s", "x2", "x3", "x4"]].to_numpy(float), d.g.to_numpy())
        assert r["converged"] and np.allclose(r["coef"], ref, atol=1e-3), (seed, r["coef"], ref)
    print("  E13 geçti (cezalı koşullu logit = R survival referansı; ayrışmalı veride sonlu)", flush=True)


BRGLM2_FLAT = [7.135800, 0.416646, -0.112211, -0.529554, -0.176360, -1.380461, -0.300861, 7.266766, -1.008131, 1.209509,
               0.332277]   # R 4.3.3 brglm2 0.9.2, AS_mean, maxit 1e5, epsilon 1e-12: 6105 iterasyonda yakınsadı (2026-10-10)


def _flat_case() -> tuple[np.ndarray, np.ndarray]:
    """Yarı-tam ayrışmalı, cezalı olabilirlik yüzeyi çok düz bir tasarım: sentetik kohortun kalibrasyon simülasyonundan
    bir jackknife alt örneklemi (bir gün dışarıda). Bilgi matrisli Firth adımları burada 3000 adımda bile yakınsamıyor."""
    import statsmodels.api as sm
    C = cohort(np.random.default_rng(0)).dropna(subset=["start_hour"]).copy()
    Xc, y = E.context_matrix(C, 4), C.label.to_numpy(float)
    day = C.recording_date.astype("category").cat.codes.to_numpy()
    q = np.column_stack([E.zstd(C.age), E.zstd(C.start_hour), E.zstd(C.start_hour ** 2), E.zstd(C.days)])
    z = E.zstd(sm.add_constant(q) @ E.firth_fit(sm.add_constant(q), y)["coef"])
    rng = np.random.default_rng(int(np.random.SeedSequence([0, 0]).generate_state(40)[0]))
    X1 = np.column_stack([Xc, E.zstd(z + rng.normal(0, 1, len(y)))])
    keep = day != 134
    X = np.column_stack([np.ones(keep.sum()), np.round(X1[keep][:, 1:], 8)])
    return X, y[keep]


def test_firth_flat() -> None:
    """E17: düz yüzeyde Firth tahmini (Newton-CG cilası) R brglm2 ile aynı; cilasız bilgi-matrisi adımları yakınsamıyor."""
    X, y = _flat_case()
    f = E.firth_fit(X, y)
    assert f["converged"] and f["polished"], f
    assert np.allclose(f["coef"], BRGLM2_FLAT, atol=2e-5), (np.round(f["coef"], 6), BRGLM2_FLAT)
    print(f"  E17 geçti (düz yüzey: Firth = brglm2, ŝ katsayısı {f['coef'][-1]:.5f}; cilalama gerekti)", flush=True)


def test_cluster_se() -> None:
    """E14: gün-kümeli varyanslar bağımsız uygulamayla aynı. Ayrışmasız, büyük örneklemde Firth ≈ ML olduğundan:
    (a) Firth sandviç CR1 = statsmodels ML küme-dayanıklı SE (aynı G/(G−1) düzeltmesi; statsmodels ayrıca (n−1)/(n−k)
    çarpar, oranla düzeltilir); (b) CV3 = statsmodels ML ile birer küme dışarıda bırakılarak elle hesaplanan jackknife."""
    import statsmodels.api as sm
    from scipy import stats
    rng = np.random.default_rng(31)
    G, m = 60, 40
    g = np.repeat(np.arange(G), m)
    u = rng.normal(0, 0.7, G)[g]
    x1, s = rng.normal(size=G * m) + u, rng.normal(size=G * m) + 0.5 * u
    y = (rng.random(G * m) < 1 / (1 + np.exp(-(0.2 + 0.6 * x1 + 0.4 * s + u)))).astype(float)
    Xc = np.column_stack([np.ones(G * m), x1])
    r = E.t1(y, Xc, s, groups=g, ml=False, jackknife=True)
    X1 = np.column_stack([Xc, E.zstd(s)])
    n, k = X1.shape
    ml = sm.Logit(y, X1).fit(disp=0, cov_type="cluster", cov_kwds={"groups": g})
    se_ml = float(ml.bse[-1]) / np.sqrt((n - 1) / (n - k))
    assert abs(r["se_cluster"] / se_ml - 1) < 0.02, (r["se_cluster"], se_ml)
    b_ml = float(sm.Logit(y, X1).fit(disp=0).params[-1])
    bj = np.array([sm.Logit(y[g != j], X1[g != j]).fit(disp=0).params[-1] for j in range(G)])
    se3 = np.sqrt((G - 1) / G * np.sum((bj - b_ml) ** 2))
    assert abs(r["se_cv3"] / se3 - 1) < 0.02 and r["n_clusters"] == G and r["cv3_fails"] == 0, (r["se_cv3"], se3)
    p3 = 2 * stats.t.sf(abs(b_ml) / se3, G - 1)
    assert abs(np.log(r["p_cv3"]) - np.log(p3)) < 0.1, (r["p_cv3"], p3)
    print(f"  E14 geçti (CR1 Firth/ML {r['se_cluster'] / se_ml:.4f}; CV3 Firth/ML {r['se_cv3'] / se3:.4f})", flush=True)


def test_protocol_gate_and_k2_rule() -> None:
    """E15: gerçek veri kipi kapısı ve K2 fail-safe kuralı (birim düzeyi).
    (a) KABUL + sabit ayarlar → geçer; herhangi bir ayar farklıysa (B, permütasyon, tohum, backbone alt kümesi) → reddedilir;
    --allow-unapproved → geçer (rapor uyarısıyla). (b) Diğer koşullar sağlanırken K2 geçersizse C1 'kurulamadı', geçerliyse
    'Aşama 2 bekliyor'; koşul 3 başarısızsa 'sağlanmadı'."""
    from argparse import Namespace
    base = dict(E.FIXED, allow_unapproved=False)
    ok = Namespace(**base)
    assert E.PROTOCOL["status"] == "KABUL"
    E.protocol_gate(ok)
    for k, v in [("n_boot", 200), ("n_perms", 100), ("n_sim", 40), ("seed", 1), ("n_repeats", 2), ("g1_repeats", 2),
                 ("backbones", "cnn10,beats")]:
        try:
            E.protocol_gate(Namespace(**dict(base, **{k: v})))
        except SystemExit as e:
            assert "sapma" in str(e) and k in str(e), str(e)
        else:
            raise AssertionError(f"E15: {k}={v} reddedilmedi")
    E.protocol_gate(Namespace(**dict(base, n_boot=200, allow_unapproved=True)))
    shuffled = Namespace(**dict(base, backbones=",".join(reversed(E.FIXED["backbones"].split(",")))))
    E.protocol_gate(shuffled)                                   # sıra önemli değil
    T = {"x": {"or_sd": 2.0, "p_cv3": 0.01, "p_cluster": 0.001, "or_sd_cluster": 2.0}}
    args = ({"x": 0.001}, {"x": {"lo": 1.3}}, {"x": {"or_sd": 1.9}}, {"x": {"or_sd": 1.5}}, {"x": {"g1_positive": False}}, {})
    assert E.verdicts(T, *args, k2_valid=True)["x"]["c1_status"].startswith("Aşama 2 bekliyor")
    v = E.verdicts(T, *args, k2_valid=False)["x"]
    assert v["c1_status"].startswith("kurulamadı") and not v["stage1_all_met"] and v["conditions"]["3_cluster"] is None, v
    assert v["stage2_triggered"]                                # kabul edilen kural yalnız Holm-T1'e bağlı
    T["x"]["p_cv3"] = 0.2
    assert E.verdicts(T, *args, k2_valid=True)["x"]["c1_status"] == "sağlanmadı"
    T["x"]["p_cv3"] = float("nan")                              # jackknife fit'i başarısız → koşul 3 sağlanmaz (fail-safe)
    assert E.verdicts(T, *args, k2_valid=True)["x"]["c1_status"] == "sağlanmadı"
    print("  E15 geçti (kapı: sabit ayarlar zorunlu; K2 geçersizse C1 kurulamadı)", flush=True)


def test_calibration() -> None:
    """E8: sinyalsiz, bağlamı düzgün kodlayan skorla T1-LRT ret oranı (sentetik bağlam)."""
    rej = []
    for k in range(60):
        rng = np.random.default_rng(100 + k)
        C = cohort(rng).dropna(subset=["start_hour"])
        q = np.column_stack([E.zstd(C.start_hour), E.zstd(C.days), E.zstd(C.age)])
        s = q @ np.array([-1.0, 0.8, 0.3]) + rng.normal(0, 1, len(C))
        r = E.t1(C.label.to_numpy(), E.context_matrix(C, 4), s, ml=False)
        rej.append(r["p_lrt"] < 0.05)
    rate = float(np.mean(rej))
    assert rate <= 0.15, f"E8: T1-PLR sinyalsiz skorda {rate:.2f} oranında reddetti"
    print(f"  E8 geçti (ret oranı {rate:.3f})", flush=True)


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="test_eval016_"))
    try:
        test_firth_reference()
        test_clogit_reference()
        test_firth_flat()
        test_cluster_se()
        test_protocol_gate_and_k2_rule()
        test_calibration()
        inp = write_inputs(work, np.random.default_rng(0))
        r = run(*eval_args(work, inp["ctx"]))
        R = json.loads((work / "rep" / "EVAL-016_context.json").read_text())
        md = (work / "rep" / "EVAL-016_context.md").read_text()
        ci = R["cohort"]
        assert ci["n_cohort"] == N1 + N0 and ci["n_excluded"] == 2 and ci["n_t1"] == N1 + N0 - 2, ci
        print("  E1 geçti", flush=True)
        T, V, G = R["t1"], R["verdicts"], R["g1"]
        t = T["cnn10"]
        assert t["p_lrt"] < 1e-3 and t["or_sd"] > 1 and R["holm"]["cnn10"] < 0.05, t
        assert V["cnn10"]["stage1_all_met"] and V["cnn10"]["stage2_triggered"], V["cnn10"]
        assert V["cnn10"]["c1_status"].startswith("Aşama 2 bekliyor") and R["boot"]["cnn10"]["lo"] > 1
        print(f"  E2 geçti (sinyal: p {t['p_lrt']:.2g}, OR {t['or_sd']:.2f})", flush=True)
        t = T["beats"]
        assert t["p_lrt"] > 0.01, f"E3: yalnız-bağlam kolu T1'de reddedildi (p {t['p_lrt']:.4f})"
        assert G["beats"]["g1_positive"], G["beats"]
        assert "beats" in R["n5"] and R["n5"]["beats"]["p_lrt"] > 0.01, R["n5"].get("beats")
        assert V["beats"]["c1_status"] == "sağlanmadı" and R["c2"]["beats"] == "pozitif"
        assert V["beats"]["conditions"]["7_n5_not_significant"] is True
        print(f"  E3 geçti (bağlam: p {t['p_lrt']:.3f}; G1 Holm p {G['beats']['holm_p']:.2g}; N5 p {R['n5']['beats']['p_lrt']:.3f})", flush=True)
        t = T["wavlm_base_plus"]
        assert t["p_lrt"] > 0.01 and not G["wavlm_base_plus"]["g1_positive"] and "wavlm_base_plus" not in R["n5"]
        assert V["wavlm_base_plus"]["conditions"]["7_n5_not_significant"] is None
        print("  E4 geçti", flush=True)
        assert all(R["holm"][b] >= T[b]["p_lrt"] - 1e-12 for b in BB) and set(R["holm"]) == set(BB)
        assert "mfcc_lr" in T and "mfcc_lr" not in R["holm"]
        assert R["spisak"] and R["spisak"]["cnn10"]["full_p"] < 0.05
        print("  E5 geçti", flush=True)
        t, v = T["wavlm_large"], V["wavlm_large"]
        assert t["p_lrt"] < 1e-3 and G["wavlm_large"]["g1_positive"] and R["n5"]["wavlm_large"]["p_lrt"] < 0.05, R["n5"].get("wavlm_large")
        assert v["conditions"]["7_n5_not_significant"] is False and v["stage1_all_met"] == (not E.PROTOCOL["c1_n5_veto"]), v
        assert R["protocol"]["status"] == E.PROTOCOL["status"]
        assert R["settings_fixed"] is False and "karar için KULLANILAMAZ" in md, "E12: test kipi raporda işaretlenmedi"
        gate = run(*[a for a in eval_args(work, inp["ctx"], "out4", "rep4") if a != "--allow-unapproved"], ok=False)
        msg = gate.stdout + gate.stderr
        if E.PROTOCOL["status"] != "KABUL":
            assert gate.returncode != 0 and "onay bekliyor" in msg, "E12: onay kapısı çalışmadı"
        else:
            assert gate.returncode != 0 and "Sabitlenmiş analiz ayarlarından sapma" in msg, "E12: ayar kapısı çalışmadı"
        assert not (work / "rep4").exists(), "E12: reddedilen çalışma rapor yazdı"
        print(f"  E12 geçti (sinyal + bağlam: T1 p {t['p_lrt']:.2g}; N5 p {R['n5']['wavlm_large']['p_lrt']:.2g} → yanlış alarm; "
              f"N5 veto={E.PROTOCOL['c1_n5_veto']}; protokol {E.PROTOCOL['status']}; test ayarlarıyla gerçek kip reddedildi)", flush=True)
        kv = R["calibration"]["k2_validity"]
        assert kv["valid"] and {"S1_smooth", "S2_step", "S3_day"} <= set(R["calibration"]), kv
        assert all(np.isfinite(T[a]["p_cv3"]) and T[a]["cv3_fails"] == 0 for a in T)
        print(f"  E16 geçti (kalibrasyonda CV3 hesaplandı, K2 kuralı değerlendirildi: {kv['cv3_wilson_lower']})", flush=True)
        l3 = R["l3_partial"]
        assert l3["cnn10"]["context"]["p"] > 0.01 and R["c2"]["cnn10"] == "gösterilemedi", (l3["cnn10"], R["c2"]["cnn10"])
        assert l3["beats"]["context"]["p"] < 0.02
        print(f"  E11 geçti (kısmi p: sinyal {l3['cnn10']['context']['p']:.3f}, bağlam {l3['beats']['context']['p']:.3f}; "
              f"mlconfound kısmi p sinyal kolunda {R['spisak']['cnn10']['partial_p']:.3f} — yalnız bilgi)", flush=True)
        assert not ids_in(md + json.dumps(R), inp["ids"]), "E6: raporda ID"
        sc = pd.read_csv(work / "out" / "eval016_participant_scores.csv")
        assert len(sc) == N1 + N0 - 2 and set(BB) <= set(sc.columns)
        print("  E6 geçti", flush=True)
        assert MISSING[0] in set(sc.participant_id) and sc.loc[sc.participant_id == MISSING[0], "cnn10"].notna().all()
        print("  E9 geçti", flush=True)
        ctx2 = work / "ctx_changed.csv"
        pd.read_csv(inp["ctx"]).assign(extra=1).to_csv(ctx2, index=False)
        r2 = run(*eval_args(work, ctx2, "out2", "rep2"), ok=False)
        assert r2.returncode != 0 and "imza uyuşmuyor" in (r2.stdout + r2.stderr), "E7: imza kontrolü çalışmadı"
        (work / "probe" / "partial" / "beats_r1_test.csv").unlink()
        r3 = run(*eval_args(work, inp["ctx"], "out3", "rep3"), ok=False)
        assert r3.returncode != 0 and "ara sonucu yok" in (r3.stdout + r3.stderr), "E7: eksik ara sonuç fark edilmedi"
        print("  E7 geçti", flush=True)
        print("Tüm EVAL-016 kontrolleri geçti (E1–E17). Not: sentetik veri; gerçek veride hiçbir sonuç üretilmedi.")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
