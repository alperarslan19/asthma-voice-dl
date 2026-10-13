"""
Keşifsel gösterim — Alper'in kendi sesi + veri setinden arka planlar → modellerin skoru (EXP-033; istatistik YOK)

BİLİMSEL AMAÇ (keşifsel)
    Aynı ses kaydına farklı bağışçıların arka planı eklendiğinde modellerin astım skoru ne kadar ve hangi yöne kayıyor?
    Tek konuşmacı, az sayıda arka plan, çıkarım yok: amaç gözle görmek ve şüphe / hipotez üretmek. Okunacak şey mutlak
    skor değil, aynı kaydın "orig" sürümüne göre KAYMA (logit farkı).

KARIŞTIRMA (kodla, deterministik)
    ses (m4a) → 48 kHz mono → önbellekteki gibi kırpma sınırları + 32 kHz model görünümünde tepe −1 dBFS olacak kazanç
    → bağışçının SEVİYESİ KORUNMUŞ arka planı (experiments/background_v1/tracks_32k.f32; bağışçının konuşma tepesi
    −1 dBFS ölçeğinde) 48 kHz'e çıkarılıp döngülenir ve 0 dB (+ isteğe bağlı kazançlar) ile eklenir → sonra veri setiyle
    BİREBİR aynı önbellek zinciri (build_audio_cache.trim_bounds + render: 32 / 16 kHz, 11 kHz FIR, −1 dBFS).
    Böylece bağışçının konuşma / arka plan oranı korunur.

EZBERLEME ÖNLEMİ
    Bütün modeller (MFCC-LR, dondurulmuş gömme + LR, fine-tune) BAĞIŞÇILAR ve onların KAYIT GÜNLERİNDEKİ herkes
    çıkarılarak yeniden eğitilir; 14 sürüm aynı modele girer. C seçimi: eğitim kümesinde gün-gruplu iç 5-fold.

ALT KOMUTLAR
    mix         karışımları üret (önbellek biçiminde: index.csv + audio_32k.f32 + audio_16k.f32) + dinleme wav'ları
    score-mfcc  MFCC-44 (cache modu) + LR, slot başına
    score-frozen --backbone B --mix-emb DIR   (önce extract_embeddings.py ile karışımların gömmeleri çıkarılır)
    score-ft    --arm cnn10|cnn14|beats       (GPU; ana önbellek yerelde olmalı)
Çıktılar (Drive; katılımcı ID'si yalnız *_PRIVATE dosyalarında): <out>/scores_*.csv, mixes_wav/, mixcache/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_audio import decode  # noqa: E402
from build_audio_cache import render, trim_bounds  # noqa: E402

ITEMS = {1: "aaa", 2: "araba", 3: "ana", 4: "ordu", 5: "gelecek", 6: "titiz", 7: "unlem"}   # veri seti slotları
PSEUDO_PID0 = 900000                                    # karışımlar için sahte kimlik (gerçek kimliklerle çakışmaz)
C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
XF = 320                                                # 10 ms @ 32 kHz eşit güçlü çapraz geçiş (döngü)


# ----------------------------------------------------------------------------- girdiler
def slot_of(name: str) -> int:
    """Dosya adından slot: 'aaaaaa', 'Ünlem', 'ünlem' (NFD dahil) … → 1..7."""
    s = unicodedata.normalize("NFKD", Path(name).stem).encode("ascii", "ignore").decode().lower().strip()
    s = "".join(ch for ch in s if ch.isalpha())
    if s and set(s) == {"a"}:
        return 1
    for k, v in ITEMS.items():
        if k > 1 and s.startswith(v):
            return k
    raise ValueError(f"tanınmayan ses adı: {name!r} (beklenen: aaa, araba, ana, ordu, gelecek, titiz, ünlem)")


def period_of(C: pd.DataFrame) -> pd.Series:
    rd = pd.to_datetime(C.recording_date)
    last = rd[C.label == 0].max()
    return pd.Series(np.select([rd.isna(), rd <= last], ["bilinmiyor", "erken"], "geç"), index=C.index)


def donors(bg_dir: Path, key: Path, ctx: pd.DataFrame) -> pd.DataFrame:
    codes = sorted({p.stem for p in bg_dir.glob("*.wav")})
    assert codes, f"{bg_dir}: .wav yok"
    K = pd.read_csv(key)
    D = K[K.code.isin(codes)].copy()
    miss = set(codes) - set(D.code)
    assert not miss, f"dinleme anahtarında olmayan kod: {sorted(miss)}"
    D["group"] = D.participant_id.map(ctx.label).map({1: "astım", 0: "sağlıklı"})
    D["period"] = D.participant_id.map(period_of(ctx))
    D["day"] = D.participant_id.map(ctx.recording_date)
    assert D.group.notna().all(), "bağışçı bağlam tablosunda yok"
    return D.sort_values("code").reset_index(drop=True)


def exclusion(D: pd.DataFrame, ctx: pd.DataFrame) -> set:
    """Bağışçılar + bağışçıların kayıt günlerinde kaydedilen herkes (gün izi ezberlenmiş olmasın)."""
    days = set(D.day.dropna())
    return set(D.participant_id) | set(ctx.index[ctx.recording_date.isin(days)])


def loop_to(x: np.ndarray, n: int) -> np.ndarray:
    """Tek parçayı eşit güçlü çapraz geçişle döngüleyerek n örneğe uzatır."""
    t = np.linspace(0, np.pi / 2, XF, endpoint=False)
    fo, fi = np.cos(t), np.sin(t)
    out, tail = [x[:-XF]], x[-XF:]
    total = len(x) - XF
    while total < n:
        out.append(np.concatenate([tail * fo + x[:XF] * fi, x[XF:-XF]]))
        tail = x[-XF:]
        total += len(x) - 2 * XF
    return np.concatenate(out)[:n]


def voice_model_view(x48: np.ndarray) -> tuple[np.ndarray, float]:
    """Sesi, önbellek zincirinde (kırpma + 32 kHz + 11 kHz FIR) tepesi −1 dBFS olacak biçimde ölçekler (48 kHz'te)."""
    b = trim_bounds(x48, 48000)
    _, g_db = render(x48, b, 32000)
    return (x48 * 10 ** (g_db / 20)).astype(np.float32), g_db


def mix_one(v48: np.ndarray, bg32: np.ndarray | None, gain_db: float) -> np.ndarray:
    if bg32 is None:
        return v48
    n32 = int(np.ceil(len(v48) * 2 / 3)) + 64
    b48 = resample_poly(loop_to(bg32.astype(np.float64), n32), 3, 2)[:len(v48)]
    assert len(b48) == len(v48)
    return (v48 + b48 * 10 ** (gain_db / 20)).astype(np.float32)


# ----------------------------------------------------------------------------- mix
def cmd_mix(a) -> None:
    import soundfile as sf
    ctx = pd.read_csv(a.context).set_index("participant_id")
    D = donors(a.bg_dir, a.key, ctx)
    TR = pd.read_csv(a.tracks_dir / "tracks_index.csv").set_index(["participant_id", "variant"])
    mm = np.memmap(a.tracks_dir / "tracks_32k.f32", dtype="<f4", mode="r")
    voices = {slot_of(p.name): p for p in sorted(a.voice_dir.iterdir()) if p.suffix.lower() in {".m4a", ".wav", ".mp3", ".aif", ".aiff"}}
    assert len(voices) == len({slot_of(p.name) for p in a.voice_dir.iterdir() if p.suffix.lower() in {".m4a", ".wav", ".mp3", ".aif", ".aiff"}}), \
        "aynı slot için birden çok ses dosyası"
    print(f"sesler: { {ITEMS[k]: p.name for k, p in sorted(voices.items())} }")
    print(D[["code", "variant", "group", "period"]].to_string(index=False))
    excl = exclusion(D, ctx)
    out = a.out_dir
    (out / "mixcache").mkdir(parents=True, exist_ok=True)
    (out / "mixes_wav").mkdir(exist_ok=True)
    f32, f16 = open(out / "mixcache" / "audio_32k.f32", "wb"), open(out / "mixcache" / "audio_16k.f32", "wb")
    o32 = o16 = 0
    rows, i = [], 0
    for slot, path in sorted(voices.items()):
        x = decode(path, 48000, 1)[:, 0].astype(np.float32)
        assert len(x) > 48000 * 2, f"{path.name}: çok kısa"
        v48, g_v = voice_model_view(x)
        versions = [("orig", None, 0.0)] + [(r.code, r, g) for r in D.itertuples() for g in a.gains]
        for code, r, g in versions:
            bg = None
            if r is not None:
                t = TR.loc[(r.participant_id, r.variant)]
                bg = np.asarray(mm[int(t.offset): int(t.offset) + int(t.n)], dtype=np.float64)
            mix = mix_one(v48, bg, g)
            b = trim_bounds(mix, 48000)
            y32, g32 = render(mix, b, 32000)
            y16, g16 = render(mix, b, 16000)
            assert np.isfinite(y32).all() and np.isfinite(y16).all()
            f32.write(y32.astype("<f4").tobytes()); f16.write(y16.astype("<f4").tobytes())
            tag = code if (r is None or len(a.gains) == 1) else f"{code}_g{g:+.0f}"
            sf.write(out / "mixes_wav" / f"{slot}_{ITEMS[slot]}__{tag}.wav", y32, 32000, subtype="PCM_16")
            rows.append({"participant_id": PSEUDO_PID0 + i, "slot": slot, "item": ITEMS[slot], "bg": code, "tag": tag,
                         "gain_db": g if r is not None else np.nan, "bg_variant": r.variant if r is not None else "",
                         "bg_group": r.group if r is not None else "orijinal", "bg_period": r.period if r is not None else "",
                         "voice_gain_db": g_v, "trim_start_s": b["trim_start_s"], "trim_end_s": b["trim_end_s"],
                         "offset_32k": o32, "n_32k": len(y32), "offset_16k": o16, "n_16k": len(y16)})
            o32 += len(y32); o16 += len(y16); i += 1
    f32.close(); f16.close()
    idx = pd.DataFrame(rows)
    idx.to_csv(out / "mixcache" / "index.csv", index=False)
    D.to_csv(out / "donors_PRIVATE.csv", index=False)                                  # ID'li → yalnız Drive
    pd.Series(sorted(excl), name="participant_id").to_csv(out / "excluded_PRIVATE.csv", index=False)
    info = {"n_mixes": len(idx), "n_items": idx.slot.nunique(), "n_backgrounds": len(D), "gains_db": a.gains,
            "n_excluded_participants": len(excl), "excluded_by_label": ctx.loc[sorted(excl & set(ctx.index)), "label"].value_counts().to_dict(),
            "n_excluded_days": len(set(D.day.dropna()))}
    (out / "mix_info.json").write_text(json.dumps(info, indent=1, default=str))
    print(json.dumps(info, indent=1, default=str))


# ----------------------------------------------------------------------------- ortak eğitim
def train_cohort(ctx: pd.DataFrame, splits_dir: Path, out: Path) -> pd.Index:
    cohort = set(pd.read_csv(splits_dir / "outer_r0.csv").participant_id)
    excl = set(pd.read_csv(out / "excluded_PRIVATE.csv").participant_id)
    tr = pd.Index(sorted(cohort - excl))
    assert len(tr) >= 0.5 * len(cohort) and ctx.loc[tr, "label"].nunique() == 2, f"eğitim kümesi küçük: {len(tr)} / {len(cohort)}"
    return tr


def fit_lr(pipe, grid, X, y, days, seed: int = 0):
    from make_splits import grouped_folds
    from sklearn.model_selection import GridSearchCV, PredefinedSplit
    inner = grouped_folds(y, days, 5, seed)
    gs = GridSearchCV(pipe, grid, cv=PredefinedSplit(inner), scoring="neg_log_loss").fit(X, y)
    return gs


def save_scores(rows: list[dict], path: Path) -> None:
    S = pd.DataFrame(rows)
    S["logit"] = np.log(np.clip(S.prob, 1e-6, 1 - 1e-6) / (1 - np.clip(S.prob, 1e-6, 1 - 1e-6))) if "logit" not in S else S.logit
    S.to_csv(path, index=False)
    print(f"yazıldı: {path.name} ({len(S)} satır)")


def day_of(ctx, pids):
    d = ctx.loc[pids, "recording_date"].astype("string").fillna("nodate")
    return np.where(d == "nodate", "nodate_" + pd.Index(pids).astype(str), d)


# ----------------------------------------------------------------------------- MFCC
def cmd_score_mfcc(a) -> None:
    import run_probes as RP
    from extract_mfcc_features import FEATURE_NAMES, cache_signal, features
    ctx = pd.read_csv(a.context).set_index("participant_id")
    tr = train_cohort(ctx, a.splits_dir, a.out_dir)
    F = pd.read_csv(a.features)
    idx = pd.read_csv(a.out_dir / "mixcache" / "index.csv")
    a32 = np.memmap(a.out_dir / "mixcache" / "audio_32k.f32", dtype="<f4", mode="r")
    Xm = np.stack([features(cache_signal(a32, int(r.offset_32k), int(r.n_32k))) for r in idx.itertuples()])
    rows = []
    for slot in sorted(idx.slot.unique()):
        f = F[(F.slot == slot) & F.participant_id.isin(tr)]
        y = ctx.loc[f.participant_id, "label"].to_numpy().astype(int)
        gs = fit_lr(RP.lr_pipe(), {"clf__C": C_GRID}, f[FEATURE_NAMES].to_numpy(), y, day_of(ctx, f.participant_id))
        m = (idx.slot == slot).to_numpy()
        p = gs.predict_proba(Xm[m])[:, 1]
        for r, pr in zip(idx[m].itertuples(), p):
            rows.append({"tag": r.tag, "slot": slot, "model": "MFCC-44 + LR", "family": "MFCC", "prob": float(pr),
                         "n_train": int(len(y)), "C": float(gs.best_params_["clf__C"])})
        print(f"  slot {slot}: n_train {len(y)} (astım {y.sum()}), C {gs.best_params_['clf__C']}")
    save_scores(rows, a.out_dir / "scores_mfcc.csv")


# ----------------------------------------------------------------------------- dondurulmuş gömmeler
NICE = {"cnn10": "CNN10", "cnn14": "CNN14", "beats": "BEATs", "wavlm_base_plus": "WavLM Base+", "wavlm_large": "WavLM Large"}


def cmd_score_frozen(a) -> None:
    import run_probes as RP
    ctx = pd.read_csv(a.context).set_index("participant_id")
    tr = train_cohort(ctx, a.splits_dir, a.out_dir)
    arm = RP.emb_arm(a.emb_root, a.backbone, "win4", None)
    layers = arm.signature["layers"]
    idx = pd.read_csv(a.out_dir / "mixcache" / "index.csv")
    E = np.load(a.mix_emb / "rec_win4.npy")
    meta = pd.read_csv(a.mix_emb / "recordings.csv")
    assert len(meta) == len(idx) and (meta.participant_id.to_numpy() == idx.participant_id.to_numpy()).all()
    Xm = E[:, layers, :].reshape(len(E), -1)
    assert Xm.shape[1] == arm.X[1].shape[1] and np.isfinite(Xm).all()
    rows = []
    for slot in sorted(idx.slot.unique()):
        keep = np.isin(arm.pids[slot], tr)
        pid, X = arm.pids[slot][keep], arm.X[slot][keep]
        y = ctx.loc[pid, "label"].to_numpy().astype(int)
        gs = fit_lr(arm.pipe, arm.grid, X, y, day_of(ctx, pid))
        m = (idx.slot == slot).to_numpy()
        p = gs.predict_proba(Xm[m])[:, 1]
        for r, pr in zip(idx[m].itertuples(), p):
            rows.append({"tag": r.tag, "slot": slot, "model": f"{NICE[a.backbone]} (dondurulmuş) + LR", "family": "dondurulmuş",
                         "prob": float(pr), "n_train": int(len(y)), "C": float(gs.best_params_["clf__C"])})
        print(f"  {a.backbone} slot {slot}: n_train {len(y)}, C {gs.best_params_['clf__C']}")
    save_scores(rows, a.out_dir / f"scores_frozen_{a.backbone}.csv")


# ----------------------------------------------------------------------------- fine-tune
def cmd_score_ft(a) -> None:
    import types
    import torch
    import backbones as B
    import finetune as FT
    from smoke_test_backbones import Cache
    ctx = pd.read_csv(a.context).set_index("participant_id")
    tr = train_cohort(ctx, a.splits_dir, a.out_dir)
    cache = Cache(a.cache_dir)
    idx = cache.idx.reset_index().rename(columns={"index": "row"})
    rows_train = idx[idx.participant_id.isin(tr)].row.to_numpy()
    label = ctx.label.astype(int)
    y = label[idx.set_index("row").participant_id[rows_train].to_numpy()].to_numpy()
    fd = types.SimpleNamespace(rows_train=rows_train, rows_eval=np.array([], dtype=int), label=label,
                               pid_of_row=idx.set_index("row").participant_id, pos_weight=float((1 - y).sum() / y.sum()))
    cal = json.loads((a.ft_root / a.arm / "calibration" / "calibration.json").read_text())
    E = FT.select_epochs(cal)["epochs"]
    seed = 4242
    sig = hashlib.sha256((a.arm + ",".join(map(str, sorted(tr))) + str(E)).encode()).hexdigest()[:12]
    wd = a.work_dir / f"{a.arm}_{sig}"
    print(f"{a.arm}: {E} epoch (D-043 kalibrasyonundan), eğitim {len(tr)} katılımcı / {len(rows_train)} kayıt "
          f"(astım kaydı {int(y.sum())}), pos_weight {fd.pos_weight:.3f}; çalışma klasörü {wd.name}", flush=True)
    B.set_determinism(seed)
    m = FT.build_model(a.arm, a.model_dir, a.device)
    res = FT.run_training(m, fd, cache, FT.ARMS[a.arm], E, seed, wd, a.device, ckpt_dir=wd)
    assert res["done"]
    mix = Cache(a.out_dir / "mixcache")
    P = FT.predict_recordings(m, mix, list(range(len(mix.idx))), a.device, amp=False)
    mi = mix.idx
    rows = [{"tag": mi.tag.iloc[r], "slot": int(mi.slot.iloc[r]), "model": f"{NICE[FT.ARMS[a.arm]['backbone']]} (fine-tune)",
             "family": "fine-tune", "logit": float(lg), "prob": float(1 / (1 + np.exp(-lg))), "n_train": int(len(tr)),
             "C": np.nan} for r, lg in zip(P.row, P.logit)]
    save_scores(rows, a.out_dir / f"scores_ft_{a.arm}.csv")


def load_all(out: Path) -> pd.DataFrame:
    """Arayüz için: bütün skor dosyaları + karışım bilgisi + orijinale göre kayma (aynı slot, aynı model)."""
    idx = pd.read_csv(out / "mixcache" / "index.csv")
    S = pd.concat([pd.read_csv(p) for p in sorted(out.glob("scores_*.csv"))], ignore_index=True)
    S = S.merge(idx[["tag", "slot", "item", "bg", "gain_db", "bg_variant", "bg_group", "bg_period"]], on=["tag", "slot"],
                validate="many_to_one")
    o = S[S.bg == "orig"].set_index(["slot", "model"]).logit
    S["delta_logit"] = S.logit - o.reindex(pd.MultiIndex.from_frame(S[["slot", "model"]])).to_numpy()
    S["delta_prob"] = S.prob - S[S.bg == "orig"].set_index(["slot", "model"]).prob.reindex(
        pd.MultiIndex.from_frame(S[["slot", "model"]])).to_numpy()
    return S


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mix")
    m.add_argument("--voice-dir", type=Path, required=True)
    m.add_argument("--bg-dir", type=Path, required=True, help="yalnız KOD listesi için (dosya adları L03b.wav …)")
    m.add_argument("--key", type=Path, required=True, help="background_v1/listening/listening_key.csv")
    m.add_argument("--tracks-dir", type=Path, required=True, help="background_v1 (seviye korunmuş izler)")
    m.add_argument("--context", type=Path, required=True)
    m.add_argument("--gains", type=float, nargs="+", default=[0.0])
    m.add_argument("--out-dir", type=Path, required=True)
    for name in ("score-mfcc", "score-frozen", "score-ft"):
        p = sub.add_parser(name)
        p.add_argument("--context", type=Path, required=True)
        p.add_argument("--splits-dir", type=Path, required=True)
        p.add_argument("--out-dir", type=Path, required=True)
        if name == "score-mfcc":
            p.add_argument("--features", type=Path, required=True, help="data_derived/mfcc/features_cache.csv")
        if name == "score-frozen":
            p.add_argument("--backbone", required=True, choices=list(NICE))
            p.add_argument("--emb-root", type=Path, required=True, help="data_derived/embeddings_v1")
            p.add_argument("--mix-emb", type=Path, required=True)
        if name == "score-ft":
            p.add_argument("--arm", required=True, choices=["cnn10", "cnn14", "beats"])
            p.add_argument("--cache-dir", type=Path, required=True, help="ana önbellek (yerel kopya)")
            p.add_argument("--ft-root", type=Path, required=True, help="experiments/phase3/finetune (kalibrasyon)")
            p.add_argument("--model-dir", type=Path, required=True)
            p.add_argument("--work-dir", type=Path, required=True)
            p.add_argument("--device", default="cuda")
    a = ap.parse_args()
    {"mix": cmd_mix, "score-mfcc": cmd_score_mfcc, "score-frozen": cmd_score_frozen, "score-ft": cmd_score_ft}[a.cmd](a)


if __name__ == "__main__":
    main()
