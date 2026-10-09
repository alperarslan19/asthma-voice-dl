"""
Faz 2 — Dondurulmuş gömme çıkarımı (D-034 madde 1, 2, 9; docs/PHASE2_DESIGN.md Bölüm 5.2–5.3, 5.9)

BİLİMSEL AMAÇ
    Harmonize önbellekteki (D-030) her kaydı, ağırlıkları DEĞİŞMEYEN bir backbone'dan geçirip katman katman,
    zaman ortalamalı bir gömmeye dönüştürmek. Etiket okunmaz; bizim veride hiçbir parametre öğrenilmez (V3).
    Bu yüzden çıkarım split'ten önce bir kez yapılır ve bütün fold'lar aynı gömmeleri kullanır.

GÖRÜNÜMLER
    win4  4 s pencere / 2 s adım, son pencere sona hizalı. Kayıt gömmesi = pencere gömmelerinin ortalaması
          (birincil, EXP-016/017). 4 s'den kısa kayıtlar (önbellekte 17) tek parça; İKİ politikayla da hesaplanır:
            zeropad  D-015: 4 s'ye sağdan sıfır dolgusu (Boll). BEATs/WavLM'de resmi maske + yalnız gerçek çerçevelerde
                     ortalama; PANNs'te maske yok (resmi API) → dolgu çerçeveleri havuzlamaya girer
            nopad    dolgusuz, gerçek uzunlukta tek girdi
          --short-policy birincil olanı seçer (rec_win4.npy); diğeri short_alt_win4.npz'ye yazılır (duyarlılık, EXP-016S).
          Batch: bir kaydın pencereleri tek batch'tir; kısa kayıt tek satırlık batch'tir. Farklı uzunluklar asla
          aynı batch'e girmez (dolgu yalnız zeropad politikasında ve yalnız kısa kayıtta).
    full  tüm kayıt tek girdi, dolgusuz (keşifsel, EXP-018)

ÇIKTILAR (<out_dir> = Drive data_derived/embeddings_v1/<backbone>/; katılımcı düzeyi → git'e GİRMEZ, D-014)
    rec_win4.npy  float32 (kayıt, n_store, dim)      rec_full.npy  float32 (aynı)
    short_alt_win4.npz  kısa kayıtlar için diğer politikanın gömmeleri: rows, emb (k, n_store, dim), policy
    win4_windows.npy float16 (pencere, n_store, dim) + windows.csv  — yalnız --save-windows ile
    recordings.csv  row, participant_id, slot, n_samples, dur_s, n_windows, short   (sıra = önbellek index.csv sırası)
    info.json  backbone, checkpoint sha256, yükleme bilgisi, sürümler, git commit, önbellek index sha256, süre
    MANIFEST.sha256
    shards/    100 kayıtlık ara parçalar (atomik yazım; yeniden çalıştırmada biten parçalar atlanır)

KONTROLLER
    her kayıt: pencere sayısı = window_starts(); çıktı şekli/sonluluk (backbones.embed içinde)
    sonda: kayıt sayısı = index satır sayısı; parça kimlikleri index ile birebir; NaN/Inf yok
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backbones as B  # noqa: E402
from smoke_test_backbones import HOP_S, WIN_S, Cache  # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).resolve().parent).stdout.strip()
    except Exception:
        return ""


def save_npz_atomic(path: Path, **arrays) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


POLICIES = ("zeropad", "nopad")


def short_embeddings(bb, x: torch.Tensor, win: int) -> dict:
    """Kısa kayıt (len(x) < win) için iki politikanın gömmesi: {politika: (1, n_store, dim)}."""
    n = len(x)
    assert 0 < n < win
    xp = torch.zeros(1, win, dtype=torch.float32)
    xp[0, :n] = x
    return {"nopad": bb.embed(x[None, :]), "zeropad": bb.embed(xp, lengths=torch.tensor([n]))}


def extract_shard(bb, cache: Cache, rows: list[int], views: list[str], save_windows: bool, short_policy: str) -> dict:
    assert short_policy in POLICIES
    alt_policy = [p for p in POLICIES if p != short_policy][0]
    out = {v: [] for v in views}
    n_win, wins, win_rec, win_start, short_rows, short_alt = [], [], [], [], [], []
    win = int(WIN_S * bb.sr)
    for r in rows:
        x = torch.from_numpy(cache.audio(r, bb.sr))
        if "win4" in views:
            ws = B.window_starts(len(x), win, int(HOP_S * bb.sr))
            if len(x) < win:                                 # kısa kayıt: tek parça, iki politika
                assert ws == [(0, len(x))]
                both = short_embeddings(bb, x, win)
                E = both[short_policy]
                short_rows.append(r)
                short_alt.append(both[alt_policy][0].numpy())
            else:
                W = torch.stack([x[s:s + n] for s, n in ws])     # (pencere, win) — hepsi tam 4 s, dolgu yok
                assert W.shape == (len(ws), win)
                E = bb.embed(W)                                  # (pencere, n_store, dim)
            out["win4"].append(E.mean(0).numpy())
            n_win.append(len(ws))
            if save_windows:
                wins.append(E.numpy().astype(np.float16))
                win_rec += [r] * len(ws)
                win_start += [s for s, _ in ws]
        if "full" in views:
            out["full"].append(bb.embed(x[None, :])[0].numpy())
    shard = {f"rec_{v}": np.stack(out[v]).astype(np.float32) for v in views}
    shard["rows"] = np.asarray(rows, dtype=np.int64)
    shard["n_windows"] = np.asarray(n_win if n_win else [0] * len(rows), dtype=np.int32)
    shard["short_rows"] = np.asarray(short_rows, dtype=np.int64)
    shard["short_alt"] = (np.stack(short_alt).astype(np.float32) if short_alt
                          else np.zeros((0, bb.n_store, bb.dim), np.float32))
    if save_windows and wins:
        shard["win"] = np.concatenate(wins)
        shard["win_rec"] = np.asarray(win_rec, dtype=np.int64)
        shard["win_start"] = np.asarray(win_start, dtype=np.int64)
    for k, v in shard.items():
        if v.dtype.kind == "f":
            assert np.isfinite(v).all(), f"parça {rows[0]}: {k} sonlu değil"
    return shard


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=list(B.BACKBONES))
    ap.add_argument("--model-dir", type=Path)
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--views", default="win4,full")
    ap.add_argument("--save-windows", action="store_true")
    ap.add_argument("--short-policy", default="zeropad", choices=list(POLICIES),
                    help="4 s'den kısa kayıtlar için birincil politika (D-015 = zeropad; D-034 onayına bağlı)")
    ap.add_argument("--shard-size", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0, help="yalnız ilk N kayıt (test)")
    ap.add_argument("--random-init", action="store_true", help="yalnız birim testi")
    args = ap.parse_args()
    views = args.views.split(",")
    assert set(views) <= {"win4", "full"} and views
    B.check_vendored()
    B.set_determinism()
    cache = Cache(args.cache_dir)
    n = len(cache.idx) if not args.limit else min(args.limit, len(cache.idx))
    assert not cache.idx.duplicated(["participant_id", "slot"]).any()
    shard_dir = args.out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    bb = B.build_backbone(args.backbone, model_dir=args.model_dir, device=args.device, random_init=args.random_init)
    print(f"{args.backbone}: n_store={bb.n_store} dim={bb.dim} sr={bb.sr} params={bb.n_params() / 1e6:.1f}M "
          f"cihaz={args.device} kayıt={n}", flush=True)
    # Parça imzası: farklı ağırlık / önbellek / pencere ayarıyla yazılmış eski parçalar sessizce karışmasın
    sig = json.dumps({"backbone": args.backbone, "checkpoint_sha256": bb.info.get("checkpoint_sha256", ""),
                      "random_init": args.random_init, "cache_index_sha256": B.sha256_file(args.cache_dir / "index.csv"),
                      "window_s": WIN_S, "hop_s": HOP_S, "n_store": bb.n_store, "dim": bb.dim,
                      "short_policy": args.short_policy}, sort_keys=True)
    starts = list(range(0, n, args.shard_size))
    for i, s in enumerate(starts):
        rows = list(range(s, min(s + args.shard_size, n)))
        path = shard_dir / f"shard_{s:05d}.npz"
        if path.exists():
            z = np.load(path)
            ok = np.array_equal(z["rows"], rows) and all(f"rec_{v}" in z for v in views) and \
                (not args.save_windows or "win" in z) and "sig" in z and str(z["sig"]) == sig
            if ok:
                continue
            print(f"  {path.name} uyumsuz → yeniden üretiliyor", flush=True)
        t1 = time.time()
        save_npz_atomic(path, sig=np.array(sig), **extract_shard(bb, cache, rows, views, args.save_windows, args.short_policy))
        done = min(s + args.shard_size, n)
        print(f"  parça {i + 1}/{len(starts)} ({done}/{n} kayıt) {time.time() - t1:.0f} s · toplam {time.time() - t0:.0f} s",
              flush=True)

    # birleştir
    Z = [np.load(shard_dir / f"shard_{s:05d}.npz") for s in starts]
    assert all(str(z["sig"]) == sig for z in Z), "parça imzaları uyuşmuyor"
    rows = np.concatenate([z["rows"] for z in Z])
    assert np.array_equal(rows, np.arange(n)), "parçalar index sırasını kapsamıyor"
    files = {}
    for v in views:
        arr = np.concatenate([z[f"rec_{v}"] for z in Z])
        assert arr.shape == (n, bb.n_store, bb.dim) and np.isfinite(arr).all()
        np.save(args.out_dir / f"rec_{v}.npy", arr)
        files[f"rec_{v}.npy"] = arr.shape
    meta = cache.idx.iloc[:n][["participant_id", "slot"]].copy()
    meta.insert(0, "row", np.arange(n))
    meta["n_samples"] = cache.idx.iloc[:n][f"n_{bb.sr // 1000}k"].to_numpy()
    meta["dur_s"] = meta.n_samples / bb.sr
    meta["n_windows"] = np.concatenate([z["n_windows"] for z in Z])
    meta["short"] = meta.n_samples < int(WIN_S * bb.sr)
    if "win4" in views:
        exp_nw = [len(B.window_starts(int(m), int(WIN_S * bb.sr), int(HOP_S * bb.sr))) for m in meta.n_samples]
        assert (meta.n_windows.to_numpy() == exp_nw).all(), "pencere sayısı beklenenden farklı"
    meta.to_csv(args.out_dir / "recordings.csv", index=False)
    alt_policy = [p for p in POLICIES if p != args.short_policy][0]
    if "win4" in views:
        srows = np.concatenate([z["short_rows"] for z in Z])
        assert np.array_equal(srows, meta.row[meta.short].to_numpy()), "kısa kayıt listesi index ile uyuşmuyor"
        save_npz_atomic(args.out_dir / "short_alt_win4.npz", rows=srows, emb=np.concatenate([z["short_alt"] for z in Z]),
                        policy=np.array(alt_policy))
        files["short_alt_win4.npz"] = (len(srows), bb.n_store, bb.dim)
    if args.save_windows:
        win = np.concatenate([z["win"] for z in Z])
        np.save(args.out_dir / "win4_windows.npy", win)
        pd.DataFrame({"row": np.concatenate([z["win_rec"] for z in Z]),
                      "start_sample": np.concatenate([z["win_start"] for z in Z])}).to_csv(args.out_dir / "windows.csv", index=False)
        files["win4_windows.npy"] = win.shape
    import transformers
    import torchaudio
    info = {**{k: v for k, v in bb.info.items()}, "views": views, "window_s": WIN_S, "hop_s": HOP_S,
            "last_window": "end-aligned", "short_policy": args.short_policy, "short_alt_policy": alt_policy,
            "short_policy_note": "zeropad: 4 s'ye sıfır dolgusu; BEATs/WavLM resmi maske + maskeli ortalama; PANNs maskesiz. "
                                 "nopad: gerçek uzunluk", "precision": "fp32",
            "n_recordings": int(n), "n_participants": int(meta.participant_id.nunique()),
            "n_windows_total": int(meta.n_windows.sum()), "n_short_lt4s": int((meta.dur_s < WIN_S).sum()),
            "shapes": {k: list(v) for k, v in files.items()}, "device": args.device,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
            "versions": {"python": platform.python_version(), "torch": torch.__version__,
                         "torchaudio": torchaudio.__version__, "transformers": transformers.__version__,
                         "numpy": np.__version__},
            "cache_index_sha256": B.sha256_file(args.cache_dir / "index.csv"), "git_commit": git_commit(),
            "elapsed_s": round(time.time() - t0, 1), "date": time.strftime("%Y-%m-%d %H:%M")}
    (args.out_dir / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False, default=str))
    man = [f"{B.sha256_file(args.out_dir / f)}  {f}" for f in sorted(list(files) + ["recordings.csv", "info.json"]
                                                                     + (["windows.csv"] if args.save_windows else []))]
    (args.out_dir / "MANIFEST.sha256").write_text("\n".join(man) + "\n")
    print(f"bitti: {n} kayıt, {info['n_windows_total']} pencere, {info['n_short_lt4s']} kısa kayıt; "
          f"{time.time() - t0:.0f} s → {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
