"""
Faz 1 — Split dosyaları: katılımcı düzeyinde, tekrarlı, tabakalı 5-fold (D-003, D-009, D-024, D-029)

BİLİMSEL AMAÇ
    Hangi katılımcının hangi turda eğitimde, doğrulamada veya testte olacağını, HİÇBİR model sonucuna
    bakmadan, bir kez ve sabit olarak belirlemek. Böylece:
      1. Aynı kişinin sesi hem eğitimde hem testte olamaz (katılımcı sızıntısı yok, D-003).
      2. Bütün modeller aynı katılımcılarda test edilir → eşleştirilmiş karşılaştırma mümkün (D-011).
      3. Küçük sağlıklı grubu (59 kişi), kayıt dönemi ve yaş fold'lar arasında dengeli dağılır (D-009).
      4. Sonuç yeniden üretilebilir: aynı girdi + aynı kütüphane sürümü → aynı sha256.

VERİNİN YAPISI
    participant_context.csv  (Drive, katılımcı başına bir satır; scripts/analyze_recording_context.py üretir)
        participant_id, label (1 = astım, 0 = sağlıklı), age, recording_date (D-024: dosya zaman damgası
        öncelikli), start_hour (yalnız betimsel rapor için)
    recording_map.csv        (Drive, katılımcı × slot başına bir satır; scripts/audit_audio.py üretir)
        participant_id, slot (1–7), status (OK / MISSING / MULTIPLE)

ÇIKTILAR
    <out_dir>/outer_r{r}.csv   r = 0…4 — Drive'da kalır (katılımcı düzeyi, D-014). Sütunlar:
        participant_id, repeat, outer_fold (k = 0…4), role ("train" / "test"; k. dış fold'a göre),
        inner_fold (train için 0…4, test için -1), is_inner_val (train ve inner_fold == 0)
        → Her (r, k) bloğu tüm kohortu bir kez listeler: 342 katılımcı × 5 fold = 1 710 satır / dosya.
    <report_dir>/splits_manifest.{json,md}  yalnız agrega sayılar + sha256'lar (git'e girebilir)

KULLANIM (sonraki script'lerde)
    s = pd.read_csv("outer_r0.csv"); blok = s[s.outer_fold == k]
    test = blok[blok.role == "test"]; val = blok[blok.is_inner_val]; train = blok[(blok.role == "train") & ~blok.is_inner_val]

ÇALIŞTIRMA
    python scripts/make_splits.py --context .../data_derived/participant_context.csv \
        --recording-map .../data_derived/recording_map.csv \
        --out-dir .../data_derived/splits --report-dir reports/splits
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold

N_SPLITS = 5
N_REPEATS = 5
MIN_STRATUM = N_SPLITS  # 5'ten küçük tabaka her fold'a en az bir kişi veremez → birleştirilir
AGE_CUT = 40            # D-009: ≤40 / >40


def outer_seed(r: int) -> int:
    return r


def inner_seed(r: int, k: int) -> int:
    return 1000 + 10 * r + k


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cohort(context_csv: Path, recording_map_csv: Path) -> tuple[pd.DataFrame, str]:
    """Kohort = en az bir OK kaydı olan herkes (D-002, D-029). Dönem ve yaş grubu burada hesaplanır."""
    ctx = pd.read_csv(context_csv)
    rm = pd.read_csv(recording_map_csv)
    need_ctx = {"participant_id", "label", "age", "recording_date", "start_hour"}
    need_rm = {"participant_id", "slot", "status"}
    assert need_ctx <= set(ctx.columns), f"participant_context.csv eksik sütun: {need_ctx - set(ctx.columns)}"
    assert need_rm <= set(rm.columns), f"recording_map.csv eksik sütun: {need_rm - set(rm.columns)}"
    assert ctx["participant_id"].is_unique, "participant_context.csv'de tekrarlanan katılımcı ID'si var"
    assert ctx["label"].isin([0, 1]).all(), "label yalnız 0/1 olmalı"
    assert rm["slot"].between(1, 7).all(), "slot 1–7 dışında"
    assert not rm.duplicated(["participant_id", "slot"]).any(), "recording_map'te aynı (katılımcı, slot) iki kez"

    ok = rm[rm["status"] == "OK"]
    slots_ok = ok.pivot_table(index="participant_id", columns="slot", values="status", aggfunc="size", fill_value=0)
    slots_ok = slots_ok.reindex(columns=range(1, 8), fill_value=0).astype(bool)
    slots_ok.columns = [f"slot{s}_ok" for s in slots_ok.columns]

    unknown_ids = set(slots_ok.index) - set(ctx["participant_id"])
    assert not unknown_ids, f"recording_map'te katılımcı tablosunda olmayan {len(unknown_ids)} ID var"
    if "has_audio" in ctx.columns:  # analyze_recording_context.py çıktısıyla tutarlılık
        mism = set(ctx.loc[ctx["has_audio"].astype(bool), "participant_id"]) ^ set(slots_ok.index)
        assert not mism, f"has_audio ile recording_map OK kayıtları {len(mism)} katılımcıda uyuşmuyor"

    c = ctx[ctx["participant_id"].isin(slots_ok.index)].copy()
    c = c.merge(slots_ok, left_on="participant_id", right_index=True, how="left")
    rd = pd.to_datetime(c["recording_date"], errors="coerce")
    assert rd.notna().sum() == c["recording_date"].notna().sum(), "çözümlenemeyen kayıt tarihi var"
    last_healthy = rd[c["label"] == 0].max()
    assert pd.notna(last_healthy), "hiçbir sağlıklının kayıt tarihi yok; dönem tanımlanamaz"
    c["period"] = np.select([rd.isna(), rd <= last_healthy], ["unknown", "early"], default="late")
    c["age_group"] = np.select([c["age"].isna(), c["age"] <= AGE_CUT], ["unknown", f"<={AGE_CUT}"], default=f">{AGE_CUT}")
    return c.sort_values("participant_id").reset_index(drop=True), str(last_healthy.date())


def assign_strata(c: pd.DataFrame, min_size: int = MIN_STRATUM) -> tuple[pd.Series, list[dict]]:
    """Etiket × dönem × yaş grubu. min_size'dan küçük tabaka, aynı etiketin en kalabalık uyumlu
    tabakasına katılır (önce aynı etiket + dönem, yoksa aynı etiket)."""
    key = c["label"].astype(str) + "|" + c["period"] + "|" + c["age_group"]
    counts = key.value_counts()
    big = counts[counts >= min_size]
    log = []
    for small in sorted(counts[counts < min_size].index):
        lab, per, _ = small.split("|")
        same_period = big[[k.startswith(f"{lab}|{per}|") for k in big.index]]
        same_label = big[[k.startswith(f"{lab}|") for k in big.index]]
        assert len(same_label), f"etiket {lab} için {min_size}+ kişilik tabaka yok; split yapılamaz"
        target = (same_period if len(same_period) else same_label).sort_values(ascending=False).index[0]
        log.append({"stratum": small, "n": int(counts[small]), "merged_into": target})
        key = key.where(key != small, target)
    assert key.value_counts().min() >= min_size
    return key, log


def make_repeat(c: pd.DataFrame, strata: pd.Series, r: int) -> pd.DataFrame:
    ids = c["participant_id"].to_numpy()
    y = strata.to_numpy()
    rows = []
    outer = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=outer_seed(r))
    for k, (tr, te) in enumerate(outer.split(ids, y)):
        inner_fold = np.full(len(ids), -1)
        inner = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=inner_seed(r, k))
        for j, (_, iv) in enumerate(inner.split(ids[tr], y[tr])):
            inner_fold[tr[iv]] = j
        role = np.where(np.isin(np.arange(len(ids)), te), "test", "train")
        rows.append(pd.DataFrame({"participant_id": ids, "repeat": r, "outer_fold": k, "role": role,
                                  "inner_fold": inner_fold}))
    s = pd.concat(rows, ignore_index=True)
    s["is_inner_val"] = (s["role"] == "train") & (s["inner_fold"] == 0)
    return s


def check_repeat(s: pd.DataFrame, c: pd.DataFrame) -> None:
    """Sızıntı ve bütünlük kontrolleri. Biri bozulursa split dosyası YAZILMAZ."""
    cohort = set(c["participant_id"])
    n = len(cohort)
    assert len(s) == n * N_SPLITS, "her (r, k) bloğu tüm kohortu tam bir kez içermeli"
    tests = s[s["role"] == "test"]
    assert tests["participant_id"].is_unique and set(tests["participant_id"]) == cohort, \
        "bir tekrarda her katılımcı TAM BİR KEZ testte olmalı"
    lab = c.set_index("participant_id")["label"]
    expected_h = (lab == 0).sum() / N_SPLITS
    for k, b in s.groupby("outer_fold"):
        tr = set(b.loc[b["role"] == "train", "participant_id"])
        te = set(b.loc[b["role"] == "test", "participant_id"])
        assert not (tr & te), f"fold {k}: train ∩ test boş değil (SIZINTI)"
        assert tr | te == cohort, f"fold {k}: train ∪ test kohorta eşit değil"
        bt = b[b["role"] == "train"]
        assert (bt["inner_fold"].between(0, N_SPLITS - 1)).all(), f"fold {k}: train'de iç fold atanmamış katılımcı"
        assert (b.loc[b["role"] == "test", "inner_fold"] == -1).all(), f"fold {k}: test katılımcısına iç fold atanmış"
        assert not set(b.loc[b["is_inner_val"], "participant_id"]) & te, f"fold {k}: iç doğrulama testle kesişiyor"
        h_test = int((lab[list(te)] == 0).sum())
        assert np.floor(expected_h) - 1 <= h_test <= np.ceil(expected_h) + 1, f"fold {k}: testte {h_test} sağlıklı (dengesiz)"
        h_val = int((lab[list(b.loc[b["is_inner_val"], "participant_id"])] == 0).sum())
        h_tr = int((lab[list(tr)] == 0).sum())
        assert h_val >= 1 and np.floor(h_tr / N_SPLITS) - 1 <= h_val <= np.ceil(h_tr / N_SPLITS) + 1, \
            f"fold {k}: iç doğrulamada {h_val} sağlıklı (train'de {h_tr}; dengesiz)"
        if h_val < 5:
            print(f"UYARI fold {k}: iç doğrulamada yalnız {h_val} sağlıklı → early stopping / eşik çok gürültülü olur")


def summarize(s: pd.DataFrame, c: pd.DataFrame) -> list[dict]:
    """Fold başına agrega sayılar (katılımcı ID'si yok)."""
    ci = c.set_index("participant_id")
    out = []
    for k, b in s.groupby("outer_fold"):
        te = ci.loc[b.loc[b["role"] == "test", "participant_id"]]
        va = ci.loc[b.loc[b["is_inner_val"], "participant_id"]]
        out.append({
            "outer_fold": int(k), "n_test": int(len(te)),
            "test_asthma": int((te["label"] == 1).sum()), "test_healthy": int((te["label"] == 0).sum()),
            "test_period": {k2: int(v) for k2, v in te["period"].value_counts().items()},
            "test_age_group": {k2: int(v) for k2, v in te["age_group"].value_counts().items()},
            "test_all_7_tasks": int(te[[f"slot{i}_ok" for i in range(1, 8)]].all(axis=1).sum()),
            # Betimsel (tabakalama değişkeni değil; D-028 kayıt yükümlülüğü): seans başlangıcı öğleden önce
            "test_before_noon_descriptive": {"asthma": int(((te["start_hour"] < 12) & (te["label"] == 1)).sum()),
                                             "healthy": int(((te["start_hour"] < 12) & (te["label"] == 0)).sum()),
                                             "unknown_hour": int(te["start_hour"].isna().sum())},
            "n_train_excl_val": int(((b["role"] == "train") & ~b["is_inner_val"]).sum()),
            "inner_val_asthma": int((va["label"] == 1).sum()), "inner_val_healthy": int((va["label"] == 0).sum()),
        })
    return out


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).resolve().parent).stdout.strip()
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", required=True, type=Path)
    ap.add_argument("--recording-map", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--report-dir", required=True, type=Path)
    ap.add_argument("--n-repeats", type=int, default=N_REPEATS)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    c, last_healthy = load_cohort(args.context, args.recording_map)
    strata, merge_log = assign_strata(c)
    print(f"Kohort: {len(c)} katılımcı ({int((c.label == 1).sum())} astım / {int((c.label == 0).sum())} sağlıklı); "
          f"son sağlıklı kayıt günü {last_healthy}")
    print("Tabakalar (birleştirmeden sonra):", strata.value_counts().sort_index().to_dict())
    if merge_log:
        print("Birleştirilen seyrek tabakalar:", merge_log)

    files, folds = {}, {}
    for r in range(args.n_repeats):
        s = make_repeat(c, strata, r)
        check_repeat(s, c)
        # Belirlenimcilik: aynı tohumla ikinci üretim birebir aynı olmalı
        assert s.equals(make_repeat(c, strata, r)), f"r={r}: split belirlenimci değil"
        path = args.out_dir / f"outer_r{r}.csv"
        s.to_csv(path, index=False, lineterminator="\n")
        files[path.name] = sha256_file(path)
        folds[f"r{r}"] = summarize(s, c)
        print(f"r={r}: {path.name} sha256={files[path.name][:12]}…  test sağlıklı/fold = "
              f"{[f['test_healthy'] for f in folds[f'r{r}']]}  iç doğrulama sağlıklı = "
              f"{[f['inner_val_healthy'] for f in folds[f'r{r}']]}")

    manifest = {
        "decisions": ["D-003", "D-009", "D-024", "D-029"],
        "inputs_sha256": {"participant_context.csv": sha256_file(args.context),
                          "recording_map.csv": sha256_file(args.recording_map)},
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "git_commit": git_commit(),
        "n_splits": N_SPLITS, "n_repeats": args.n_repeats,
        "seeds": {"outer": "random_state = r", "inner": "random_state = 1000 + 10*r + k"},
        "cohort": {"n": int(len(c)), "asthma": int((c.label == 1).sum()), "healthy": int((c.label == 0).sum()),
                   "last_healthy_recording_date": last_healthy,
                   "period": {f"{a}|{b}": int(v) for (a, b), v in c.groupby(["label", "period"]).size().items()},
                   "age_group": {f"{a}|{b}": int(v) for (a, b), v in c.groupby(["label", "age_group"]).size().items()},
                   "per_task_n": {f"slot{i}": int(c[f"slot{i}_ok"].sum()) for i in range(1, 8)}},
        "strata_after_merge": {k: int(v) for k, v in strata.value_counts().sort_index().items()},
        "merged_sparse_strata": merge_log,
        "split_files_sha256": files,
        "folds": folds,
    }
    (args.report_dir / "splits_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    L = ["# Split manifesti (otomatik üretildi; katılımcı ID'si içermez)", "",
         f"- Kohort: {manifest['cohort']['n']} katılımcı ({manifest['cohort']['asthma']} astım / "
         f"{manifest['cohort']['healthy']} sağlıklı); son sağlıklı kayıt günü {manifest['cohort']['last_healthy_recording_date']}",
         f"- Görev başına katılımcı: {manifest['cohort']['per_task_n']}",
         f"- {args.n_repeats} tekrar × {N_SPLITS} dış fold; her dış train içinde {N_SPLITS} iç fold (iç fold 0 = doğrulama)",
         f"- Tabakalar (etiket|dönem|yaş grubu, birleştirmeden sonra): {manifest['strata_after_merge']}",
         f"- Birleştirilen seyrek tabakalar: {merge_log if merge_log else 'yok'}",
         f"- Sürümler: {manifest['versions']}", "",
         "## Dosya sha256'ları (Colab'da yeniden üretilen dosyalar bunlarla aynı olmalı)", ""]
    L += [f"- `{k}`: `{v}`" for k, v in files.items()]
    for rk, fl in folds.items():
        L += ["", f"## Tekrar {rk}", "",
              "| fold | test n | astım | sağlıklı | erken / geç / bilinmiyor | ≤40 / >40 / bilinmiyor | 7 görevin hepsi | öğleden önce astım / sağlıklı (betimsel) | train (val hariç) | iç val astım / sağlıklı |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for f in fl:
            per, age, bn = f["test_period"], f["test_age_group"], f["test_before_noon_descriptive"]
            L.append(f"| {f['outer_fold']} | {f['n_test']} | {f['test_asthma']} | {f['test_healthy']} | "
                     f"{per.get('early', 0)} / {per.get('late', 0)} / {per.get('unknown', 0)} | "
                     f"{age.get(f'<={AGE_CUT}', 0)} / {age.get(f'>{AGE_CUT}', 0)} / {age.get('unknown', 0)} | "
                     f"{f['test_all_7_tasks']} | {bn['asthma']} / {bn['healthy']} | {f['n_train_excl_val']} | "
                     f"{f['inner_val_asthma']} / {f['inner_val_healthy']} |")
    (args.report_dir / "splits_manifest.md").write_text("\n".join(L) + "\n")
    print(f"Manifest: {args.report_dir / 'splits_manifest.md'}")


if __name__ == "__main__":
    main()
