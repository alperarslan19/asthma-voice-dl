"""(B) Makalenin kendi özellikleriyle (04072025/master_acoustic_features_v2.csv; kimlikli) bağlam kontrolü.
Modeller makalenin ayarlarıyla (LR C=0.1 dengeli, SVM rbf, RF max_depth 10 dengeli; StandardScaler -> SMOTE).
CV: (i) katılımcı düzeyi tabakalı 5-fold (karıştırmalı), (ii) kayıt GÜNÜNE göre gruplu 5-fold (bizim grouped_folds);
her biri 5 tekrar. Katılımcı skoru = 7 sesin OOF olasılık ortalaması (tekrar ortalaması).
Referans: yalnız tarih (cr(gün, 4)), yalnız yaş, tarih + yaş — aynı fold'larda. T1: y ~ cr(gün,4) + yaş + z(logit skor),
Firth PLR + gün-kümeli jackknife (bizim evaluate_context.t1). Saat bu dosyada yok (yalnız tarih)."""
import sys, glob, warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/ (evaluate_context, make_splits)
import evaluate_context as E
import make_splits as MS
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
U, OUT = sys.argv[1], sys.argv[2]
f = lambda k: [p for p in glob.glob(U + '/*.csv') if p.endswith(k)][0]
C = pd.read_csv(f('clinical_data.csv'), sep=None, engine='python')
C['d'] = pd.to_datetime(C['Veri  toplama tarihi'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
M = pd.read_csv(f('master_acoustic_features_v2.csv'))
feat = [c for c in M.columns if c.startswith('feature_')]
assert len(feat) == 44
P = M.drop_duplicates('SubjectNumber').set_index('SubjectNumber')[['DiseaseCategory', 'Age']]
P = P.join(C.set_index('Hasta')[['d']]).dropna()
P['days'] = (P.d - P.d.min()).dt.days.astype(float); P['age'] = P.Age.astype(float); P['label'] = P.DiseaseCategory.astype(int)
day = P.d.dt.strftime('%Y-%m-%d')
print('kohort', len(P), P.label.value_counts().to_dict(), 'gün', day.nunique(), flush=True)
pids = P.index.to_numpy(); y = P.label.to_numpy()
def model(n):
    return {'LR': LogisticRegression(max_iter=2000, C=0.1, penalty='l2', solver='saga', class_weight='balanced', random_state=42),
            'SVM': SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=42),
            'RF': RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)}[n]
def folds(scheme, seed):
    if scheme == 'katilimci':
        return [(tr, te) for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(pids, y)]
    a = MS.grouped_folds(y, day.to_numpy(), 5, seed)          # örnek başına fold no (pids sırasıyla hizalı)
    assert a.shape == (len(pids),) and set(np.unique(a)) == set(range(5))
    return [(np.flatnonzero(a != k), np.flatnonzero(a == k)) for k in range(5)]
rows, scores = [], {}
Xc_full = E.context_matrix(P.assign(start_hour=0.0), 4, hour=False, date=True, age=True)
refs = {'tarih': dict(hour=False, date=True, age=False), 'yas': dict(hour=False, date=False, age=True), 'tarih+yas': dict(hour=False, date=True, age=True)}
for scheme in ['katilimci', 'gun']:
    for rep in range(5):
        FO = folds(scheme, 100 + rep)
        if scheme == 'gun':
            for tr, te in FO: assert not set(day.iloc[tr]) & set(day.iloc[te]), 'gün sızıntısı'
        for k, (tr, te) in enumerate(FO):
            for rn, kw in refs.items():
                X = E.context_matrix(P.assign(start_hour=0.0), 4, **kw)[:, 1:]
                m = LogisticRegression(C=1e4, max_iter=5000).fit(X[tr], y[tr])
                scores.setdefault((scheme, 'ref_' + rn, rep), np.full(len(y), np.nan))[te] = m.predict_proba(X[te])[:, 1]
            for mn in ['LR', 'SVM', 'RF']:
                acc = np.zeros(len(te)); cnt = np.zeros(len(te))
                for snd in range(1, 8):
                    S = M[M.SoundFileNumber == snd].set_index('SubjectNumber').reindex(pids)
                    ok = S[feat].notna().all(1).to_numpy()
                    trk, tek = tr[ok[tr]], te[ok[te]]
                    pipe = ImbPipeline([('s', StandardScaler()), ('sm', SMOTE(random_state=42)), ('m', model(mn))])
                    pipe.fit(S[feat].to_numpy()[trk], y[trk])
                    pr = pipe.predict_proba(S[feat].to_numpy()[tek])[:, 1]
                    pos = np.searchsorted(te, tek); acc[pos] += pr; cnt[pos] += 1
                    scores.setdefault((scheme, f'{mn}_snd{snd}', rep), np.full(len(y), np.nan))[tek] = pr
                scores.setdefault((scheme, mn + '_fuzyon', rep), np.full(len(y), np.nan))[te] = acc / cnt
        print(scheme, rep, 'bitti', flush=True)
out = []
day_codes = day.astype('category').cat.codes.to_numpy()
for (scheme, arm, rep), s in scores.items():
    out.append(dict(scheme=scheme, arm=arm, rep=rep, auc=roc_auc_score(y[~np.isnan(s)], s[~np.isnan(s)])))
A = pd.DataFrame(out).groupby(['scheme', 'arm']).auc.agg(['mean', 'std']).round(3).reset_index()
print(A.to_string())
t1 = []
for scheme in ['katilimci', 'gun']:
    for arm in ['LR_fuzyon', 'SVM_fuzyon', 'RF_fuzyon']:
        s = np.mean([scores[(scheme, arm, r)] for r in range(5)], 0)
        s = E.logit(s)
        t = E.t1(y.astype(float), Xc_full, s, groups=day_codes, ml=False, jackknife=True)
        b = E.boot_or(y.astype(float), Xc_full, s, 1000, 0)
        t1.append(dict(scheme=scheme, arm=arm, auc_shat=round(roc_auc_score(y, s), 3), p_plr=t['p_lrt'], or_sd=round(t['or_sd'], 2),
                       or_lo=round(b['lo'], 2), or_hi=round(b['hi'], 2), p_cv3=t['p_cv3']))
T = pd.DataFrame(t1); print(T.to_string())
A.to_csv(OUT + '_auc.csv', index=False); T.to_csv(OUT + '_t1.csv', index=False)
