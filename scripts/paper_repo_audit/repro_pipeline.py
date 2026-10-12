"""(A) Makalenin astım-sağlıklı hattının birebir yeniden üretimi: 18072025/model_training.py ile aynı modeller,
aynı pipeline (StandardScaler -> SMOTE -> model), cross_validate(cv=5) [karıştırmasız StratifiedKFold, dosya sırası],
en iyi 5 modelden Voting / Stacking / Weighted Voting. Girdi: 18072025/<ses>_features.csv (kimliksiz)."""
import sys, glob, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import make_scorer, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier, StackingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
U = sys.argv[1]; OUT = sys.argv[2]; PERM = int(sys.argv[3]) if len(sys.argv) > 3 else 0
def models():
    return {'Naive_Bayes': GaussianNB(), 'KNN': KNeighborsClassifier(n_neighbors=5),
     'Decision_Tree': DecisionTreeClassifier(max_depth=5, min_samples_split=5, min_samples_leaf=2, random_state=42),
     'Neural_Network_Paper': MLPClassifier(hidden_layer_sizes=(256,128,64,32), activation='relu', solver='sgd', learning_rate='adaptive', max_iter=500, random_state=42),
     'Neural_Network_Adam': MLPClassifier(hidden_layer_sizes=(256,128,64,32), activation='relu', solver='adam', max_iter=500, random_state=42),
     'CatBoost': CatBoostClassifier(iterations=100, max_depth=5, learning_rate=0.1, verbose=0, random_state=42, thread_count=1, allow_writing_files=False),
     'AdaBoost': AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3), n_estimators=200, learning_rate=0.1, random_state=42),
     'GradientBoost': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=5, random_state=42),
     'XGBoost': XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, eval_metric='logloss', random_state=42, n_jobs=1),
     'Random_Forest': RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42),
     'SVM': SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=42),
     'Logistic_Regression': LogisticRegression(max_iter=2000, C=0.1, penalty='l2', solver='saga', class_weight='balanced', random_state=42)}
scoring = {'accuracy': 'accuracy', 'recall': 'recall', 'roc_auc': 'roc_auc', 'specificity': make_scorer(recall_score, pos_label=0)}
def ev(m, X, y):
    p = ImbPipeline([('scaler', StandardScaler()), ('smote', SMOTE(random_state=42)), ('classifier', m)])
    r = cross_validate(p, X, y, cv=5, scoring=scoring, n_jobs=5)
    return {k: (float(np.mean(r[f'test_{k}'])), float(np.std(r[f'test_{k}']))) for k in scoring}
rng = np.random.default_rng(PERM)
res = []
for snd in ['aaa', 'ana', 'araba', 'gelecek', 'ordu', 'titiz', 'unlem']:
    D = pd.read_csv([p for p in glob.glob(U + '/*.csv') if p.endswith(f'-{snd}_features.csv')][0])
    X, y = D.drop(columns='label'), D['label'].copy()
    if PERM: y = pd.Series(rng.permutation(y.to_numpy()))
    base = models(); rr = []
    for n, m in base.items():
        r = ev(m, X, y); rr.append((n, r)); res.append(dict(sound=snd, model=n, **{k: v[0] for k, v in r.items()}, auc_sd=r['roc_auc'][1]))
    top = sorted(rr, key=lambda t: -t[1]['roc_auc'][0])[:5]
    est = [(n, base[n]) for n, _ in top]; w = [t[1]['roc_auc'][0] for t in top]
    for n, m in [('Voting_Ensemble', VotingClassifier(est, voting='soft')),
                 ('Stacking_Ensemble', StackingClassifier(est, final_estimator=LogisticRegression(max_iter=2000, C=0.5), cv=5)),
                 ('Weighted_Voting_Ens', VotingClassifier(est, voting='soft', weights=w))]:
        r = ev(m, X, y); res.append(dict(sound=snd, model=n, **{k: v[0] for k, v in r.items()}, auc_sd=r['roc_auc'][1]))
    print(snd, 'bitti', flush=True)
pd.DataFrame(res).to_csv(OUT, index=False)
