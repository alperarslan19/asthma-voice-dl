# EXP-015 — Makalenin model seçim prosedürü için iç içe (nested) CV (keşifsel; katılımcı ID'si içermez)

Makale özellikleri, makale pipeline'ı (StandardScaler → SMOTE → model), 10 model: AdaBoost, DecisionTree, GradientBoosting, KNN, LogisticRegression, MLP, NaiveBayes, RandomForest, SVM, XGBoost. Dış: split dosyaları 5×5 (25 fold). İç: her dış train içinde 5-fold; model iç ortalama AUC'ye göre seçilir, dış test seçime karışmaz.

| görev | iç içe CV AUC (seçim dahil) | 25 fold'da sabit en iyi model (sonradan bakarak) | her fold'da en iyiyi seçmek (kâhin, üst sınır) | 10 modelin medyanı | seçim sıklığı (25 fold) |
|---|---|---|---|---|---|
| aaa | 0.693 ± 0.078 | MLP 0.699 | 0.726 | 0.610 | {'LogisticRegression': 14, 'MLP': 11} |
| araba | 0.598 ± 0.067 | XGBoost 0.639 | 0.686 | 0.601 | {'LogisticRegression': 8, 'XGBoost': 6, 'MLP': 4, 'KNN': 2, 'AdaBoost': 2, 'RandomForest': 1, 'SVM': 1, 'GradientBoosting': 1} |
| ana | 0.630 ± 0.066 | XGBoost 0.662 | 0.704 | 0.627 | {'XGBoost': 7, 'SVM': 7, 'LogisticRegression': 6, 'KNN': 1, 'GradientBoosting': 1, 'AdaBoost': 1, 'MLP': 1, 'RandomForest': 1} |
| ordu | 0.644 ± 0.082 | SVM 0.684 | 0.729 | 0.641 | {'SVM': 11, 'XGBoost': 5, 'RandomForest': 2, 'MLP': 2, 'AdaBoost': 2, 'LogisticRegression': 1, 'GradientBoosting': 1, 'KNN': 1} |
| gelecek | 0.681 ± 0.089 | MLP 0.702 | 0.759 | 0.682 | {'XGBoost': 9, 'MLP': 8, 'GradientBoosting': 5, 'AdaBoost': 2, 'RandomForest': 1} |
| titiz | 0.667 ± 0.094 | SVM 0.694 | 0.742 | 0.657 | {'SVM': 10, 'MLP': 8, 'GradientBoosting': 3, 'RandomForest': 2, 'XGBoost': 1, 'NaiveBayes': 1} |
| ünlem | 0.664 ± 0.071 | MLP 0.704 | 0.759 | 0.674 | {'XGBoost': 8, 'MLP': 7, 'AdaBoost': 7, 'LogisticRegression': 1, 'SVM': 1, 'RandomForest': 1} |

**Füzyon (her görevde iç CV'nin seçtiği model):** 0.734 ± 0.087; havuzlanmış 0.759 [0.690–0.822]

Süre: 1224.4 s
