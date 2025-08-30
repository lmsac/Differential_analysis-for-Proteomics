import pandas as pd
from math import ceil
import os

import numpy as np
from pyod.models.cblof import CBLOF
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.copod import COPOD
from pyod.models.gmm import GMM
from pyod.models.hbos import HBOS
from pyod.models.kde import KDE
from pyod.models.kpca import KPCA
from pyod.models.lmdd import LMDD
from pyod.models.lscp import LSCP
from pyod.models.lunar import LUNAR
from pyod.models.mcd import MCD
from pyod.models.qmcd import QMCD
from pyod.models.sampling import Sampling
from pyod.models.suod import SUOD
from scipy.stats import norm
from sklearn.ensemble import IsolationForest
from pyod.models.knn import KNN
from pyod.utils.example import visualize
from pyod.models.ecod import ECOD
from sklearn.neighbors import KernelDensity
from utils.score2prob import score2prob
from sklearn.preprocessing import StandardScaler


def AD_Bench(bench_data, outliers_fraction, ground_truth, res_path, min_prob):
    classifiers = {
        'K Nearest Neighbors (KNN)': KNN(),
        'Average KNN': KNN(method='mean'),
        'Median KNN': KNN(method='median'),
        'Local Outlier Factor (LOF)': LOF(),
        'Isolation Forest': IForest(),
        'SUOD': SUOD(),
        'Minimum Covariance Determinant (MCD)': MCD(),
        'Principal Component Analysis (PCA)': PCA(),
        'KPCA': KPCA(n_jobs=-1),
        'Probabilistic Mixture Modeling (GMM)': GMM(),
        'Histogram-based Outlier Detection (HBOS)': HBOS(),
        'Copula-base Outlier Detection (COPOD)': COPOD(),
        'ECDF-baseD Outlier Detection (ECOD)': ECOD(),
        'Kernel Density Functions (KDE)': KDE(),
        'QMCD': QMCD(),
        'Sampling': Sampling(),
        'LUNAR': LUNAR(),
        'Cluster-based Local Outlier Factor (CBLOF)': CBLOF(),
        'One-class SVM (OCSVM)': OCSVM(),
    }
    scaler = StandardScaler()
    bench_data = scaler.fit_transform(bench_data)
    result = pd.DataFrame()
    recall, precision, positive_precision, model = [], [], [], []
    for i, (clf_name, clf) in enumerate(classifiers.items()):
        print()
        print(i + 1, 'fitting', clf_name)

        clf.fit(bench_data)
        clf.predict(bench_data)
        y_score = pd.DataFrame(clf.decision_scores_, columns=['a'])

        y_prob = score2prob(y_score['a'])
        csv_res = pd.DataFrame()
        csv_res['score'] = y_score
        csv_res['prob'] = y_prob
        csv_res.to_csv(os.path.join(res_path, clf_name))

        judge = []
        for prob in csv_res['prob']:
            if prob > min_prob:
                judge.append(1)
            else:
                judge.append(0)

        TN = 0
        TP = 0
        for j in range(len(judge)):
            if judge[j] == ground_truth[j]:
                if judge[j] == 0:
                    TN += 1
                else:
                    TP += 1

        FN = np.array(ground_truth).sum() - TP
        FP = len(ground_truth) - FN - TN - TP
        recall.append(TP / (TP + FN))
        precision.append((TP + TN) / (TP + FP + TN + FN))
        positive_precision.append(TP / (TP + FP))
        model.append(clf_name)

        print(f'True Negative = {TN}, True Positive = {TP}, False Negative = {FN}, False Positive = {FP}')
        print(f'recall: {TP / (TP + FN)}')
        print(f'positive_precision: {TP / (TP + FP)}')
        print(f'precision: {(TP + TN) / (TP + FP + TN + FN)}')

    result['recall'] = recall
    result['precision'] = precision
    result['positive_precision'] = positive_precision
    result['model'] = model

    return result


def set_zeros(row, percent=0.3):
    cols = len(row)
    if percent == 0:
        return row
    if percent > 1 or percent < 0:
        raise ValueError('percent need to be between 0 and 1')

    num_zeros = int(np.ceil(cols * percent))
    zero_indices = np.random.choice(range(cols), size=num_zeros, replace=False)
    row.iloc[zero_indices] = 0
    return row


def normal_distribution_generator(raw: pd.DataFrame, num: int):
    if num == 0:
        return raw


def origin_data_generator(raw: pd.DataFrame, num: int):
    if num == 0:
        return raw
    res = raw.copy()
    for i in range(num):
        col = f'random{i}'
        res[col] = raw.apply(lambda row: np.random.choice(row), axis=1)
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            res.iat[i, j] *= np.random.uniform(0.95, 1.05)

    return res


def test_data_generator(origin: pd.DataFrame, zeros: float, max_fc):
    pos_cnt = ceil(len(origin) * 0.1)
    neg_cnt = len(origin) - 2 * pos_cnt
    judge = [-1] * pos_cnt + [0] * neg_cnt + [1] * pos_cnt
    ground_truth = [1] * pos_cnt + [0] * neg_cnt + [1] * pos_cnt
    out_frac = pos_cnt * 2 / len(origin)
    fc = pd.DataFrame()
    for i in range(len(origin.columns)):
        fc[f'Column_{i}'] = judge

    fc = fc.apply(set_zeros, axis=1)

    for i in range(fc.shape[0]):
        for j in range(fc.shape[1]):
            judge_value = fc.iat[i, j]
            if judge_value == 0:
                fc.iat[i, j] = np.random.uniform(-np.log2(1.1), np.log2(1.1))
            else:
                fc.iat[i, j] = np.random.uniform(np.log2(1.75), max_fc) * judge_value

    fc.columns = origin.columns
    fc.index = origin.index

    fc_data = fc + origin

    return ground_truth, out_frac, fc_data


def kde_data_generator(df: pd.DataFrame):
    array = np.array(df).reshape(1, -1)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(array)
    new = kde.sample(1)

    return new


def per_protein_normal_generator(df: pd.DataFrame, num: int):
    new_df = pd.DataFrame()

    for index, row in df.iterrows():
        array = np.array(row).reshape(1, -1)
        loc, scale = norm.fit(array)
        new = pd.DataFrame(np.random.normal(loc=loc, scale=scale, size=num)).T
        new_df = new if len(new_df) == 0 else new_df._append(new)

    new_df.index = df.index
    return new_df
