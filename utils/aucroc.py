import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from math import ceil
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, precision_recall_curve


def aucroc(input_dir: str, result_path: str):
    os.chdir(input_dir)
    plt.figure()
    lw = 2
    total, cnt = 0, 0
    filename, threshold12 = [], []
    threshold_frame = pd.DataFrame()
    for file in os.listdir():
        if os.path.isfile(file):

            data = pd.read_csv(file, index_col=0)

            cnt = ceil(len(data) * 0.1)

            truth = [1] * cnt + (len(data) - 2 * cnt) * [0] + cnt * [1]

            y_test = truth

            y_score = data['prob']

            fpr, tpr, threshold = roc_curve(y_test, y_score)

            precision, recall, thresholds = precision_recall_curve(y_test, y_score)

            roc_auc = auc(fpr, tpr)

            if roc_auc < 0.5:
                continue

            sensitivity = 1 - fpr

            jindex = tpr - fpr

            # maximum = (tpr - fpr).tolist().index(max(tpr - fpr))
            maximum = jindex.tolist().index(max(jindex))
            filename.append(file)
            threshold12.append(threshold[maximum])
            print(f'{file}: max Jindex = {max(jindex)}')
            print(f'{file}: sensitivity = {sensitivity[maximum]}')
            print(f'{file}: tpr = {tpr[maximum]}')
            print(f'{file}: threshold = {threshold[maximum]}')

            total += threshold[maximum]

    threshold_frame['algorithm'] = filename
    threshold_frame['threshold'] = threshold12

    threshold_frame.to_csv(result_path, index=False)
