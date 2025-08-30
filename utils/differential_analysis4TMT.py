import numpy as np
import pandas as pd

data = pd.read_csv('data/TMT/final_result.csv', index_col=0)

sum1 = []
score_MCD, score_COPOD, score_KNN, score_Limma = [], [], [], []
for index, row in data.iterrows():
	sum_row = np.sum(row.to_list())
	if sum_row == 1:
		sum_row = 0
	sum1.append(sum_row)
	score_MCD.append(row['MCD'] * sum_row)
	score_COPOD.append(row['COPOD'] * sum_row)
	score_KNN.append(row['KNN'] * sum_row)
	score_Limma.append(row['pred'] * sum_row)

print(f'score_MCO = {np.sum(score_MCD)}')
print(f'score_COPOD = {np.sum(score_COPOD)}')
print(f'score_KNN = {np.sum(score_KNN)}')
print(f'score_Limma = {np.sum(score_Limma)}')

COPOD_dict = dict(zip(data.index.tolist(), data['COPOD'].tolist()))

limma = pd.read_csv('data/TMT/limma_res_real_TMT.csv', index_col=0)

copod = [COPOD_dict.get(protein) for protein in limma.index.tolist()]

limma['pred'] = copod


prob1 = pd.read_csv('data/TMT/real_Copula-base Outlier Detection (COPOD).csv')

prob_dict = dict(zip(data.index.tolist(), prob1['prob'].tolist()))

copod_prob = [prob_dict.get(protein) for protein in limma.index.tolist()]

score_dict = dict(zip(data.index.tolist(), prob1['score'].tolist()))

copod_score = [score_dict.get(protein) for protein in limma.index.tolist()]
limma['score'] = copod_score

raw = pd.read_csv('data/TMT/raw.csv')



