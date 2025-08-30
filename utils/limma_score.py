import os

import pandas as pd
from math import ceil
import numpy as np
# limma_df = pd.read_csv('/home/xuguanyang/py/Differential_analysis/reference_data/final_bench/dataset4/Reallife/limma_res.csv', index_col=0)


# for num in [10, 20, 40, 60, 80, 100]:
for num in [100]:
	os.chdir(f'E:\\Differential_analysis\\reference_data\\final_bench\\dataset4_new\\{num}')

	for x in ['limma_res_0.csv']:
		limma = pd.read_csv(x, index_col=0)
		print(len(limma))
		p = limma.index.tolist()

		judge = [0 if i == 'unchanged' else 1 for i in limma['regulate']]

		d = dict(zip(p, judge))

		raw = pd.read_csv('raw_quant.csv', index_col=0)
		p2 = raw.index.tolist()
		limma_judge = [d.get(i) for i in p2]

		pos_cnt = ceil(len(raw) * 0.1)
		neg_cnt = len(raw) - 2 * pos_cnt
		ground_truth = [1] * pos_cnt + [0] * neg_cnt + [1] * pos_cnt

		TN = 0
		TP = 0
		for j in range(len(judge)):
			if limma_judge[j] == ground_truth[j]:
				if limma_judge[j] == 0:
					TN += 1
				else:
					TP += 1

		FN = np.array(ground_truth).sum() - TP
		FP = len(ground_truth) - FN - TN - TP

		print(f'True Negative = {TN}, True Positive = {TP}, False Negative = {FN}, False Positive = {FP}')
		print(f'recall: {TP / (TP + FN)}')
		print(f'positive_precision: {TP / (TP + FP)}')
		print(f'precision: {(TP + TN) / (TP + FP + TN + FN)}')

		# result = pd.read_csv('result.csv')
		#
		# result = result.reindex()
		recall = (TP / (TP + FN))
		positive_precision = (TP / (TP + FP))
		precision = ((TP + TN) / (TP + FP + TN + FN))
		limma_result = pd.DataFrame([recall, precision, positive_precision, 'limma_1.5'],)
		a = limma_result.T
		a.columns = ['recall', 'precision', 'positive_precision', 'model']
		# res = pd.concat([result, a], ignore_index=True)
		# res.reindex()
		# res.to_csv('result.csv', index=False)
		print(f'Sample Number = {num}')
		print(f'Limma_recall = {recall}')
		print(f'Limma_precision = {precision}')
		print(f'Limma_positive_precision = {positive_precision}')
		lis1 = [recall, precision, positive_precision]
		# limma_df[f'{num}'] = lis1
		# limma_df.index = ['recall', 'precision', 'positive_precision']
		# limma_df.to_csv(f'/home/xuguanyang/py/Differential_analysis/reference_data/final_bench/TMT/limma_2_df.csv')
