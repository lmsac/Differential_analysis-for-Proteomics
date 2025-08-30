import numpy as np
import pandas as pd
import os
from math import ceil

threshold_frame = pd.read_csv('your_path')
threshold_dict = dict(zip(threshold_frame['algorithm'], threshold_frame['threshold']))
print(threshold_frame)
all_res = pd.DataFrame()
col1, col2, col3 = [], [], []

dirpath = 'your_dir_path'
respath = 'your_res_path'
result = pd.DataFrame()
recall, precision, positive_precision, model = [], [], [], []
for file in os.listdir(dirpath):
	if os.path.isdir(os.path.join(dirpath, file)):
		continue

	result_frame = pd.read_csv(os.path.join(dirpath, file), index_col=0)

	cnt = ceil(len(result_frame) * 0.1)
	if not threshold_dict.get(file):
		continue
	threshold = float(threshold_dict.get(file))
	truth = [1] * cnt + (len(result_frame) - 2 * cnt) * [0] + cnt * [1]
	y_pred = []
	for prob in result_frame['prob']:
		y_pred.append(1) if prob > threshold else y_pred.append(0)
	result_frame['y_pred'] = y_pred
	TN = 0
	TP = 0
	for j in range(len(y_pred)):
		if y_pred[j] == truth[j]:
			if y_pred[j] == 0:
				TN += 1
			else:
				TP += 1

	FN = np.array(truth).sum() - TP
	FP = len(truth) - FN - TN - TP
	recall.append(TP / (TP + FN))
	precision.append((TP + TN) / (TP + FP + TN + FN))

	if (TP + FP) == 0:
		positive_precision.append(0)
	else:
		positive_precision.append(TP / (TP + FP))
	result_frame.to_csv(os.path.join(respath, f'{file}.csv'), index=False)
	model.append(file)

result[f'recall_{x}'] = recall
result[f'precision_{x}'] = precision
result[f'positive_precision_{x}'] = positive_precision
result['model'] = model
col1.append(f'recall_{x}')
col2.append(f'precision_{x}')
col3.append(f'positive_precision_{x}')
result.to_csv(os.path.join(respath, 'benchmark_result.csv'), index=False)
if x == '10':
	all_res = result
else:
	all_res = pd.merge(all_res, result, how='outer', on='model')

all_res.index = all_res['model']
all_res.drop(columns=['model'], inplace=True)

recall_res = all_res[col1]
precision_res = all_res[col2]
positive_precision_res = all_res[col3]


