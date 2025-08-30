import numpy as np
import pandas as pd

from benchmark_functions import *
import time
raw = pd.read_csv('your_tmt_data_path')

raw.drop(columns='PG.PSMsForQuant', inplace=True)
TMT_groups = raw.groupby('S.Name')
cols = raw.columns
CA = ['PG.TMT11_126', 'PG.TMT11_127C', 'PG.TMT11_128C', 'PG.TMT11_129C', 'PG.TMT11_130C']
NAT = ['PG.TMT11_127N', 'PG.TMT11_128N', 'PG.TMT11_129N', 'PG.TMT11_130N', 'PG.TMT11_131N']
QC = 'PG.TMT11_131C'
blocks = raw['S.Name'].drop_duplicates().to_list()
res1 = pd.DataFrame()
res_number = []
set1 = []
QC_frame = pd.DataFrame()
CA_frame = pd.DataFrame()
NAT_frame = pd.DataFrame()
for block in blocks:

	CA_colnames = [f'{block}_A{i}' for i in range(1, 6)]
	NAT_colnames = [f'{block}_B{i}' for i in range(1, 6)]
	QC_col = f'{block}_QC'
	TMT_blockdata = TMT_groups.get_group(block).reset_index()
	TMT_blockdata.drop(columns=['index', 'S.Name'])

	if len(set1) == 0:
		set1 = set(TMT_blockdata['PG.ProteinAccessions'].to_list())
	else:
		set1 = list(set(set1) & set(TMT_blockdata['PG.ProteinAccessions'].to_list()))

	QC_df, CA_df, NAT_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

	for i in range(0, 5):
		ca1 = CA_colnames[i]
		ca2 = CA[i]
		nat1 = NAT_colnames[i]
		nat2 = NAT[i]
		CA_df[CA_colnames[i]] = TMT_blockdata[CA[i]]
		NAT_df[NAT_colnames[i]] = TMT_blockdata[NAT[i]]
	NAT_df[QC_col] = TMT_blockdata[QC].to_frame()
	CA_df[QC_col] = TMT_blockdata[QC].to_frame()
	QC_df[QC_col] = TMT_blockdata[QC].to_frame()

	CA_df = CA_df.div(CA_df[QC_col], axis=0)
	NAT_df = NAT_df.div(NAT_df[QC_col], axis=0)

	res_number.append(len(TMT_blockdata))

	CA_df.drop(columns=QC_df, inplace=True)
	NAT_df.drop(columns=QC_df, inplace=True)

	QC_df['PG.ProteinAccessions'] = TMT_blockdata['PG.ProteinAccessions']
	CA_df['PG.ProteinAccessions'] = TMT_blockdata['PG.ProteinAccessions']
	NAT_df['PG.ProteinAccessions'] = TMT_blockdata['PG.ProteinAccessions']
	if block == 'Block #1':
		QC_frame = QC_df
		CA_frame = CA_df
		NAT_frame = NAT_df
	else:
		QC_frame = pd.merge(QC_frame, QC_df, on='PG.ProteinAccessions', how='outer')
		CA_frame = pd.merge(CA_frame, CA_df, on='PG.ProteinAccessions', how='outer')
		NAT_frame = pd.merge(NAT_frame, NAT_df, on='PG.ProteinAccessions', how='outer')

data = pd.merge(CA_frame, NAT_frame, on='PG.ProteinAccessions', how='outer')

QC_num = QC_frame.drop(columns='PG.ProteinAccessions')
mean = np.nanmean(np.array(QC_num).flatten())

proteins = [s.split()[0] for s in data['PG.ProteinAccessions']]
data['PG.ProteinAccessions'] = proteins
data.set_index(data['PG.ProteinAccessions'], inplace=True)

data.drop(columns='PG.ProteinAccessions', inplace=True)

data = data * mean
quant_data = ProteinQuantData(raw_data=data)
quant_data.fill_na(method='delete')
data2 = quant_data.data.T
data2 = np.log2(data2)
data2.to_csv('your_tmt_data_output_path')

