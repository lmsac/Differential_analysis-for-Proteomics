from ProteinQuantData import *

def BenchmarkDataCleaning(in_path):

	data = pd.read_csv(in_path)
	data.columns[0] = 'PG.ProteinAccessions'
	proteins = [s.split()[0] for s in data['PG.ProteinAccessions']]
	data['PG.ProteinAccessions'] = proteins
	data.set_index(data['PG.ProteinAccessions'], inplace=True)
	processed_data = data.sample(frac=1.0)
	processed_data.drop(columns='PG.ProteinAccessions', inplace=True)
	quant_data = ProteinQuantData(raw_data=processed_data)
	quant_data.fill_na(method='delete')
	res_data = quant_data.data.T
	res_data = np.log2(res_data)

	return res_data, proteins
