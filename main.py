import argparse

from utils.logging_format import Logger
from utils.ProteinQuantData import ProteinQuantData
from utils.benchmark_functions import *

logger = Logger(name='Benchmarking Differential Analysis For Proteomics')
parser = argparse.ArgumentParser(
    description='Benchmarking '
)
parser.add_argument(
    '--raw_data', '-rd',
    type=str,
    help='Input Path Of Protein Expression Data Matrix '
)
parser.add_argument(
    '--process', '-p',
    type=bool,
    default=1,
    help='Whether the data need to be processed (e.g. fill missing value log transformation)'
)
parser.add_argument(
    '--result_dir', '-rsd',
    type=str,
    help='result dir'
)
parser.add_argument(
    '--outlier_fractions', '-of',
    default=0.1,
    help='Fractions of outliers in the generated data'
)

parser.add_argument(
    '--sample_number', '-sn',
    default=100,
    help='Number of in silico samples'
)
args = parser.parse_args()
in_path = args.raw_data
result_dir = args.result_dir
data_type = args.data_type
process = args.process
outlier_fraction = args.outlier_fraction
num = args.sample_number
data = pd.read_csv(in_path, index_col=0)

if process:
    quant_data = ProteinQuantData(raw_data=data)
    quant_data.fill_na(method='delete')
    res_data = quant_data.data.T
    data = np.log2(res_data)

sdir = os.path.join(result_dir, f'{num}')
os.chdir(result_dir)
os.mkdir(sdir)
control = per_protein_normal_generator(data, num)
test = per_protein_normal_generator(data, num)

Sample, Label = [], []
for i in range(1, num + 1):
    Sample.append(f'A{i}')
    Label.append(f'A')
for i in range(1, num + 1):
    Sample.append(f'B{i}')
    Label.append(f'B')
class_1 = pd.DataFrame()
class_1['Sample'] = Sample
class_1['Label'] = Label
class_1.to_csv(os.path.join(sdir, 'class.csv'), index=False)

control.to_csv(f'control_{num}.csv')
test.to_csv(f'test_{num}.csv')
judge1, out_f, test2 = test_data_generator(test, 0.3, np.log2(4))
control_test = pd.concat([control, test2], axis=1)
control_test.to_csv(os.path.join(sdir, 'raw_quant.csv'))

res = AD_Bench(control_test, out_f, judge1, sdir, 0.1)
res.to_csv(os.path.join(sdir, 'result.csv'))
