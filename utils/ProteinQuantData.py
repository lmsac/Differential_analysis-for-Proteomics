import math
import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN


class ProteinQuantData:

    def __init__(self, raw_data: pd.DataFrame = None, sample_label: pd.DataFrame = None) -> None:

        self.raw_data = raw_data
        self.sample = self.raw_data.columns
        self.protein = self.raw_data.index.tolist()
        self.standard_judge = False
        self.data = raw_data.T

        self.filter_data = pd.DataFrame()
        self.sample_label = sample_label
        self.size_factor = None
        self.properties = pd.DataFrame()

    def fill_na(self, method: str = None) -> pd.DataFrame:

        ref_df = self.data[self.data.notnull()]
        if method == 'mean':
            print(ref_df.mean())
            self.data.fillna(ref_df.mean(), inplace=True)
        if method == 'min':
            self.data.fillna((ref_df.min() / 5), inplace=True)
            print(ref_df.min() / 5)
        if method == 'median':
            self.data.fillna(ref_df.median(), inplace=True)

        if method == 'delete':
            self.data.dropna(axis=1, thresh=math.ceil(len(self.data) / 2), inplace=True)
            self.data.fillna(self.data.min(), inplace=True)

        return self.data.T

    def standardize(self, method: str = 'default') -> pd.DataFrame:
        if method == 'standard':
            self.data = zscore(self.data)
        if method == 'DEseq2':

            stdata = np.log(self.data)

            for protein in stdata.columns:
                stdata[protein] -= stdata[protein].mean()

            self.size_factor = stdata.median(axis=1)
            self.size_factor = np.exp(self.size_factor)

            self.data = np.divide(self.data.T, self.size_factor).T

        if method == 'decentralize':
            mean = self.data.mean()
            self.data -= mean

        if method == 'default':
            scaler = MinMaxScaler()
            self.data = pd.DataFrame(scaler.fit_transform(self.data), columns=self.data.columns)

        self.standard_judge = True

        return self.data.T

    def decomposition(self, n_components=None, method='PCA') -> pd.DataFrame:

        res = pd.DataFrame()

        if method == 'PCA':
            nparray = self.data.to_numpy()

            pca = PCA(n_components=n_components)

            result = pca.fit_transform(nparray)
            max_res, min_res = result.max(0), result.min(0)

            result_norm = (result - min_res) / (max_res - min_res)

            res = pd.DataFrame(result_norm, index=self.sample)
            var_ex = pca.explained_variance_ratio_

            res.columns = ['axis_0', 'axis_1']
            plt.scatter(x=result_norm[:, 0], y=result_norm[:, 1])
            plt.show()

        if method == 'T-SNE':
            nparray = self.data.to_numpy()
            tsne = TSNE(n_components=n_components, init='pca', perplexity=5)
            result = tsne.fit_transform(nparray)
            max_res, min_res = result.max(0), result.min(0)
            result_norm = (result - min_res) / (max_res - min_res)
            db = DBSCAN(eps=0.1, min_samples=3)
            labels = db.fit_predict(result_norm)
            labels_df = pd.DataFrame(labels, columns=['label'])
            plt.subplot(121)
            plt.scatter(x=result_norm[:, 0], y=result_norm[:, 1], label=self.sample_label['cell_type'],
                        c=self.sample_label['numeric_label'], cmap='viridis'
                        )
            plt.subplot(122)
            plt.scatter(x=result_norm[:, 0], y=result_norm[:, 1], label=labels_df['label'],
                        c=labels_df['label'], cmap='viridis'
                        )

            plt.show()
        return res

    def cal_properties(self) -> pd.DataFrame:

        if not self.standard_judge:
            self.standardize()
        self.properties['variance'] = np.var(self.data)
        self.properties['standard division'] = np.std(self.data)
        self.properties['mean'] = self.data.mean()

        return self.properties
