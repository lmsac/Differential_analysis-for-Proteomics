import numpy as np
import pandas as pd
import math
import scipy.stats as stats


def softmax(x):
    exps = np.exp(x)
    return np.array(exps / np.sum(exps))


def score2prob(_score: list, method='default'):
    _array = np.array(_score).flatten()
    _array[_array <= 0] = 1e-6
    _array = np.log10(_array)
    _mean = np.mean(_array)
    _sig = np.std(_array)
    _t = (_array - _mean) / (math.sqrt(2) * _sig)
    if method == 'default':
        matrix = pd.DataFrame(_t, columns=['tt'])
        _res = [np.max([0, math.erf(i)]) for i in matrix['tt']]
    elif method == 'sqrt':
        matrix = pd.DataFrame(_t, columns=['tt'])
        _res = [math.sqrt(np.max([0, math.erf(i)])) for i in matrix['tt']]
    else:
        fit_alpha, fit_loc, fit_scale = stats.gamma.fit(_array)
        _mean = stats.gamma.cdf(_mean, fit_alpha, loc=fit_loc, scale=fit_scale)
        matrix = pd.DataFrame(_array, columns=['tt'])
        _res = [np.max([0, (stats.gamma.cdf(i, fit_alpha, loc=fit_loc, scale=fit_scale) - _mean) / (1 - _mean)]) for i
                in matrix['tt']]
    return _res
