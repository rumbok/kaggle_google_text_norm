import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import coo_matrix
from typing import Union


class StringToChar(BaseEstimator, TransformerMixin):
    def __init__(self, max_width='', to_coo=False):
        self.max_width = max_width
        self.to_coo = to_coo
        self.tags = {}

    def fit(self, words, y=None):
        return self

    def transform(self, words) -> Union[pd.SparseDataFrame, coo_matrix]:
        x = np.array(words, dtype=f'U{self.max_width}')
        y = x.view('U1').view(np.uint32).reshape((x.size, -1))
        if self.to_coo:
            return coo_matrix(y, dtype=np.uint16)
        else:
            return pd.SparseDataFrame(y, default_fill_value=0)


if __name__ == '__main__':
    data = [u'в 1905 году'] + u'Определение частей речи работает не так как задумывалось'.split()
    print(data)

    res_df = StringToChar().transform(data)
    print(res_df)
    print(res_df.info())
    print(res_df.density)

    res_coo = StringToChar(to_coo=True).transform(data)
    print(res_coo)
    print(res_coo.nnz/res_coo.shape[0]/res_coo.shape[1])
