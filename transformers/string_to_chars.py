import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import coo_matrix
from typing import Union


class StringToChar(BaseEstimator, TransformerMixin):
    def __init__(self, max_width=0, to_coo=False):
        self.max_width = max_width
        self.to_coo = to_coo
        self.tags = {}

    def fit(self, words, y=None, **fit_params):
        return self

    def transform(self, words, y=None, **fit_params) -> Union[pd.SparseDataFrame, coo_matrix]:
        if self.max_width == 0:
            x = np.array(words, dtype=f'U')
        elif self.max_width > 0:
            x = np.array(words, dtype=f'U{self.max_width}')
        else:
            #TODO alignment to right, maybe flip
            x = np.char.lstrip(np.char.rjust(np.array(words, dtype=f'U'), width=-self.max_width))

        y = x.view('U1').view(np.uint32).reshape((x.size, -1))
        if self.to_coo:
            return coo_matrix(y, dtype=np.uint16)
        else:
            return pd.SparseDataFrame(y, default_fill_value=0)


if __name__ == '__main__':
    df = pd.SparseDataFrame(['в 1905 году', '123', 'dfhsd', '-', '&', '0546']
                            + 'съешь ещё этих мягких французских булок, да выпей чаю'.split(),
                            columns=['before'])
    print(df)

    res_df = StringToChar(-4).transform(df['before'])
    print(res_df)
    print(res_df.info())
    print(res_df.density)

    # res_coo = StringToChar(10, to_coo=True).transform(df['before'])
    # print(res_coo.shape)
    # print(res_coo.nnz/res_coo.shape[0]/res_coo.shape[1])
    #
    # res_coo = StringToChar(10, to_coo=True).transform(df['before'].shift(1).fillna('').to_dense())
    # print(res_coo.shape)
    # print(res_coo.nnz/res_coo.shape[0]/res_coo.shape[1])
