import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PandasShift(BaseEstimator, TransformerMixin):
    def __init__(self, period):
        self.period = period

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.shift(self.period).fillna('').to_dense()


if __name__ == '__main__':
    data = pd.SparseDataFrame(['1', '2', '3'])
    print(data.info())
    res = PandasShift(1).transform(data)
    print(res.info())
    print(res)
