import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PandasUnion(BaseEstimator, TransformerMixin):
    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y=None):
        for step in self.steps:
            step[1].fit(X, y)
        return self

    def transform(self, X):
        return pd.concat([s[1].transform(X).add_prefix(s[0]+'_') for s in self.steps], axis=1)
