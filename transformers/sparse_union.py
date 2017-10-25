from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import hstack


class SparseUnion(BaseEstimator, TransformerMixin):
    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return hstack([s[1].transform(X) for s in self.steps])
