from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import hstack


class SparseUnion(BaseEstimator, TransformerMixin):
    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y=None, **fit_params):
        if y is not None:
            for s in self.steps:
                s[1].fit(X, y, **fit_params)
        else:
            for s in self.steps:
                s[1].fit(X, **fit_params)

        return self

    def transform(self, X, y=None, **fit_params):
        if y is not None:
            return hstack([s[1].transform(X, y, **fit_params) for s in self.steps])
        else:
            return hstack([s[1].transform(X, **fit_params) for s in self.steps])

    # def fit_transform(self, X, y=None, **fit_params):
    #     if y is not None:
    #         return hstack([s[1].transform(X, y, **fit_params) for s in self.steps])
    #     else:
    #         return hstack([s[1].fit_transform(X, **fit_params) for s in self.steps])

