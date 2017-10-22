import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Reshape2D(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, values):
        return np.array(values).reshape(len(values), -1)
