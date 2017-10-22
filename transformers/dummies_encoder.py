import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

from transformers.morphology_extractor import MorphologyExtractor
from transformers.multi_label_encoder import MultiLabelEncoder


class DummiesEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.tags = {}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            return pd.get_dummies(X)
        else:
            return OneHotEncoder().fit_transform(X)


if __name__ == '__main__':
    data = [u'в 1905 году'] + u'Определение частей речи работает не так как задумывалось'.split()
    print(data)
    res = MorphologyExtractor().transform(data)
    print(res)

    res = DummiesEncoder().fit_transform(res)
    print(res)
