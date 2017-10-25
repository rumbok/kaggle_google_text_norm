import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from transformers.morphology_extractor import MorphologyExtractor


class PandasDummies(BaseEstimator, TransformerMixin):
    def __init__(self, category_columns=[]):
        self.category_columns = category_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.SparseDataFrame) or isinstance(X, pd.Series):
            if self.category_columns:
                res = pd.get_dummies(X, sparse=True, dummy_na=False, columns=self.category_columns).to_sparse(fill_value=0)
            else:
                res = pd.get_dummies(X, sparse=True, dummy_na=False).to_sparse(fill_value=0)
            conv = res.select_dtypes(exclude=[np.number]).astype(np.float16)
            res[conv.columns] = conv
            return res


if __name__ == '__main__':
    data = [u'в 1905 году'] + u'Определение частей речи работает не так как задумывалось'.split()
    print(data)
    morph = MorphologyExtractor().transform(data)
    print(morph.info())
    print(morph.density)

    res = PandasDummies(category_columns=['pos', 'animacy', 'aspect', 'case', 'gender', 'involvement', 'mood',
                                          'number', 'person', 'tense', 'transitivity']
                        ).fit_transform(morph)
    print(res)
    print(res.info())
    print(res.density)
