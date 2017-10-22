from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from tqdm import tqdm


class MultiLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        """
        :param columns: # array of column names or indexes to encode
        """
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        :param X: DataFrame or numpy.array
        :return: numpy array
        """
        res = []
        if self.columns is not None:
            for col in tqdm(self.columns, f'{self.__class__.__name__} transform'):
                res.append(LabelEncoder().fit_transform(X[col]))
        # else:
        #     if isinstance(X, np.ndarray):
        #         for col in X.T:
        #             res(LabelEncoder().fit_transform(np.array(col)))
        return pd.DataFrame(np.column_stack(res), columns=self.columns)


if __name__ == '__main__':
    np_array1d = np.array([('s','dfg'), ('f','s'), ('H','h')], dtype=[('col1', 'O'), ('col2', 'O')])
    print(np_array1d)
    res = MultiLabelEncoder(['col1', 'col2']).transform(np_array1d)
    print(res, flush=True)

    from transformers.morphology_extractor import MorphologyExtractor
    data = [u'в 1905 году'] + u'Определение частей речи работает не так как задумывалось в ПП'.split()
    print(data, flush=True)
    context = MorphologyExtractor().transform(data)
    print(context, flush=True)
    res = MultiLabelEncoder(('is_first_upper', 'is_upper',
                             'pos', 'animacy', 'aspect', 'case', 'gender', 'mood', 'number', 'person',
                             'tense', 'transitivity', 'voice'))\
        .transform(context)
    print(res)

