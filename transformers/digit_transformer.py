import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from tqdm import tqdm
from num2words import num2words
import re


DIGIT_REGEXP = '^0\d+$'
regexp = re.compile(DIGIT_REGEXP)


class DigitTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None, *args, **kwargs):
        return self

    def transform(self, X: pd.DataFrame, y=None, *args, **kwargs):
        data = []
        for before in tqdm(X['before'], f'{self.__class__.__name__} transform', total=len(X)):
            if len(before) <= 4 and regexp.match(before):
                data.append(' '.join([num2words(n, lang='ru') for n in before]))
            else:
                data.append(None)
        if 'after' in X.columns:
            return X.assign(after=X['after'].combine_first(pd.Series(data, index=X.index)))
        else:
            return X.assign(after=data)


if __name__ == '__main__':
    df = pd.DataFrame([[u'056', u'dfdfgh'], [u'56', u'dfghdfg'], [u'05665', u'dfdfgh']], columns=['before', 'after'])
    dt = DigitTransformer()

    print(dt.fit_transform(df.drop(['after'], axis=1), df['after']).head())

