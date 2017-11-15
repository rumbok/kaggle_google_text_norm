import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from tqdm import tqdm
from num2words import num2words
import re

ROME_REGEXP = '^[XVI]+$'
regexp = re.compile(ROME_REGEXP)


numeral_map = tuple(zip(
    (1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1),
    ('M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I')
))


def roman_to_int(n):
    i = result = 0
    for integer, numeral in numeral_map:
        while n[i:i + len(numeral)] == numeral:
            result += integer
            i += len(numeral)
    return result


class RomeTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, use_case=False):
        self.use_case = use_case

    def fit(self, X, y=None, *args, **kwargs):
        return self

    def transform(self, X: pd.DataFrame, y=None, *args, **kwargs):
        data = []
        for before in tqdm(X['before'], f'{self.__class__.__name__} transform', total=len(X)):
            if len(before) > 1 and regexp.match(before):
                data.append(num2words(roman_to_int(before), lang='ru'))
            else:
                data.append(None)
        if 'after' in X.columns:
            return X.assign(after=X['after'].combine_first(pd.Series(data, index=X.index)))
        else:
            return X.assign(after=data)


if __name__ == '__main__':
    df = pd.DataFrame([[u'XVI', u'16'], [u'XXV', u'25'], [u'IV', u'4'], [u'VI', u'6']], columns=['before', 'after'])
    dt = RomeTransformer()

    print(dt.fit_transform(df.drop(['after'], axis=1), df['after']).head())
