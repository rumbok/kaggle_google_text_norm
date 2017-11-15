import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from tqdm import tqdm
from num2words import num2words
import re


CARDINAL_REGEXP = '^-?\d+$'
regexp = re.compile(CARDINAL_REGEXP)


class CardinalTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, use_case=False):
        self.use_case = use_case

    def fit(self, X, y=None, *args, **kwargs):
        return self

    def transform(self, X: pd.DataFrame, y=None, *args, **kwargs):
        data = []
        for (before, cls) in tqdm(zip(X['before'], X['class']), f'{self.__class__.__name__} transform', total=len(X)):
            if cls == 'CARDINAL' and regexp.match(before):
                data.append(num2words(before, lang='ru'))
            else:
                data.append(None)

        if 'after' in X.columns:
            return X.assign(after=X['after'].combine_first(pd.Series(data, index=X.index)))
        else:
            return X.assign(after=data)


if __name__ == '__main__':
    df = pd.DataFrame([[u'0', 'CARDINAL'], [u'56', 'CARDINAL'], [u'-05665', 'CARDINAL'], [u'0.5', 'CARDINAL']], columns=['before', 'class'])
    dt = CardinalTransformer()

    print(dt.fit_transform(df).head())

