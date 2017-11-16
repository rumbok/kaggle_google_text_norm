import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from tqdm import tqdm
import re

PREV_DASH_REGEXP = '\d+$'
NEXT_DASH_REGEXP = '^\d'
re_prev = re.compile(PREV_DASH_REGEXP)
re_next = re.compile(NEXT_DASH_REGEXP)


class DashTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None, *args, **kwargs):
        return self

    def transform(self, X: pd.DataFrame, y=None, *args, **kwargs):
        data = []
        for (w_prev, w, w_next) in tqdm(zip(X['prev'], X['before'], X['next']),
                                        f'{self.__class__.__name__} transform',
                                        total=len(X)):
            if w.strip() == u'-' and re_prev.match(w_prev) and re_next.match(w_next):
                data.append(u'до')
            elif w.strip() == u'\u2014' and re_prev.match(w_prev) and re_next.match(w_next):
                data.append(u'до')
            else:
                data.append(None)
        if 'after' in X.columns:
            return X.assign(after=X['after'].combine_first(pd.Series(data, index=X.index)))
        else:
            return X.assign(after=data)


if __name__ == '__main__':
    df = pd.DataFrame([[u'056', u'dfdfgh'], [u'56', u'dfghdfg'], [u'05665', u'dfdfgh']], columns=['before', 'after'])
    dt = DashTransformer()

    print(dt.fit_transform(df.drop(['after'], axis=1), df['after']).head())
