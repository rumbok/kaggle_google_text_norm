import numpy as np
from pandas import DataFrame
from sklearn.base import TransformerMixin, BaseEstimator


class FlatTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None, *args, **kwargs):
        return self

    def transform(self, X: DataFrame, y=None, *args, **kwargs):
        if 'after' in X.columns:
            return X.assign(after=X['after'].combine_first(X['before']))
        else:
            return X.assign(after=X['before'])


if __name__ == '__main__':
    df = DataFrame(np.random.randn(10, 2), columns=['before', 'after'])
    dt = FlatTransformer()

    print(dt.fit_transform(df.drop(['after'], axis=1), df['after']).head())
