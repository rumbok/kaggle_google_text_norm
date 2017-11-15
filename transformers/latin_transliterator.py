import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from tqdm import tqdm


LATIN = frozenset(['A', 'a', 'p', 'I', 'C', 'X', 'B', 'e', 'i', 'V', 'R', 'c', 'D', 'F',
                   'P', 'U', 'y', 'T', 'O', 'E', 'v', 'K', 'f', 'o', 'H', 'Q', 'd', 'x',
                   'u', 'Y', 'b', 't', 'k', 'h', 'r', 'q'])


class LatinTransliterator(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None, *args, **kwargs):
        return self

    def transform(self, X: pd.DataFrame, y=None, *args, **kwargs):
        data = []
        for w in tqdm(X['before'], f'{self.__class__.__name__} transform', total=len(X)):
            if w in LATIN:
                data.append(f'{w.lower()}_latin')
            else:
                data.append(None)
        if 'after' in X.columns:
            return X.assign(after=X['after'].combine_first(pd.Series(data, index=X.index)))
        else:
            return X.assign(after=data)
