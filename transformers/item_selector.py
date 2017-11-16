from sklearn.base import BaseEstimator, TransformerMixin


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None, **fit_params):
        return self

    def transform(self, x, y=None, **fit_params):
        return x[self.key]


class Reshape2d(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None, **fit_params):
        return self

    def transform(self, x, y=None, **fit_params):
        return x.values.reshape(len(x), -1)


class ToCategoryCodes(BaseEstimator, TransformerMixin):
    def __init__(self, category_type=None):
        self.category_type = category_type

    def fit(self, x, y=None, **fit_params):
        return self

    def transform(self, x, y=None, **fit_params):
        if self.category_type:
            return x.astype(self.category_type).cat.codes
        else:
            return x.cat.codes
