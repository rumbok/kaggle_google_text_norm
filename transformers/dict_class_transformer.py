from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from tqdm import tqdm


class DictClassTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, classname, threshold=0.0):
        self.classname = classname
        self.threshold = threshold
        self.word_dict = defaultdict(Counter)
        self.mean_confidence = 0.0

    def fit(self, X, y=None, *args, **kwargs):
        for (before, cls, after) in tqdm(zip(X['before'], X['class'], y),
                                         f'{self.__class__.__name__}_{self.classname} fit',
                                         total=len(X)):
            if cls == self.classname:
                self.word_dict[before][after] += 1
            elif self.classname == 'TRANS' and '_trans' in after:
                self.word_dict[before][after] += 1
        return self

    def _most_common(self):
        self.mean_confidence = 0.0
        kv = {}
        for key in tqdm(self.word_dict,
                        f'{self.__class__.__name__}_{self.classname} pre transform',
                        total=len(self.word_dict.keys())):
            most = self.word_dict[key].most_common(1)
            confidence = most[0][1] / sum(self.word_dict[key].values())
            kv[key] = (most[0][0], confidence)
            self.mean_confidence += confidence
        self.mean_confidence /= len(kv.keys()) if len(kv.keys()) > 0 else 1.0

        return kv

    def transform(self, X: pd.DataFrame, y=None, *args, **kwargs):
        data = []
        kv = self._most_common()
        for before in tqdm(X['before'], f'{self.__class__.__name__}_{self.classname} transform', total=len(X)):
            if before in kv and kv[before][1] >= self.threshold:
                data.append(kv[before][0])
            else:
                data.append(None)
        del kv

        if 'after' in X.columns:
            return X.assign(after=X['after'].combine_first(pd.Series(data, index=X.index)))
        else:
            return X.assign(after=data)

    def get_params(self):
        params = super(self.__class__, self).get_params()
        params['mean_confidence'] = self.mean_confidence
        return params


if __name__ == '__main__':
    df = pd.DataFrame(np.random.randn(10, 3), columns=['before', 'class', 'after'])
    dt = DictClassTransformer('PUNCT')

    print(dt.fit_transform(df.drop(['after'], axis=1), df['after']).head())
