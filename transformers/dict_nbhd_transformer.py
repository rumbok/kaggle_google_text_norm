from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from tqdm import tqdm


class DictNBHDTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self.word_dict = {}
        self.mean_confidence = 0.0

    def fit(self, X, y=None, *args, **kwargs):
        threegramms = X['before_prev'].map(str) + X['before'].map(str) + X['before_next'].map(str)
        for (tgr, after) in tqdm(zip(threegramms, y),
                                 f'{self.__class__.__name__} fit',
                                 total=len(X)):
            hsh = str.__hash__(tgr)
            if hsh in self.word_dict:
                if isinstance(self.word_dict[hsh], Counter):
                    self.word_dict[hsh][after] += 1
                else:
                    self.word_dict[hsh] = Counter([self.word_dict[hsh], after])
            else:
                self.word_dict[hsh] = after
        del threegramms
        #print(len(self.word_dict.keys()))
        return self

    def _most_common(self):
        self.mean_confidence = 0.0
        kv = {}
        for key in tqdm(self.word_dict,
                        f'{self.__class__.__name__} pre transform',
                        total=len(self.word_dict.keys())):
            if isinstance(self.word_dict[key], Counter):
                most = self.word_dict[key].most_common(1)
                confidence = most[0][1] / sum(self.word_dict[key].values())
                kv[key] = (most[0][0], confidence)
            else:
                confidence = 1.0
                kv[key] = (self.word_dict[key], confidence)
            self.mean_confidence += confidence
        self.mean_confidence /= len(kv.keys()) if len(kv.keys()) > 0 else 1.0
        del self.word_dict
        return kv

    def transform(self, X: pd.DataFrame, y=None, *args, **kwargs):
        data = []
        kv = self._most_common()
        threegramms = X['before_prev'].map(str) + X['before'].map(str) + X['before_next'].map(str)
        for tgr in tqdm(threegramms,
                        f'{self.__class__.__name__} transform',
                        total=len(X)):
            hsh = str.__hash__(tgr)
            #hsh = tgr
            if hsh in kv and kv[hsh][1] >= self.threshold:
                data.append(kv[hsh][0])
            else:
                data.append(None)
        del kv, threegramms

        if 'after' in X.columns:
            return X.assign(after=X['after'].combine_first(pd.Series(data, index=X.index)))
        else:
            return X.assign(after=data)

    def get_params(self):
        params = super(self.__class__, self).get_params()
        params['mean_confidence'] = self.mean_confidence
        return params


if __name__ == '__main__':
    df = pd.DataFrame(np.random.randn(10, 4), columns=['before_prev', 'before', 'before_next', 'after'])
    dt = DictNBHDTransformer()

    print(dt.fit_transform(df.drop(['after'], axis=1), df['after']).head())
