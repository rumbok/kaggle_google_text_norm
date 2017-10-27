from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from tqdm import tqdm, tqdm_pandas


class DictNBHDTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self.kv = {}
        self.mean_confidence = 0.0

    def fit(self, X, y=None, *args, **kwargs):
        word_dict = defaultdict(Counter)
        for (w_prev, w, w_next, after) in tqdm(zip(X['before_prev'], X['before'], X['before_next'], y),
                                               f'{self.__class__.__name__} fit stage 1',
                                               total=len(X)):
            word_dict[w_prev + w + w_next][after] += 1

        self.mean_confidence = 0.0
        for threegramm in tqdm(word_dict,
                               f'{self.__class__.__name__} fit stage 2',
                               total=len(word_dict.keys())):
            most = word_dict[threegramm].most_common(1)
            confidence = most[0][1] / sum(word_dict[threegramm].values())
            self.kv[threegramm] = (most[0][0], confidence)
            self.mean_confidence += confidence

        self.mean_confidence /= len(self.kv.keys())
        return self

    def transform(self, X: pd.DataFrame, y=None, *args, **kwargs):
        data = []
        for (w_prev, w, w_next) in tqdm(zip(X['before_prev'], X['before'], X['before_next']),
                                        f'{self.__class__.__name__} transform',
                                        total=len(X)):
            threegramm = w_prev + w + w_next
            if threegramm in self.kv and self.kv[threegramm][1] >= self.threshold:
                data.append(self.kv[threegramm][0])
            else:
                data.append(None)
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

    print(dt.fit_transform(df, df['after']).head())
