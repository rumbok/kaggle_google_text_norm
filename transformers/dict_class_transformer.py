from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from tqdm import tqdm


class DictClassTransformer(TransformerMixin):
    def __init__(self, classname, threshold=0.0):
        self.classname = classname
        self.threshold = threshold
        self.kv = {}
        self.mean_confidence = 0.0

    def fit(self, X, y=None, *args, **kwargs):
        word_dict = defaultdict(Counter)
        for (before, cls, after) in tqdm(zip(X['before'], X['class'], y), f'{self.__class__.__name__} fit stage 1', total=len(X)):
            if cls == self.classname:
                word_dict[before][after] += 1

        self.mean_confidence = 0.0
        for before in tqdm(word_dict, f'{self.__class__.__name__} fit stage 2', total=len(word_dict.keys())):
            most = word_dict[before].most_common(1)
            confidence = most[0][1] / sum(word_dict[before].values())
            self.kv[before] = (most[0][0], confidence)
            self.mean_confidence += confidence

        self.mean_confidence /= len(self.kv.keys()) if len(self.kv.keys()) > 0 else 1.0
        return self

    def transform(self, X: pd.DataFrame, y=None, *args, **kwargs):
        data = []
        for before in tqdm(X['before'], f'{self.__class__.__name__} transform', total=len(X)):
            if before in self.kv and self.kv[before][1] >= self.threshold:
                data.append(self.kv[before][0])
            else:
                data.append(None)
        if 'after' in X.columns:
            return X.assign(after=X['after'].combine_first(pd.Series(data, index=X.index)))
        else:
            return X.assign(after=data)


if __name__ == '__main__':
    df = pd.DataFrame(np.random.randn(10, 3), columns=['before', 'class', 'after'])
    dt = DictClassTransformer('PUNCT')

    print(dt.fit_transform(df.drop(['after'], axis=1), df['after']).head())
