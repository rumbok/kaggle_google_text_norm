from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from tqdm import tqdm
from loaders.loading import load_train

INPUT_PATH = r'../input/norm_challenge_ru'


class DictNBHDTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, threshold=0.0, to_lower=True):
        self.threshold = threshold
        self.word_dict = {}
        self.mean_confidence = 0.0
        self.to_lower = to_lower

    def fit(self, X, y=None, *args, **kwargs):
        if self.to_lower:
            threegramms = (X['prev'].map(str) + X['before'].map(str) + X['next'].map(str)).str.lower()
            afters = y.map(str).str.lower()
        else:
            threegramms = X['prev'].map(str) + X['before'].map(str) + X['next'].map(str)
            afters = y

        for (tgr, after) in tqdm(zip(threegramms, afters),
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
        del threegramms, afters


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
        self.word_dict = {}
        return kv

    def transform(self, X: pd.DataFrame, y=None, *args, **kwargs):
        data = []
        kv = self._most_common()
        if self.to_lower:
            threegramms = (X['prev'].map(str) + X['before'].map(str) + X['next'].map(str)).str.lower()
        else:
            threegramms = X['prev'].map(str) + X['before'].map(str) + X['next'].map(str)
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
    df = load_train(columns=['before', 'after'], input_path=INPUT_PATH)
    df['prev'] = df['before'].shift(1).str.lower()
    df['next'] = df['before'].shift(-1).str.lower()
    df['before'] = df['before'].str.lower()
    df['after'] = df['after'].str.lower()
    df = df.fillna('')
    print(df.info())

    dt = DictNBHDTransformer(0.5, to_lower=False)

    dt.fit(df.drop(['after'], axis=1), df['after'])
    print('All threegramms', len(df))
    print('Unique threegramms', len(dt.word_dict)) #7711445 7614565

    res_df = dt.transform(df.rename(columns={'after': 'actual'}))
    print('Acc', len(res_df[res_df['after'] == res_df['actual']])/ len(res_df))
    print(res_df)

    #to_lower Acc 0.9995168573199946


