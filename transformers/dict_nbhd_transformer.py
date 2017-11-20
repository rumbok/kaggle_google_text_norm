from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from tqdm import tqdm
from loaders.loading import load_train

INPUT_PATH = r'../input/norm_challenge_ru'


class DictNBHDTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self.mean_confidence = 0.0
        self.kv = {}

    def fit(self, X, y=None, *args, **kwargs):
        word_dict = {}
        threegramms = (X['prev'].map(str) + X['before'].map(str) + X['next'].map(str)).str.lower()
        for (tgr, after) in tqdm(zip(threegramms, y),
                                 f'{self.__class__.__name__} fit stage 1',
                                 total=len(X)):
            hsh = str.__hash__(tgr)
            if hsh in word_dict:
                if isinstance(word_dict[hsh], Counter):
                    word_dict[hsh][after] += 1
                else:
                    word_dict[hsh] = Counter([word_dict[hsh], after])
            else:
                word_dict[hsh] = after
        del threegramms

        for hsh in tqdm(word_dict,
                        f'{self.__class__.__name__} fit stage 2',
                        total=len(word_dict.keys())):
            if isinstance(word_dict[hsh], Counter):
                if hsh in self.kv:
                    (prev_val, prev_conf) = self.kv[hsh]
                    if prev_conf == 1.0:
                        prev_count = 1
                    else:
                        count = sum(word_dict[hsh].values())
                        prev_count = int(max(1.0, count*prev_conf))
                    word_dict[hsh][prev_val] += prev_count

                most = word_dict[hsh].most_common(1)
                confidence = most[0][1] / sum(word_dict[hsh].values())
                self.kv[hsh] = (most[0][0], confidence)

            else:
                if hsh in self.kv:
                    (prev_val, prev_conf) = self.kv[hsh]
                    if prev_val == word_dict[hsh]:
                        self.kv[hsh] = (word_dict[hsh], 1.0)
                    else:
                        self.kv[hsh] = (prev_val, 1.0/(2.0-prev_conf))
                else:
                    self.kv[hsh] = (word_dict[hsh], 1.0)

        del word_dict

        print(len(self.kv))

        return self

    def _calc_mean_confidence(self):
        for val in self.kv.values():
            self.mean_confidence += val[1]
        self.mean_confidence /= len(self.kv.keys()) if len(self.kv.keys()) > 0 else 1.0

    def transform(self, X: pd.DataFrame, y=None, *args, **kwargs):
        self._calc_mean_confidence()
        data = []
        threegramms = (X['prev'].map(str) + X['before'].map(str) + X['next'].map(str)).str.lower()
        for tgr in tqdm(threegramms,
                        f'{self.__class__.__name__} transform',
                        total=len(X)):
            hsh = str.__hash__(tgr)
            if hsh in self.kv and self.kv[hsh][1] >= self.threshold:
                data.append(self.kv[hsh][0])
            else:
                data.append(None)
        del threegramms

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
    df['prev'] = df['before'].shift(1)
    df['next'] = df['before'].shift(-1)
    df['before'] = df['before']
    df['after'] = df['after']
    df = df.fillna('')
    print(df.info())

    dt = DictNBHDTransformer(0.5)

    dt.fit(df.drop(['after'], axis=1), df['after'])
    dt.fit(df.drop(['after'], axis=1), df['after'])

    res_df = dt.transform(df.rename(columns={'after': 'actual'}))
    print('Acc', len(res_df[res_df['after'] == res_df['actual']])/ len(res_df))

#1x Acc 0.9970029833989565
#2x Acc 0.9970147097039713
