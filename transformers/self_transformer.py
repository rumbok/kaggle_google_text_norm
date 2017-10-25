from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
import xgboost as xgb
from transformers.item_selector import ItemSelector
from transformers.morphology_extractor import MorphologyExtractor
from transformers.pandas_dummies import PandasDummies
from transformers.pandas_shift import PandasShift
from transformers.pandas_union import PandasUnion
from transformers.sparse_union import SparseUnion
from transformers.string_to_chars import StringToChar
from sklearn.pipeline import Pipeline


class SelfTransformer(TransformerMixin):
    def __init__(self, threshold=0.5, modelpath=''):
        self.threshold = threshold
        self.model = None
        if modelpath:
            self.model = xgb.Booster()
            self.model.load_model(modelpath)

        morph_extractor = MorphologyExtractor(to_coo=True)
        self.pipeline = Pipeline([
            ('select', ItemSelector('before')),
            ('features', SparseUnion([
                ('char', StringToChar(10, to_coo=True)),
                ('char_prev', Pipeline([
                    ('shift', PandasShift(1)),
                    ('split', StringToChar(5, to_coo=True))
                ])),
                ('char_next', Pipeline([
                    ('shift', PandasShift(-1)),
                    ('split', StringToChar(5, to_coo=True))
                ])),
                ('ctx', morph_extractor),
                ('ctx_prev', Pipeline([
                    ('shift', PandasShift(1)),
                    ('extract', morph_extractor)
                ])),
                ('ctx_next', Pipeline([
                    ('shift', PandasShift(-1)),
                    ('extract', morph_extractor)
                ])),
            ])),
        ])

    def fit(self, X: pd.DataFrame, y=None, *args, **kwargs):
        if self.model is None:
            dtrain = xgb.DMatrix(self.pipeline.fit_transform(X), label=y)
            param = {'objective': 'binary:logistic',
                     'tree_method': 'hist',
                     'learning_rate': 0.3,
                     'num_boost_round': 500,
                     'max_depth': 5,
                     'silent': 1,
                     'nthread': 4,
                     'scale_pos_weight': 1 / np.mean(dtrain.get_label()) - 1,
                     'eval_metric': ['auc'],
                     'seed': '2017'}
            self.model = xgb.train(param, dtrain, num_boost_round=param['num_boost_round'],
                                   early_stopping_rounds=30, verbose_eval=1)
        return self

    def transform(self, X: pd.DataFrame, y=None, *args, **kwargs):
        dpredict = xgb.DMatrix(self.pipeline.fit_transform(X))
        predicted = self.model.predict(dpredict)
        return pd.Series(predicted, name='self')
        if 'after' in X.columns:
            return X.assign(after=X['after'].combine_first(X[self_predicted > self.threshold]['before']))
        else:
            return X.assign(after=X[self_predicted > self.threshold]['before'])


if __name__ == '__main__':
    df = pd.SparseDataFrame(['в 1905 году', '123', 'dfhsd', '-', '&',
                             '0546'] + 'съешь ещё этих мягких французских булок, да выпей чаю'.split(),
                            columns=['before'])
    print(df)

    dt = SelfTransformer(threshold=0.5, modelpath='../models/self.model.train')

    print(dt.fit_transform(df))
