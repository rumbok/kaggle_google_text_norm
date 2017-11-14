import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
import xgboost as xgb
from transformers.item_selector import ItemSelector
from transformers.morphology_extractor import MorphologyExtractor
from transformers.sparse_union import SparseUnion
from transformers.string_to_chars import StringToChar
from sklearn.pipeline import Pipeline


class SelfTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, threshold=0.5, modelpath=''):
        self.threshold = threshold
        self.modelpath = modelpath
        self.model = None
        if self.modelpath:
            self.model = xgb.Booster()
            self.model.load_model(modelpath)

        morph_extractor = MorphologyExtractor(to_coo=True)
        self.pipeline = SparseUnion([
            ('orig', Pipeline([
                ('select', ItemSelector('before')),
                ('features', SparseUnion([
                    ('char', StringToChar(10, to_coo=True)),
                    ('ctx', morph_extractor),
                ])),
            ])),
            ('prev', Pipeline([
                ('select', ItemSelector('before_prev')),
                ('features', SparseUnion([
                    ('char', StringToChar(5, to_coo=True)),
                    ('ctx', morph_extractor),
                ])),
            ])),
            ('next', Pipeline([
                ('select', ItemSelector('before_next')),
                ('features', SparseUnion([
                    ('char', StringToChar(5, to_coo=True)),
                    ('ctx', morph_extractor),
                ])),
            ]))
        ])

    def fit(self, X: pd.DataFrame, y=None, *args, **kwargs):
        if self.model is None:
            dtrain = xgb.DMatrix(self.pipeline.fit_transform(X), label=y)
            param = {'objective': 'binary:logistic',
                     'tree_method': 'hist',
                     'learning_rate': 0.3,
                     'num_boost_round': 500,
                     'max_depth': 6,
                     'silent': 1,
                     'nthread': 4,
                     'njobs': 4,
                     'scale_pos_weight': 1 / np.mean(dtrain.get_label()) - 1,
                     'eval_metric': ['error', 'auc'],
                     'seed': '2017'}
            self.model = xgb.train(param, dtrain, num_boost_round=param['num_boost_round'],
                              early_stopping_rounds=50, verbose_eval=1)
        return self

    def transform(self, X: pd.DataFrame, y=None, *args, **kwargs):
        x_predict = self.pipeline.fit_transform(X)
        dpredict = xgb.DMatrix(x_predict)
        del x_predict
        predicted = pd.Series(self.model.predict(dpredict), index=X.index)
        del dpredict
        if 'after' in X.columns:
            return X.assign(after=X['after'].combine_first(X[predicted >= self.threshold]['before']))
        else:
            return X.assign(after=X[predicted >= self.threshold]['before'])


if __name__ == '__main__':
    df = pd.SparseDataFrame(['в 1905 году', '123', '123', '-', '321', '&', '0546']
                            + 'съешь ещё этих мягких французских булок, да выпей чаю по - фиг'.split(),
                            columns=['before'])
    df['before_prev'] = df['before'].shift(1).fillna('').to_dense()
    df['before_next'] = df['before'].shift(-1).fillna('').to_dense()
    print(df)

    st = SelfTransformer(threshold=0.5, modelpath='../models/self.model.train')

    print(st.fit_transform(df))
