import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
import xgboost as xgb
from transformers.item_selector import ItemSelector
from transformers.morphology_extractor import MorphologyExtractor
from transformers.sparse_union import SparseUnion
from transformers.string_to_chars import StringToChar
from sklearn.pipeline import Pipeline
from pandas.api.types import CategoricalDtype
from models.trans_helpers import train_model, test_model, create_model


LAYER_NUM = 3
HIDDEN_DIM = 128
EMBEDDING_DIM = 32
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MEM_SIZE = 10000
NB_EPOCH = 3


class LSTMTransliterator(TransformerMixin, BaseEstimator):
    def __init__(self,
                 modelpath='',
                 layer_num=LAYER_NUM, hidden_dim=HIDDEN_DIM, batch_size=BATCH_SIZE,
                 learning_rate=LEARNING_RATE, mem_size=MEM_SIZE, num_epochs=NB_EPOCH):
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
            param = {'objective': 'multi:softmax',
                     'tree_method': 'hist',
                     'learning_rate': 0.3,
                     'num_boost_round': 500,
                     'max_depth': 6,
                     'silent': 1,
                     'nthread': 4,
                     'njobs': 4,
                     'num_class': len(set(dtrain.get_label())),
                     'eval_metric': ['merror', 'mlogloss'],
                     'seed': '2017'}
            self.model = xgb.train(param, dtrain, num_boost_round=param['num_boost_round'],
                                   early_stopping_rounds=50, verbose_eval=1)
        return self

    def transform(self, X: pd.DataFrame, y=None, *args, **kwargs):
        x_predict = self.pipeline.fit_transform(X)
        dpredict = xgb.DMatrix(x_predict, nthread=4)
        del x_predict
        predicted = pd.Series(pd.Categorical.from_codes(self.model.predict(dpredict),
                                                        categories=self.class_type.categories),
                              index=X.index)
        del dpredict
        if 'class' in X.columns:
            return X.assign(**{'class': X['class'].combine_first(predicted)})
        else:
            return X.assign(**{'class': predicted})


if __name__ == '__main__':
    df = pd.SparseDataFrame(['в 1905 году', '25 января 1933 г', '123', '123', '-', '321', '&', '0546']
                            + 'съешь ещё этих мягких французских булок, да выпей чаю по - фиг'.split(),
                            columns=['before'])
    df['before_prev'] = df['before'].shift(1).fillna('').to_dense()
    df['before_next'] = df['before'].shift(-1).fillna('').to_dense()
    print(df)

    ct = AddClassTransformer(modelpath='../models/class.model.train_1190428_0.00101_0.3_500_6')

    print(ct.fit_transform(df))
