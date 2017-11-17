import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

from transformers.add_class_transformer import AddClassTransformer
from transformers.case_extractor import CaseExtractor
from transformers.item_selector import ItemSelector, Reshape2d, ToCategoryCodes
from transformers.morphology_extractor import MorphologyExtractor
from transformers.sparse_union import SparseUnion
from transformers.string_to_chars import StringToChar
from sklearn.pipeline import Pipeline
from pandas.api.types import CategoricalDtype


class AddCaseTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, modelpath=''):
        self.modelpath = modelpath
        self.model = None
        if self.modelpath:
            self.model = xgb.Booster()
            self.model.load_model(modelpath)

        self.class_type = CategoricalDtype(
            categories=['PLAIN', 'DATE', 'PUNCT', 'ORDINAL', 'VERBATIM', 'LETTERS', 'CARDINAL', 'MEASURE',
                        'TELEPHONE', 'ELECTRONIC', 'DECIMAL', 'DIGIT', 'FRACTION', 'MONEY', 'TIME',
                        'TRANS', 'DASH'])

        morph_extractor = MorphologyExtractor(sparse=True, multi_words=True)
        self.pipeline = SparseUnion([
            ('class', Pipeline([
                ('select', ItemSelector('class')),
                ('codes', ToCategoryCodes(self.class_type)),
                ('reshape', Reshape2d()),
                ('onehot', OneHotEncoder(n_values=len(self.class_type.categories), sparse=True, dtype=np.uint8))
            ])),
            ('orig', Pipeline([
                ('select', ItemSelector('before')),
                ('features', SparseUnion([
                    ('char', StringToChar(10, to_coo=True)),
                    ('ctx', morph_extractor),
                ])),
            ])),
            ('prev_prev', Pipeline([
                ('select', ItemSelector('prev_prev')),
                ('features', SparseUnion([
                    ('char', StringToChar(-5, to_coo=True)),
                    ('ctx', morph_extractor),
                ])),
            ])),
            ('prev', Pipeline([
                ('select', ItemSelector('prev')),
                ('features', SparseUnion([
                    ('char', StringToChar(-5, to_coo=True)),
                    ('ctx', morph_extractor),
                ])),
            ])),
            ('next', Pipeline([
                ('select', ItemSelector('next')),
                ('features', SparseUnion([
                    ('char', StringToChar(-5, to_coo=True)),
                    ('ctx', morph_extractor),
                ])),
            ])),
            ('next_next', Pipeline([
                ('select', ItemSelector('next_next')),
                ('features', SparseUnion([
                    ('char', StringToChar(-5, to_coo=True)),
                    ('ctx', morph_extractor),
                ])),
            ])),
        ])
        self.case_extractor = CaseExtractor(multi_words=True)

    def fit(self, X: pd.DataFrame, y=None, *args, **kwargs):
        if self.model is None:
            dtrain = xgb.DMatrix(data=self.pipeline.fit_transform(X),
                                 label=self.case_extractor.fit_transform(y)['case'].cat.codes)
            param = {'objective': 'multi:softmax',
                     'learning_rate': 0.2,
                     'num_boost_round': 400,
                     'max_depth': 6,
                     'silent': 1,
                     'nthread': 4,
                     'njobs': 4,
                     'num_class': len(set(dtrain.get_label())) + 1,
                     'eval_metric': ['merror', 'mlogloss']}
            self.model = xgb.train(param, dtrain, num_boost_round=param['num_boost_round'],
                                   early_stopping_rounds=20, verbose_eval=1)
        return self

    def transform(self, X: pd.DataFrame, y=None, *args, **kwargs):
        x_predict = self.pipeline.fit_transform(X)
        dpredict = xgb.DMatrix(x_predict)
        del x_predict
        predicted = pd.Series(pd.Categorical.from_codes(self.model.predict(dpredict),
                                                        categories=self.case_extractor.case_type.categories),
                              index=X.index)
        del dpredict
        if 'case' in X.columns:
            return X.assign(**{'case': X['case'].combine_first(predicted)})
        else:
            return X.assign(**{'case': predicted})


if __name__ == '__main__':
    df = pd.DataFrame(['в 1905 году', '25 января 1933 г', '123', '123', '-', '321', '&', '0546']
                            + 'съешь ещё этих мягких французских булок, да выпей чаю по - фиг'.split(),
                            columns=['before'])
    df['prev_prev'] = df['before'].shift(2).fillna('').to_dense()
    df['prev'] = df['before'].shift(1).fillna('').to_dense()
    df['next'] = df['before'].shift(-1).fillna('').to_dense()
    df['prev'] = df['prev']
    df['next'] = df['next']
    df['next_next'] = df['before'].shift(-2).fillna('').to_dense()
    print(df)

    class_trns = AddClassTransformer(modelpath='models/class.model.train_1190428_0.00101_0.3_500_6')
    case_trns = AddCaseTransformer(modelpath='case.model.train_502554_0.02781_0.3_500_7')

    print(case_trns.fit_transform(class_trns.fit_transform(df)))
