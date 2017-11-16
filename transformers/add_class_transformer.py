import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
import xgboost as xgb
from transformers.item_selector import ItemSelector
from transformers.morphology_extractor import MorphologyExtractor
from transformers.sparse_union import SparseUnion
from transformers.string_to_chars import StringToChar
from sklearn.pipeline import Pipeline
from pandas.api.types import CategoricalDtype


class AddClassTransformer(TransformerMixin, BaseEstimator):
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

        morph_extractor = MorphologyExtractor(sparse=True)
        self.pipeline = SparseUnion([
            ('orig', Pipeline([
                ('select', ItemSelector('before')),
                ('features', SparseUnion([
                    ('char', StringToChar(10, to_coo=True)),
                    ('ctx', morph_extractor),
                ])),
            ])),
            ('prev', Pipeline([
                ('select', ItemSelector('prev')),
                ('features', SparseUnion([
                    ('char', StringToChar(5, to_coo=True)),
                    ('ctx', morph_extractor),
                ])),
            ])),
            ('next', Pipeline([
                ('select', ItemSelector('next')),
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
        dpredict = xgb.DMatrix(x_predict)
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
    df['prev'] = df['before'].shift(1).fillna('').to_dense()
    df['next'] = df['before'].shift(-1).fillna('').to_dense()
    print(df)

    ct = AddClassTransformer(modelpath='models/class.model.train_1190428_0.00101_0.3_500_6')

    print(ct.fit_transform(df))
