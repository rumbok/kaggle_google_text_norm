from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from tqdm import tqdm
import xgboost as xgb

from transformers.item_selector import ItemSelector
from transformers.morphology_extractor import MorphologyExtractor
from transformers.pandas_dummies import PandasDummies
from transformers.pandas_union import PandasUnion
from transformers.string_to_chars import StringToChar
from sklearn.pipeline import Pipeline


class ClassTransformer(TransformerMixin):
    def __init__(self, threshold=0.5, modelpath=''):
        self.threshold = threshold
        self.modelpath = modelpath
        self.model = xgb.Booster()
        if self.modelpath:
            self.model.load_model(modelpath)

    def fit(self, X, y=None, *args, **kwargs):
        return self

    def transform(self, X, y=None, *args, **kwargs):
        dpredict = xgb.DMatrix(X)
        predicted = self.model.predict(dpredict)
        return pd.Series(predicted, name='self')
        if 'after' in X.columns:
            return X.assign(after=X['after'].combine_first(X[self_predicted > self.threshold]['before']))
        else:
            return X.assign(after=X[self_predicted > self.threshold]['before'])

    def get_params(self):
        return f'threshold={self.threshold}, model={self.modelpath}'


if __name__ == '__main__':
    df = pd.SparseDataFrame([u'в 1905 году', u'123', u'dfhsd', u'-', u'&', u'0546'] + u'съешь ещё этих мягких французских булок, да выпей чаю'.split(), columns=['before'])
    df['before_prev'] = df['before'].shift(1).fillna('')
    df['before_next'] = df['before'].shift(-1).fillna('')
    print(df)

    morph_extractor = MorphologyExtractor()
    morph_dummies = PandasDummies(['pos', 'animacy', 'aspect', 'case', 'gender', 'involvement', 'mood', 'number', 'person', 'tense', 'transitivity'])
    pipeline = Pipeline([
        ('features', PandasUnion([
            ('char', Pipeline([
                ('select', ItemSelector('before')),
                ('split', StringToChar(10))
            ])),
            ('ctx', Pipeline([
                ('select', ItemSelector('before')),
                ('extract', morph_extractor),
                ('one_hot', morph_dummies)
            ])),
            ('char_prev', Pipeline([
                ('select', ItemSelector('before_prev')),
                ('split', StringToChar(5))
            ])),
            ('ctx_prev', Pipeline([
                ('select', ItemSelector('before_prev')),
                ('extract', morph_extractor),
                ('one_hot', morph_dummies)
            ])),
            ('char_next', Pipeline([
                ('select', ItemSelector('before_next')),
                ('split', StringToChar(5))
            ])),
            ('ctx_next', Pipeline([
                ('select', ItemSelector('before_next')),
                ('extract', morph_extractor),
                ('one_hot', morph_dummies)
            ])),
        ])),
    ])
    x_data = pipeline.fit_transform(df)
    print(x_data)

    dt = ClassTransformer(threshold=0.5, modelpath='../models/class_model_train.bin')

    print(dt.fit_transform(x_data))
