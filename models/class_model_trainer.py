import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from sklearn.metrics import accuracy_score

from loading import load_train
from transformers.dummies_encoder import DummiesEncoder
from transformers.item_selector import ItemSelector
from transformers.morphology_extractor import MorphologyExtractor
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.datasets import dump_svmlight_file

from transformers.pandas_shift import PandasShift
from transformers.pandas_union import PandasUnion
from transformers.string_splitter import StringSplitter
import matplotlib.pyplot as plt
import seaborn as sns

df = load_train(['before', 'class']).head(10000)
print(df.info())

pipeline = Pipeline([
    # ('features', FeatureUnion([
    ('select', ItemSelector('before')),
    ('features', PandasUnion([
        ('char', Pipeline([
            ('split', StringSplitter(10))
        ])),
        ('ctx', Pipeline([
            ('extract', MorphologyExtractor()),
            ('one_hot', DummiesEncoder())
        ])),
        ('char_prev', Pipeline([
            ('shift', PandasShift(1)),
            ('split', StringSplitter(5))
        ])),
        ('ctx_prev', Pipeline([
            ('shift', PandasShift(1)),
            ('extract', MorphologyExtractor()),
            ('one_hot', DummiesEncoder())
        ])),
        ('char_next', Pipeline([
            ('shift', PandasShift(-1)),
            ('split', StringSplitter(5))
        ])),
        ('ctx_next', Pipeline([
            ('shift', PandasShift(-1)),
            ('extract', MorphologyExtractor()),
            ('one_hot', DummiesEncoder())
        ])),
    ])),
])

x_data = pipeline.fit_transform(df.drop(['class'], axis=1))
print(x_data.info())

y_data = pd.factorize(df['class'])
labels = y_data[1]
y_data = y_data[0]

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=2017)

dump_svmlight_file(x_train, y_train, 'models/class.txt.train')
dump_svmlight_file(x_test, y_test, 'models/class.txt.test')

dtrain = xgb.DMatrix('models/class.txt.train#dtrain.cache')
dtest = xgb.DMatrix('models/class.txt.test#dtest.cache')
watchlist = [(dtest, 'test'), (dtrain, 'train')]

param = {'objective': 'multi:softmax',
         'eta': '0.3',
         'max_depth': 5,
         'silent': 1,
         'nthread': -1,
         'num_class': len(labels),
         'eval_metric': 'merror',
         'seed': '2017'}
model = xgb.train(param, dtrain, 200, watchlist, early_stopping_rounds=50, verbose_eval=10)

predicted_val = model.predict(dtrain)
print(f'pipeline val error {1.0-accuracy_score(y_train, predicted_val)}', flush=True)
predicted = model.predict(dtest)
print(f'pipeline test error {1.0-accuracy_score(y_test, predicted)}', flush=True)

plt.rcParams['font.size'] = 8
feat_imp = pd.Series(model.get_fscore()).sort_values(ascending=False)
print(feat_imp.index)
print(x_data.columns[~x_data.columns.isin(feat_imp.index)])
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')
plt.tight_layout()
plt.show()
