import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score
from transformers.dummies_encoder import DummiesEncoder
from transformers.item_selector import ItemSelector
from transformers.morphology_extractor import MorphologyExtractor
from transformers.dict_class_transformer import DictClassTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from transformers.multi_label_encoder import MultiLabelEncoder
from transformers.reshape_2d import Reshape2D
from transformers.string_splitter import StringSplitter
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_PATH = r'../input/norm_challenge_ru'
DATA_INPUT_PATH = r'../input/norm_challenge_ru/ru_with_types'
SUBM_PATH = INPUT_PATH

df = pd.read_csv(os.path.join(INPUT_PATH, 'ru_train.csv'),
                 encoding='utf-8',
                 index_col=False,
                 usecols=['before', 'class'])

df = df.head(10000)

# df['before'] = df['before'].str.lower()
# df['after'] = df['after'].str.lower()
# df['after_wc'] = df['after'].map(lambda x: len(str(x).split()))
# df['before_prevprev'] = df['before'].shift(2)
df['before_prev'] = df['before'].shift(1)
df['before_next'] = df['before'].shift(-1)
# df['before_nextnext'] = df['before'].shift(-2)
df['before_len'] = df['before'].str.len()
df['before_wc'] = df['before'].map(lambda x: len(str(x).split()))
df = df.fillna('')
print(df.info())

classes = ['PLAIN', 'DATE', 'PUNCT', 'ORDINAL', 'VERBATIM', 'LETTERS', 'CARDINAL', 'MEASURE', 'TELEPHONE', 'ELECTRONIC',
           'DECIMAL', 'DIGIT', 'FRACTION', 'MONEY', 'TIME']

pipeline = Pipeline([
    ('features', FeatureUnion([
        ('chars', Pipeline([
            ('select', ItemSelector('before')),
            ('split', StringSplitter())
        ])),
        ('context', Pipeline([
            ('select', ItemSelector('before')),
            ('extract', MorphologyExtractor()),
            ('one_hot', DummiesEncoder())
        ])),
        ('chars_prev', Pipeline([
            ('select', ItemSelector('before_prev')),
            ('split', StringSplitter())
        ])),
        ('context_prev', Pipeline([
            ('select', ItemSelector('before_prev')),
            ('extract', MorphologyExtractor()),
            ('one_hot', DummiesEncoder())
        ])),
        ('chars_next', Pipeline([
            ('select', ItemSelector('before_next')),
            ('split', StringSplitter())
        ])),
        ('context_next', Pipeline([
            ('select', ItemSelector('before_next')),
            ('extract', MorphologyExtractor()),
            ('one_hot', DummiesEncoder())
        ])),
        ('length', Pipeline([
            ('select', ItemSelector('before_len')),
            ('reshape', Reshape2D())
        ])),
        ('words_count', Pipeline([
            ('select', ItemSelector('before_wc')),
            ('reshape', Reshape2D())
        ])),
    ], n_jobs=1)),
])

x_data = pipeline.fit_transform(df.drop(['class'], axis=1))

y_data = pd.factorize(df['class'])
labels = y_data[1]
y_data = y_data[0]

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=2017)

dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)
watchlist = [(dtest, 'test'), (dtrain, 'train')]

param = {'objective': 'multi:softmax',
         'eta': '0.3',
         'max_depth': 5,
         'silent': 1,
         'nthread': -1,
         'num_class': len(labels),
         'eval_metric': 'merror',
         'seed': '2017'}
model = xgb.train(param, dtrain, 50, watchlist, early_stopping_rounds=20, verbose_eval=10)

predicted_val = model.predict(dtrain)
print(f'pipeline val error {1.0-accuracy_score(y_train, predicted_val)}', flush=True)
predicted = model.predict(dtest)
print(f'pipeline test error {1.0-accuracy_score(y_test, predicted)}', flush=True)

xgb.plot_importance(model, color='red')
plt.show()
