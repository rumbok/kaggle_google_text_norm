import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score

from transformers.item_selector import ItemSelector
from transformers.morphology_extractor import MorphologyExtractor
from transformers.dict_class_transformer import DictClassTransformer
from sklearn.pipeline import Pipeline, FeatureUnion

from transformers.multi_label_encoder import MultiLabelEncoder
from transformers.string_splitter import StringSplitter

INPUT_PATH = r'../input/norm_challenge_ru'
DATA_INPUT_PATH = r'../input/norm_challenge_ru/ru_with_types'
SUBM_PATH = INPUT_PATH

df = pd.read_csv(os.path.join(INPUT_PATH, 'ru_train.csv'), encoding='utf-8', index_col=False)

df = df.head(10000)

df['before'] = df['before'].str.lower()
df['after'] = df['after'].str.lower()
# df['after_wc'] = df['after'].map(lambda x: len(str(x).split()))
# df['before_prevprev'] = df['before'].shift(2)
df['before_prev'] = df['before'].shift(1)
df['before_next'] = df['before'].shift(-1)
# df['before_nextnext'] = df['before'].shift(-2)
df = df.fillna('')

print(df.info())

classes = ['PLAIN', 'DATE', 'PUNCT', 'ORDINAL', 'VERBATIM', 'LETTERS', 'CARDINAL', 'MEASURE', 'TELEPHONE', 'ELECTRONIC',
           'DECIMAL', 'DIGIT', 'FRACTION', 'MONEY', 'TIME']

x_train, x_valid, y_train, y_valid = train_test_split(df.drop(['class'], axis=1), df['class'], test_size=0.1)

pipeline = Pipeline([
    ('features', FeatureUnion([
        ('chars', Pipeline([
            ('select', ItemSelector('before')),
            ('split', StringSplitter())
        ])),
        # ('chars_prev', Pipeline([
        #     ('select', ItemSelector('before_prev')),
        #     ('split', StringSplitter())
        # ])),
        # ('chars_next', Pipeline([
        #     ('select', ItemSelector('before_next')),
        #     ('split', StringSplitter())
        # ])),
        ('context', Pipeline([
            ('select', ItemSelector('before')),
            ('extract', MorphologyExtractor()),
            ('label', MultiLabelEncoder(['is_first_upper', 'is_upper', 'pos', 'animacy',
                                         'aspect', 'case', 'gender', 'mood', 'number',
                                         'person', 'tense', 'transitivity', 'voice'])),
            ('one_hot', OneHotEncoder())
        ])),
        # ('context_prev', Pipeline([
        #     ('select', ItemSelector('before_prev')),
        #     ('extract', ContextExtractor()),
        #     ('label', MultiLabelEncoder(['is_first_upper', 'is_upper', 'pos', 'animacy',
        #                                  'aspect', 'case', 'gender', 'mood', 'number',
        #                                  'person', 'tense', 'transitivity', 'voice'])),
        #     ('one_hot', OneHotEncoder())
        # ])),
        # ('context_next', Pipeline([
        #     ('select', ItemSelector('before_next')),
        #     ('extract', ContextExtractor()),
        #     ('label', MultiLabelEncoder(['is_first_upper', 'is_upper', 'pos', 'animacy',
        #                                  'aspect', 'case', 'gender', 'mood', 'number',
        #                                  'person', 'tense', 'transitivity', 'voice'])),
        #     ('one_hot', OneHotEncoder())
        # ]))
    ], n_jobs=1)),

    ('xgb', xgb.XGBClassifier(n_jobs=-1, objective='multi:softmax', max_depth=10, silent=1, learning_rate=0.3))
])

# pipeline = Pipeline([
#     ('dash', DashTransformer()),
#     ('digit', DigitTransformer()),
#     ('punсt', DictClassTransformer(u'PUNCT', 1.0)),
#     ('verbatim', DictClassTransformer(u'VERBATIM', 1.0)),
#     ('dict_nbhd', DictNBHDTransformer(1.0)),
#     ('dict', DictTransformer(1.0)),
#     ('flat', FlatTransformer())
# ])
fit_params = {'xgb__eval_metric': 'merror',
              'xgb__early_stopping_rounds': 20}
model = pipeline.fit(x_train, y_train, xgb__eval_metric='merror')
predicted_val = model.predict(x_train)
print(f'pipeline val TP {accuracy_score(y_train, predicted_val)}', flush=True)
print(f'pipeline val FP {1.0-accuracy_score(y_train, predicted_val)}', flush=True)
predicted = model.predict(x_valid)
print(f'pipeline test TP {accuracy_score(y_valid, predicted)}', flush=True)
print(f'pipeline test FP {1.0-accuracy_score(y_valid, predicted)}', flush=True)

import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = (7.0, 7.0)
xgb.plot_importance(model)
plt.show()

# predicted_val = pipeline.fit_transform(x_train, y_train)
# print(f'pipeline val TP {np.mean(predicted_val["after"] == y_train)}')
# print(f'pipeline val FP {np.mean(~(predicted_val["after"] == "") & ~(predicted_val["after"] == y_train))}')
# predicted = pipeline.transform(x_valid)
# print(f'pipeline test TP {np.mean(predicted["after"] == y_valid)}')
# print(f'pipeline test FP {np.mean(~(predicted["after"] == "") & ~(predicted["after"] == y_valid))}')




# dict_pnc = DictClassTransformer(u'PUNCT')
# dict_pnc.fit(x_train, y_train)
# predicted = dict_pnc.transform(x_valid)
# print(f'dict test TP {np.mean(predicted["after"] == y_valid)}')
# print(f'dict test FP {np.mean(~(predicted["after"].isnull()) & ~(predicted["after"] == y_valid))}')
# print(f'confidence {dict_pnc.mean_confidence}')
#
#
# dict_pnc = DictClassTransformer(u'VERBATIM')
# dict_pnc.fit(x_train, y_train)
# predicted = dict_pnc.transform(x_valid)
# print(f'dict test TP {np.mean(predicted["after"] == y_valid)}')
# print(f'dict test FP {np.mean(~(predicted["after"].isnull()) & ~(predicted["after"] == y_valid))}')
# print(f'confidence {dict_pnc.mean_confidence}')

# dict_tr = DictTransformer(1.0)
# dict_tr.fit(x_train, y_train)
# predicted = dict_tr.transform(x_valid)
# print(f'dict test TP {np.mean(predicted["after"] == y_valid)}')
# print(f'dict test FP {np.mean(~(predicted["after"] == "") & ~(predicted["after"] == y_valid))}')
# print(f'confidence {dict_tr.mean_confidence}')
#
#
# dict_context_tr = DictNBHDTransformer(1.0)
# dict_context_tr.fit(x_train, y_train)
# predicted = dict_context_tr.transform(x_valid)
# print(f'dict with context TP {np.mean(predicted["after"] == y_valid)}')
# print(f'dict with context FP {np.mean(~(predicted["after"] == "") & ~(predicted["after"] == y_valid))}')
# print(f'confidence {dict_context_tr.mean_confidence}')
