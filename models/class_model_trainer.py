import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.datasets import dump_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from loaders.loading import load_train
from transformers.pandas_dummies import PandasDummies
from transformers.item_selector import ItemSelector
from transformers.morphology_extractor import MorphologyExtractor
from transformers.pandas_union import PandasUnion
from transformers.string_to_chars import StringToChar

# df = load_train(['before', 'after', 'class']).fillna('')
# df['before_prev'] = df['before'].shift(1).fillna('')
# df['before_next'] = df['before'].shift(-1).fillna('')
# df = df[~(df['before'] == df['after'])]
# del df['after']
# print(df.info())
#
# morph_extractor = MorphologyExtractor()
# morph_dummies = PandasDummies(['pos', 'animacy', 'aspect', 'case', 'gender', 'involvement', 'mood', 'number', 'person', 'tense', 'transitivity'])
# pipeline = Pipeline([
#     ('features', PandasUnion([
#         ('char', Pipeline([
#             ('select', ItemSelector('before')),
#             ('split', StringSplitter(10))
#         ])),
#         ('ctx', Pipeline([
#             ('select', ItemSelector('before')),
#             ('extract', morph_extractor),
#             ('one_hot', morph_dummies)
#         ])),
#         ('char_prev', Pipeline([
#             ('select', ItemSelector('before_prev')),
#             ('split', StringSplitter(5))
#         ])),
#         ('ctx_prev', Pipeline([
#             ('select', ItemSelector('before_prev')),
#             ('extract', morph_extractor),
#             ('one_hot', morph_dummies)
#         ])),
#         ('char_next', Pipeline([
#             ('select', ItemSelector('before_next')),
#             ('split', StringSplitter(5))
#         ])),
#         ('ctx_next', Pipeline([
#             ('select', ItemSelector('before_next')),
#             ('extract', morph_extractor),
#             ('one_hot', morph_dummies)
#         ])),
#     ])),
# ])
#
# x_data = pipeline.fit_transform(df.drop(['class'], axis=1))
# y_data = pd.factorize(df['class'])
# labels = y_data[1]
# y_data = y_data[0]
# del morph_extractor
# del morph_dummies
# del df
#
# print(x_data.info())
# print(x_data.density)
#
# x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=2017)
# print('data splitted')
# del x_data
# del y_data
#
# dump_svmlight_file(x_test, y_test, 'class.txt.test')
# dump_svmlight_file(x_train, y_train, 'class.txt.train')
# del x_train, x_test, y_train, y_test

dtest = xgb.DMatrix('class.txt.test#class.dtest.cache')
dtrain = xgb.DMatrix('class.txt.train#class.dtrain.cache')

# dtrain = xgb.DMatrix(x_train, label=y_train)
# dtest = xgb.DMatrix(x_test, label=y_test)
watchlist = [(dtest, 'test'), (dtrain, 'train')]

param = {'objective': 'multi:softmax',
         'learning_rate': 0.3,
         'num_boost_round': 90,
         'max_depth': 4,
         'silent': 1,
         'nthread': 4,
         'num_class': len(set(dtrain.get_label())),
         'eval_metric': ['merror', 'mlogloss'],
         'seed': '2017'}
model = xgb.train(param, dtrain, num_boost_round=param['num_boost_round'], evals=watchlist,
                  early_stopping_rounds=30, verbose_eval=1)
model.save_model('class_model_train.bin')

plt.rcParams['font.size'] = 8
feat_imp = pd.Series(model.get_fscore()).sort_values(ascending=False)
print(feat_imp.index)
# print(x_data.columns[~x_data.columns.isin(feat_imp.index)])
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')
plt.tight_layout()
# plt.savefig('class_features_imp.png')
plt.show()
