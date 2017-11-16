import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, FunctionTransformer

from loaders.loading import load_train
from transformers.case_extractor import CaseExtractor
from transformers.item_selector import ItemSelector, Reshape2d, ToCategoryCodes
from transformers.morphology_extractor import MorphologyExtractor
from transformers.sparse_union import SparseUnion
from transformers.string_to_chars import StringToChar
from sparse_helpers import sparse_memory_usage
import gc
from sklearn.metrics import accuracy_score
from pandas.api.types import CategoricalDtype


# INPUT_PATH = r'../input/norm_challenge_ru'
#
# df = load_train(['before', 'after', 'class'], INPUT_PATH).fillna('')
# #df = load_external(['before', 'after', 'class'],
# #                   only_diff=True,
# #                   input_path=r'../input/norm_challenge_ru/ru_with_types')\
# #      .fillna('')
#
# df['prev_prev'] = df['before'].shift(2).fillna('')
# df['before_prev'] = df['before'].shift(1).fillna('')
# df['before_next'] = df['before'].shift(-1).fillna('')
# df['next_next'] = df['before'].shift(-2).fillna('')
# classes = frozenset(['CARDINAL', 'DATE', 'MEASURE', 'DECIMAL', 'MONEY', 'ORDINAL', 'FRACTION', 'TIME'])
# df = df[~(df['before'] == df['after']) & (df['class'].isin(classes))]
# class_type = CategoricalDtype(categories=['PLAIN', 'DATE', 'PUNCT', 'ORDINAL', 'VERBATIM', 'LETTERS', 'CARDINAL',
#                                           'MEASURE', 'TELEPHONE', 'ELECTRONIC', 'DECIMAL', 'DIGIT', 'FRACTION',
#                                           'MONEY', 'TIME',
#                                           'TRANS', 'DASH'])
# print(df.info())
#
#
# morph_extractor = MorphologyExtractor(sparse=True, multi_words=True)
# before_pipeline = SparseUnion([
#     ('class', Pipeline([
#         ('select', ItemSelector('class')),
#         ('codes', ToCategoryCodes(class_type)),
#         ('reshape', Reshape2d()),
#         ('onehot', OneHotEncoder(n_values=len(class_type.categories), sparse=True, dtype=np.uint8))
#     ])),
#     ('orig', Pipeline([
#         ('select', ItemSelector('before')),
#         ('features', SparseUnion([
#             ('char', StringToChar(10, to_coo=True)),
#             ('ctx', morph_extractor),
#         ])),
#     ])),
#     ('prev_prev', Pipeline([
#         ('select', ItemSelector('prev_prev')),
#         ('features', SparseUnion([
#             ('char', StringToChar(-5, to_coo=True)),
#             ('ctx', morph_extractor),
#         ])),
#     ])),
#     ('prev', Pipeline([
#         ('select', ItemSelector('before_prev')),
#         ('features', SparseUnion([
#             ('char', StringToChar(-5, to_coo=True)),
#             ('ctx', morph_extractor),
#         ])),
#     ])),
#     ('next', Pipeline([
#         ('select', ItemSelector('before_next')),
#         ('features', SparseUnion([
#             ('char', StringToChar(-5, to_coo=True)),
#             ('ctx', morph_extractor),
#         ])),
#     ])),
#     ('next_next', Pipeline([
#         ('select', ItemSelector('next_next')),
#         ('features', SparseUnion([
#             ('char', StringToChar(-5, to_coo=True)),
#             ('ctx', morph_extractor),
#         ])),
#     ])),
# ])
#
# case_pipeline = CaseExtractor(multi_words=True)
#
# x_data = before_pipeline.fit_transform(df)
# print(f'data type={x_data.dtype}, '
#       f'size={x_data.shape}, '
#       f'density={x_data.nnz / x_data.shape[0] / x_data.shape[1]},'
#       f'{sparse_memory_usage(x_data):9.3} Mb')
# case_number_df = case_pipeline.fit_transform(df['after'])
# y_data = case_pipeline.fit_transform(df['after'])['case'].cat.codes
# del morph_extractor, before_pipeline, case_pipeline
# del df
# gc.collect()
#
# x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=2017)
# print(f'train type={x_train.dtype}, '
#       f'size={x_train.shape}, '
#       f'density={x_train.nnz / x_train.shape[0] / x_train.shape[1]},'
#       f'{sparse_memory_usage(x_train):9.3} Mb')
# print(f'test type={x_test.dtype}, '
#       f'size={x_test.shape}, '
#       f'density={x_test.nnz / x_test.shape[0] / x_test.shape[1]},'
#       f'{sparse_memory_usage(x_test):9.3} Mb')
# del x_data
# del y_data
# gc.collect()
#
#
# dtrain = xgb.DMatrix(x_train, label=y_train)
# dtest = xgb.DMatrix(x_test, label=y_test)
# del x_train, x_test, y_train, y_test
# gc.collect()
#
#
# dtrain.save_binary('case.matrix.train.train')
# dtest.save_binary('case.matrix.train.test')
# del dtrain
# del dtest
# gc.collect()


dtrain = xgb.DMatrix('case.matrix.train.train#case.dtrain.cache')
dtest = xgb.DMatrix('case.matrix.train.test#case.dtest.cache')
watchlist = [(dtrain, 'train'), (dtest, 'test')]

param = {'objective': 'multi:softmax',
         'learning_rate': 0.3,
         'num_boost_round': 500,
         'max_depth': 7,
         'silent': 1,
         'nthread': 4,
         'njobs': 4,
         'num_class': len(set(dtrain.get_label())) + 1,
         'eval_metric': ['merror', 'mlogloss'],
         'seed': '2017'}
model = xgb.train(param, dtrain, num_boost_round=param['num_boost_round'], evals=watchlist,
                  early_stopping_rounds=25, verbose_eval=1)

y_pred = model.predict(dtest)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(dtest.get_label(), predictions)
print("Accuracy: %.5f%%" % accuracy)

model.save_model(
    f'case.model.train_{len(dtrain.get_label())}_{1.0-accuracy:0.5f}_{param["learning_rate"]}_{param["num_boost_round"]}_{param["max_depth"]}')
