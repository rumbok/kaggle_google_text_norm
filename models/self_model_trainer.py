import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from loaders.loading import load_train
from transformers.item_selector import ItemSelector
from transformers.morphology_extractor import MorphologyExtractor
from transformers.pandas_shift import PandasShift
from transformers.sparse_union import SparseUnion
from transformers.string_to_chars import StringToChar
from sklearn_pandas import DataFrameMapper
from sparse_helpers import sparse_memory_usage
import gc


df = load_train(['before', 'after']).fillna('')
df['self'] = (df['before'] == df['after'])
del df['after']
print(df.info())


morph_extractor = MorphologyExtractor(to_coo=True)
pipeline = Pipeline([
    ('select', ItemSelector('before')),
    ('features', SparseUnion([
        ('char', StringToChar(10, to_coo=True)),
        ('char_prev', Pipeline([
            ('shift', PandasShift(1)),
            ('split', StringToChar(5, to_coo=True))
        ])),
        ('char_next', Pipeline([
            ('shift', PandasShift(-1)),
            ('split', StringToChar(5, to_coo=True))
        ])),
        ('ctx', morph_extractor),
        ('ctx_prev', Pipeline([
            ('shift', PandasShift(1)),
            ('extract', morph_extractor)
        ])),
        ('ctx_next', Pipeline([
            ('shift', PandasShift(-1)),
            ('extract', morph_extractor)
        ])),
    ])),
])


x_data = pipeline.fit_transform(df.drop(['self'], axis=1))
print(f'data type={x_data.dtype}, '
      f'size={x_data.shape}, '
      f'density={x_data.nnz / x_data.shape[0] / x_data.shape[1]},'
      f'{sparse_memory_usage(x_data):6.3} Mb')
y_data = df['self'].values
del morph_extractor
del df
gc.collect()


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=2017)
print(f'train type={x_train.dtype}, '
      f'size={x_train.shape}, '
      f'density={x_train.nnz / x_train.shape[0] / x_train.shape[1]},'
      f'{sparse_memory_usage(x_train):6.3} Mb')
print(f'test type={x_test.dtype}, '
      f'size={x_test.shape}, '
      f'density={x_test.nnz / x_test.shape[0] / x_test.shape[1]},'
      f'{sparse_memory_usage(x_test):6.3} Mb')
del x_data
del y_data
gc.collect()


dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)
del x_train, x_test, y_train, y_test
gc.collect()


dtrain.save_binary('changes.matrix.train.train')
dtest.save_binary('changes.matrix.train.test')

dtrain = xgb.DMatrix('changes.matrix.train.train#changes.dtrain.cache')
dtest = xgb.DMatrix('changes.matrix.train.test#changes.dtest.cache')
watchlist = [(dtest, 'test'), (dtrain, 'train')]

param = {'objective': 'binary:logistic',
         'tree_method': 'hist',
         'learning_rate': 0.3,
         'num_boost_round': 500,
         'max_depth': 5,
         'silent': 1,
         'nthread': 4,
         'scale_pos_weight': 1 / np.mean(dtrain.get_label()) - 1,
         'eval_metric': ['error', 'auc'],
         'seed': '2017'}
model = xgb.train(param, dtrain, num_boost_round=param['num_boost_round'], evals=watchlist,
                  early_stopping_rounds=30, verbose_eval=1)

model.save_model('changes_model_train.bin')

plt.rcParams['font.size'] = 8
feat_imp = pd.Series(model.get_fscore()).sort_values(ascending=False)
# print(feat_imp)
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')
plt.tight_layout()
plt.savefig('changes_features_imp.png')
plt.show()
