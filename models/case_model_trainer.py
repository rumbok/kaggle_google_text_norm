import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from loaders.loading import load_train
from transformers.item_selector import ItemSelector
from transformers.morphology_extractor import MorphologyExtractor
from transformers.sparse_union import SparseUnion
from transformers.string_to_chars import StringToChar
from sparse_helpers import sparse_memory_usage
import gc
from sklearn.metrics import accuracy_score


df = load_train(['before', 'after', 'class']).fillna('')
df['prev_prev'] = df['before'].shift(2).fillna('')
df['prev'] = df['before'].shift(1).fillna('')
df['next'] = df['before'].shift(-1).fillna('')
df['next_next'] = df['before'].shift(-2).fillna('')
# df = df[~(df['before'] == df['after'])]
print(df.info())


morph_extractor = MorphologyExtractor(to_coo=True, multi_words=True)
pipeline = SparseUnion([
    ('class', Pipeline([
        ('select', ItemSelector('class')),
        ('label', LabelEncoder()),
        ('onehot', OneHotEncoder())
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


x_data = pipeline.fit_transform(df.drop(['class'], axis=1))
print(f'data type={x_data.dtype}, '
      f'size={x_data.shape}, '
      f'density={x_data.nnz / x_data.shape[0] / x_data.shape[1]},'
      f'{sparse_memory_usage(x_data):9.3} Mb')
y_data = pd.factorize(df['class'])
labels = y_data[1]
y_data = y_data[0]
del morph_extractor
del df
gc.collect()

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=2017)
print(f'train type={x_train.dtype}, '
      f'size={x_train.shape}, '
      f'density={x_train.nnz / x_train.shape[0] / x_train.shape[1]},'
      f'{sparse_memory_usage(x_train):9.3} Mb')
print(f'test type={x_test.dtype}, '
      f'size={x_test.shape}, '
      f'density={x_test.nnz / x_test.shape[0] / x_test.shape[1]},'
      f'{sparse_memory_usage(x_test):9.3} Mb')
del x_data
del y_data
gc.collect()


dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)
del x_train, x_test, y_train, y_test
gc.collect()


dtrain.save_binary('class.matrix.train.train')
dtest.save_binary('class.matrix.train.test')
del dtrain
del dtest
gc.collect()


dtrain = xgb.DMatrix('class.matrix.train.train#class.dtrain.cache')
dtest = xgb.DMatrix('class.matrix.train.test#class.dtest.cache')
watchlist = [(dtrain, 'train'), (dtest, 'test')]

param = {'objective': 'multi:softmax',
         'tree_method': 'hist',
         'learning_rate': 0.3,
         'num_boost_round': 1000,
         'max_depth': 5,
         'silent': 1,
         'nthread': 4,
         'njobs': 4,
         'num_class': len(set(dtrain.get_label())),
         'eval_metric': ['merror', 'mlogloss'],
         'seed': '2017'}
model = xgb.train(param, dtrain, num_boost_round=param['num_boost_round'], evals=watchlist,
                  early_stopping_rounds=50, verbose_eval=1)

y_pred = model.predict(dtest)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(dtest.get_label(), predictions)
print("Accuracy: %.5f%%" % accuracy)

model.save_model(f'class.model.train_{len(dtrain.get_label())}_{accuracy:0.5f}_{param["learning_rate"]}_{param["num_boost_round"]}_{param["max_depth"]}')

# plt.rcParams['font.size'] = 8
# feat_imp = pd.Series(model.get_fscore()).sort_values(ascending=False)
# feat_imp.plot(kind='bar', title='Feature Importances')
# plt.ylabel('Feature Importance Score')
# plt.tight_layout()
# plt.savefig('class_features_imp.png')
# plt.show()
