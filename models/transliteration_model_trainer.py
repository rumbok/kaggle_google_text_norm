from sklearn.model_selection import train_test_split
from loaders.loading import load_train, load_external
from models.trans_helpers import train_model, test_model, int_to_str
from transformers.string_to_chars import StringToChar
from sparse_helpers import sparse_memory_usage
import numpy as np
import pandas as pd
import gc


df = load_train(['before', 'after'], input_path=r'../input/norm_challenge_ru').fillna('')
#df = load_external(['before', 'after'],
#                   only_diff=True,
#                   input_path=r'../input/norm_challenge_ru/ru_with_types')\
#      .fillna('')
df = df[~(df['before'] == df['after']) & (df['after'].str.contains('_trans'))]
df['after'] = df['after'].str.replace('_trans', '').str.replace(' ', '')
df['before'] = df['before'].str.lower()
print('drop {0} urls from strings'.format(len(df[df['before'].str.contains('\.')].index)))
df = df[~df['before'].str.contains('\.')]
print(df.info())

X_max_len = 32#min(32, df['before'].str.len().max())
y_max_len = 32#min(32, df['after'].str.len().max())

X_data = StringToChar(X_max_len, to_coo=True).fit_transform(df['before']).tocsr()
y_data = StringToChar(y_max_len, to_coo=True).fit_transform(df['after']).tocsr()
del df
gc.collect()

X_ix_to_char = [0] + sorted(set(X_data.data))
X_char_to_ix = {chr: ix for ix, chr in enumerate(X_ix_to_char)}
y_ix_to_char = [0] + sorted(set(y_data.data))
y_char_to_ix = {chr: ix for ix, chr in enumerate(y_ix_to_char)}
print(X_ix_to_char)
print(y_ix_to_char)


X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=2017)
print(f'x train type={X_train.dtype}, '
      f'size={X_train.shape}, '
      f'density={X_train.nnz / X_train.shape[0] / X_train.shape[1]},'
      f'{sparse_memory_usage(X_train):9.3} Mb')
print(f'y train type={y_train.dtype}, '
      f'size={y_train.shape}, '
      f'density={y_train.nnz / y_train.shape[0] / y_train.shape[1]},'
      f'{sparse_memory_usage(y_train):9.3} Mb')
print(f'x test type={X_test.dtype}, '
      f'size={X_test.shape}, '
      f'density={X_test.nnz / X_test.shape[0] / X_test.shape[1]},'
      f'{sparse_memory_usage(X_test):9.3} Mb')
print(f'y test type={y_test.dtype}, '
      f'size={y_test.shape}, '
      f'density={y_test.nnz / y_test.shape[0] / y_test.shape[1]},'
      f'{sparse_memory_usage(y_test):9.3} Mb')
del X_data
del y_data
gc.collect()


train_model(X_train, X_char_to_ix, y_train, y_char_to_ix, y_ix_to_char, X_test, y_test)

X_str = int_to_str(X_test.toarray())
y_str = int_to_str(y_test.toarray())
pred = np.array(test_model(X_test, X_char_to_ix, y_char_to_ix, y_ix_to_char, y_max_len))
y_predict = int_to_str(pred)

result_df = pd.DataFrame(data={'before': X_str, 'actual': y_str, 'predict': y_predict})

print(result_df[~(result_df['actual'] == result_df['predict'])])

# original
# max_len = 32
# LAYER_NUM = 2
# HIDDEN_DIM = 64
# EMBEDDING_DIM = 0
# BATCH_SIZE = 32
# MEM_SIZE = 10000
# 1 - 45696/51754=0.8829 err
# 5 - 29717/51754=0.5742
# 8 - 23277/51754=0.4498


# without url
# max_len = 32
# LAYER_NUM = 2
# HIDDEN_DIM = 64
# EMBEDDING_DIM = 0
# BATCH_SIZE = 32
# MEM_SIZE = 10000
# 1 - 43927/51221=0.8576 err
# 2 - 41312/51221=0.8065


# attention without url
# max_len = 32
# LAYER_NUM = 4
# HIDDEN_DIM = 64
# EMBEDDING_DIM = 0
# BATCH_SIZE = 32
# MEM_SIZE = 10000
# NB_EPOCH = 3
# 3 - 9998 err


# without url
# max_len = 32
# LAYER_NUM = 2
# HIDDEN_DIM = 64
# EMBEDDING_DIM = 32
# BATCH_SIZE = 32
# LEARNING_RATE = 0.01
# MEM_SIZE = 10000
# NB_EPOCH = 3
# 3 - 9881