from sklearn.model_selection import train_test_split
from loaders.loading import load_train
from models.trans_helpers import train_model
from transformers.string_to_chars import StringToChar
from sparse_helpers import sparse_memory_usage
import gc
import numpy as np


df = load_train(['before', 'after'], r'../../input/norm_challenge_ru').fillna('')
df = df[~(df['before'] == df['after']) & (df['after'].str.contains('_trans'))]
df['after'] = df['after'].str.replace('_trans', '').str.replace(' ', '')
df['before'] = df['before'].str.lower()
print(df)
print(df.info())

max_len = 64#max(df['after'].str.len().max(), df['before'].str.len().max())
string_to_char = StringToChar(max_len, to_coo=True)

X_data = string_to_char.fit_transform(df['before']).tocsr()
print(f'x data type={X_data.dtype}, '
      f'size={X_data.shape}, '
      f'density={X_data.nnz / X_data.shape[0] / X_data.shape[1]},'
      f'{sparse_memory_usage(X_data):9.3} Mb')
y_data = string_to_char.fit_transform(df['after']).tocsr()
print(f'y data type={y_data.dtype}, '
      f'size={y_data.shape}, '
      f'density={y_data.nnz / y_data.shape[0] / y_data.shape[1]},'
      f'{sparse_memory_usage(y_data):9.3} Mb')
del df
gc.collect()

X_index_to_char = list(set(X_data.data))
X_char_to_index = {chr: ix for ix, chr in enumerate(X_index_to_char)}
y_index_to_char = [0]+list(set(y_data.data))
y_char_to_index = {chr: ix for ix, chr in enumerate(y_index_to_char)}

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=2017)
print(f'x train type={X_train.dtype}, '
      f'size={X_train.shape}, '
      f'density={X_train.nnz / X_train.shape[0] / X_train.shape[1]},'
      f'{sparse_memory_usage(X_train):9.3} Mb')
print(f'x test type={X_test.dtype}, '
      f'size={X_test.shape}, '
      f'density={X_test.nnz / X_test.shape[0] / X_test.shape[1]},'
      f'{sparse_memory_usage(X_test):9.3} Mb')
print(f'y train type={y_train.dtype}, '
      f'size={y_train.shape}, '
      f'density={y_train.nnz / y_train.shape[0] / y_train.shape[1]},'
      f'{sparse_memory_usage(y_train):9.3} Mb')
print(f'y test type={y_test.dtype}, '
      f'size={y_test.shape}, '
      f'density={y_test.nnz / y_test.shape[0] / y_test.shape[1]},'
      f'{sparse_memory_usage(y_test):9.3} Mb')
del X_data
del y_data
gc.collect()

train_model(X_train, X_char_to_index, max_len, y_train, y_char_to_index, max_len)
