from sklearn.model_selection import train_test_split
from loaders.loading import load_train
from models.trans_helpers import train_model, test_model
from transformers.string_to_chars import StringToChar
from sparse_helpers import sparse_memory_usage
import numpy as np
import pandas as pd
import gc


df = load_train(['before', 'after'], r'../../input/norm_challenge_ru').fillna('')
df = df[~(df['before'] == df['after']) & (df['after'].str.contains('_trans'))]
df['after'] = df['after'].str.replace('_trans', '').str.replace(' ', '')
df['before'] = df['before'].str.lower()
df = df.sample(100000)
print(df.head())
print(df.info())

max_len = 32#max(df['after'].str.len().max(), df['before'].str.len().max())

X_data = StringToChar(max_len, to_coo=True).fit_transform(df['before']).tocsr()
y_data = StringToChar(max_len, to_coo=True).fit_transform(df['after']).tocsr()
print(f'x data type={X_data.dtype}, '
      f'size={X_data.shape}, '
      f'density={X_data.nnz / X_data.shape[0] / X_data.shape[1]},'
      f'{sparse_memory_usage(X_data):9.3} Mb')
print(f'y data type={y_data.dtype}, '
      f'size={y_data.shape}, '
      f'density={y_data.nnz / y_data.shape[0] / y_data.shape[1]},'
      f'{sparse_memory_usage(y_data):9.3} Mb')
del df
gc.collect()

X_ix_to_char = [0] + sorted(set(X_data.data))
X_char_to_ix = {chr: ix for ix, chr in enumerate(X_ix_to_char)}
y_ix_to_char = [0] + sorted(set(y_data.data))
y_char_to_ix = {chr: ix for ix, chr in enumerate(y_ix_to_char)}

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


train_model(X_train, X_char_to_ix, y_train, y_char_to_ix, X_test, y_test)

X_str = X_test.toarray().astype(np.uint32).view('U1').view(f'U{max_len}').ravel()
y_str = y_test.toarray().astype(np.uint32).view('U1').view(f'U{max_len}').ravel()
pred = np.array(test_model(X_test, X_char_to_ix, y_char_to_ix, y_ix_to_char, max_len))
print(pred)
y_predict = pred.astype(np.uint32).view('U1').view(f'U{max_len}').ravel()

result_df = pd.DataFrame(data={'before': X_str, 'actual': y_str, 'predict': y_predict})

print(result_df)
print(result_df[~(result_df['actual'] == result_df['predict'])])