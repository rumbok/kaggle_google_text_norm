from sklearn.model_selection import train_test_split
from loaders.loading import load_train, load_external
from sparse_helpers import sparse_memory_usage
import numpy as np
import pandas as pd
import gc
from keras.preprocessing.text import text_to_word_sequence
from nltk import FreqDist
from models.cardinal.cardinal_models import sparse_indexes, AttentionModel, words_list


CARDINAL_REGEXP = '\d'
INPUT_MAX_LEN = 33
OUTPUT_MAX_LEN = 23
INPUT_VOCAB_SIZE = 5000
OUTPUT_VOCAB_SIZE = 158

LAYER_NUM = 1
HIDDEN_DIM = 64
EMBEDDING_DIM = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MEM_SIZE = 1000
NB_EPOCH = 3
DROPOUT = 0.0

df = load_train(['before', 'after', 'class'], input_path=r'../input/norm_challenge_ru').fillna('')
# df = load_external(['before', 'after'],
#                   only_diff=True,
#                   input_path=r'../input/norm_challenge_ru/ru_with_types')\
#      .fillna('')
df['prev_prev'] = df['before'].shift(2)
df['prev'] = df['before'].shift(1)
df['next'] = df['before'].shift(-1)
df['next_next'] = df['before'].shift(-2)
df = df[~(df['before'] == df['after']) & (df['before'].str.contains(CARDINAL_REGEXP))].fillna('')
df['before'] = df['prev_prev'].map(str) + ' '\
               + df['prev'].map(str) + ' '\
               + df['before'].map(lambda s: ' '.join(list(s))) + ' ' \
               + df['next'].map(str) + ' ' \
               + df['next_next'].map(str)
del df['prev_prev'], df['prev'], df['next'], df['next_next'],
df['before'] = df['before'].str.lower()
df = df[df['class'] == 'CARDINAL']
# df = df.sample(10000)
print(df.info())

X_data = [text_to_word_sequence(w) for w in df['before']]
y_data = [text_to_word_sequence(w) for w in df['after']]
del df

dist = FreqDist(np.hstack(X_data))
print('X_max_vocab_len', len(dist))
X_vocab = dist.most_common(INPUT_VOCAB_SIZE-1)
print('X_vocab', X_vocab)
dist = FreqDist(np.hstack(y_data))
print('y_max_vocab_len', len(dist))
y_vocab = dist.most_common(OUTPUT_VOCAB_SIZE-1)
print('y_vocab', y_vocab)

X_ix_to_word = [word[0] for word in X_vocab]
X_ix_to_word.insert(0, 'ZERO')
X_ix_to_word.append('UNK')
X_word_to_ix = {word: ix for ix, word in enumerate(X_ix_to_word)}
print('X_vocab_len', len(X_ix_to_word))

y_ix_to_word = [word[0] for word in y_vocab]
y_ix_to_word.insert(0, 'ZERO')
y_ix_to_word.append('UNK')
y_word_to_ix = {word: ix for ix, word in enumerate(y_ix_to_word)}
print('y_vocab_len', len(y_ix_to_word))

X = sparse_indexes(X_data, X_word_to_ix, INPUT_MAX_LEN).tocsr()
y = sparse_indexes(y_data, y_word_to_ix, OUTPUT_MAX_LEN).tocsr()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01)
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
del X, y
gc.collect()

X_str = words_list(X_test.toarray(), X_ix_to_word)
y_str = words_list(y_test.toarray(), y_ix_to_word)

model = AttentionModel(INPUT_MAX_LEN, len(X_ix_to_word), OUTPUT_MAX_LEN, len(y_ix_to_word), HIDDEN_DIM, LAYER_NUM, LEARNING_RATE, DROPOUT, EMBEDDING_DIM)

model.train(X_train, y_train, X_test, y_test, NB_EPOCH, MEM_SIZE, BATCH_SIZE)
y_predict = words_list(model.test(X_test), y_ix_to_word)

result_df = pd.DataFrame(data={'before': X_str, 'actual': y_str, 'predict': y_predict})

print(result_df[~(result_df['actual'] == result_df['predict'])])