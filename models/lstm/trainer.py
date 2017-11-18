from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split
from sparse_helpers import sparse_memory_usage
import numpy as np
import pandas as pd
import gc
from keras.preprocessing.text import text_to_word_sequence
from nltk import FreqDist
from models.lstm.models import AttentionModel
import csv
import os.path


def save_index(data, filename):
    with open(filename, 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_MINIMAL)
        for s in data:
            wr.writerow([s,])


def load_index(filename):
    ix = []
    with open(filename, 'r') as myfile:
        reader = csv.reader(myfile, quoting=csv.QUOTE_MINIMAL)
        for w in reader:
            ix.append(w[0])
    return ix


def words_list(indexes: np.ndarray, ix_to_word):
    preds = []
    for sent in indexes:
        preds.append(' '.join([ix_to_word[ix] for ix in sent if ix > 0]))
    return preds


def sparse_indexes(words_list: list, word_to_ix: dict, max_len):
    rows = []
    cols = []
    data = []
    for row, sentence in enumerate(words_list):
        for col, word in enumerate(sentence):
            if col < max_len:
                rows.append(row)
                cols.append(col)
                if word in word_to_ix:
                    data.append(word_to_ix[word])
                else:
                    data.append(word_to_ix['UNK'])
    return coo_matrix((data, (rows, cols)), shape=(len(words_list), max_len), dtype=np.uint32)


def prepare_matrix(series, max_len, vocab_len, index_file):
    data = [text_to_word_sequence(w) for w in series]
    if os.path.exists(index_file):
        ix_to_word = load_index(index_file)
    else:
        dist = FreqDist(np.hstack(data))
        print(f'{index_file} max_vocab_len', len(dist))

        vocab = dist.most_common(vocab_len - 1)
        print(f'{index_file} vocab', vocab)

        ix_to_word = [word[0] for word in vocab]
        ix_to_word.insert(0, 'ZERO')
        ix_to_word.append('UNK')
        save_index(ix_to_word, index_file)

    word_to_ix = {word: ix for ix, word in enumerate(ix_to_word)}
    matrix = sparse_indexes(data, word_to_ix, max_len).tocsr()
    return matrix, ix_to_word, word_to_ix


def train(model_name,
          df: pd.DataFrame,
          input_max_len, input_vocab_len, output_max_len, output_vocab_len,
          hidden_dim, layer_num, learning_rate, dropout, embedding_dim,
          epochs, mem_size, batch_size):

    X, X_ix_to_word, X_word_to_ix = prepare_matrix(df['before'],
                                                   input_max_len,
                                                   input_vocab_len,
                                                   f'{model_name}_input_index.csv')

    y, y_ix_to_word, y_word_to_ix = prepare_matrix(df['after'],
                                                   output_max_len,
                                                   output_vocab_len,
                                                   f'{model_name}_output_index.csv')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
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

    model = AttentionModel(model_name, input_max_len, len(X_ix_to_word), output_max_len, len(y_ix_to_word),
                           hidden_dim, layer_num, learning_rate, dropout, embedding_dim)

    model.train(X_train, y_train, X_test, y_test, epochs, mem_size, batch_size)

    y_predict = words_list(model.test(X_test), y_ix_to_word)

    X_str = words_list(X_test.toarray(), X_ix_to_word)
    y_str = words_list(y_test.toarray(), y_ix_to_word)
    result_df = pd.DataFrame(data={'before': X_str, 'actual': y_str, 'predict': y_predict})

    return result_df