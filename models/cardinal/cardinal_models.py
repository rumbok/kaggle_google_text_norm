from keras.models import Sequential, load_model
from keras.layers import TimeDistributed, Dense, RepeatVector, Embedding, ConvLSTM2D, Activation
from keras.optimizers import RMSprop, SGD
from keras.layers.recurrent import LSTM
from scipy.sparse import csr_matrix, coo_matrix
import os
import numpy as np
import sys
from seq2seq.models import AttentionSeq2Seq


def find_checkpoint_file(folder, name):
    checkpoint_file = [f for f in os.listdir(folder) if name in f]
    if len(checkpoint_file) == 0:
        return []
    modified_time = [os.path.getmtime(f) for f in checkpoint_file]
    return checkpoint_file[np.argmax(modified_time)]


def index(words: csr_matrix, word_to_ix):
    sequences = np.zeros((words.shape[0], words.shape[1]))
    for i, word in enumerate(words.toarray()):
        for j, ix in enumerate(word):
            sequences[i, j] = word_to_ix[ix]
    return sequences


def sparse_indexes(words_list: list, word_to_ix: dict, max_len):
    rows = []
    cols = []
    data = []
    for row, sentence in enumerate(words_list):
        for col, word in enumerate(sentence):
            rows.append(row)
            cols.append(col)
            if word in word_to_ix:
                data.append(word_to_ix[word])
            else:
                data.append(word_to_ix['UNK'])
    return coo_matrix((data, (rows, cols)), shape=(len(words_list), max_len), dtype=np.uint32)


def words_list(indexes: np.ndarray, ix_to_word):
    preds = []
    for sent in indexes:
        preds.append(' '.join([ix_to_word[ix] for ix in sent if ix > 0]))
    return preds


class AttentionModel:
    def __init__(self,
                 x_max_len, x_vocab_len,
                 y_max_len, y_vocab_len,
                 hidden_dim, layer_num, learning_rate, dropout,
                 embedding_dim=0):
        self.X_max_len = x_max_len
        self.X_vocab_len = x_vocab_len

        self.y_max_len = y_max_len
        self.y_vocab_len = y_vocab_len

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.learning_rate = learning_rate
        self.dropout = dropout

        self.model_name = 'cardinal'

        print('[INFO] Compiling model...')
        self.model = Sequential()
        if self.embedding_dim > 0:
            self.model.add(Embedding(input_length=self.X_max_len,
                                     input_dim=self.X_vocab_len,
                                     output_dim=embedding_dim,
                                     mask_zero=True))
            self.model.add(AttentionSeq2Seq(input_length=self.X_max_len,
                                            input_dim=embedding_dim,
                                            output_length=self.y_max_len,
                                            output_dim=self.y_vocab_len,
                                            hidden_dim=hidden_dim,
                                            bidirectional=False,
                                            depth=layer_num,
                                            dropout=dropout))

        else:
            self.model.add(AttentionSeq2Seq(input_length=self.X_max_len,
                                            input_dim=self.X_vocab_len,
                                            output_length=self.y_max_len,
                                            output_dim=self.y_vocab_len,
                                            hidden_dim=hidden_dim,
                                            bidirectional=False,
                                            depth=layer_num,
                                            dropout=dropout))

        self.model.add(TimeDistributed(Dense(units=self.y_vocab_len, activation='softmax')))
        opt = RMSprop(lr=learning_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    def _vectorize(self, indexes, vocab_len):
        if self.embedding_dim == 0:
            sequences = np.zeros((indexes.shape[0], indexes.shape[1], vocab_len))
            for i, sentence in enumerate(indexes.toarray()):
                for j, ix in enumerate(sentence):
                    sequences[i, j, ix] = 1.
            return sequences
        else:
            return indexes.toarray()

    def train(self,
              x_train: csr_matrix, y_train: csr_matrix,
              x_test: csr_matrix, y_test: csr_matrix,
              epochs, mem_size, batch_size):
        saved_weights = find_checkpoint_file('.', self.model_name)

        k_start = 1
        if len(saved_weights) != 0:
            print('[INFO] Saved weights found, loading...')
            epoch = saved_weights[
                    saved_weights.find('epoch_') + 6:saved_weights.find('_', saved_weights.find('epoch_') + 6)]
            self.model.load_weights(saved_weights)
            k_start = int(epoch) + 1

        test_data = (self._vectorize(x_test, self.X_vocab_len), self._vectorize(y_test, self.y_vocab_len))
        y_test_array = y_test.toarray()

        for epoch in range(k_start, epochs + 1):
            # Shuffling the training data every epoch to avoid local minima
            indices = np.arange(x_train.shape[0])
            np.random.shuffle(indices)
            x_train = x_train[indices, :]
            y_train = y_train[indices, :]

            for i in range(0, x_train.shape[0], mem_size):
                i_end = min(i + mem_size, x_train.shape[0])
                x_mem = self._vectorize(x_train[i:i_end, :], self.X_vocab_len)
                y_mem = self._vectorize(y_train[i:i_end, :], self.y_vocab_len)

                print(f'[INFO] Training model: epoch {epoch}th {i}/{x_train.shape[0]} samples')
                self.model.fit(x_mem, y_mem,
                               validation_data=test_data,
                               batch_size=batch_size, epochs=1, verbose=1)

            y_pred_array = np.argmax(self.model.predict(test_data[0]), axis=2)
            acc = np.mean(np.all(y_pred_array == y_test_array, axis=1))
            print('Accuracy', acc)
            self.model.save_weights(f'{self.model_name}_epoch_{epoch}_{acc}_{self.embedding_dim}_{self.hidden_dim}_{self.layer_num}_{self.dropout}_{self.learning_rate}.hdf5')
            # model.save(f'model_epoch_{epoch}_{acc}.hdf5')

    def test(self, X_test, model=None):
        if model is None:
            saved_weights = find_checkpoint_file('.', self.model_name)

            if len(saved_weights) == 0:
                print("The network hasn't been trained! Program will exit...")
                sys.exit()
            self.model.load_weights(saved_weights)

        return np.argmax(self.model.predict(self._vectorize(X_test, self.X_vocab_len)), axis=2)
