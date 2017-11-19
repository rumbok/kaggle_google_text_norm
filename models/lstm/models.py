from keras.models import Sequential
from keras.layers import TimeDistributed, Dense, Embedding
from keras.optimizers import RMSprop
from scipy.sparse import csr_matrix
import os
import numpy as np
import sys
from seq2seq.models import AttentionSeq2Seq


def find_checkpoint_file(folder, name):
    checkpoint_file = [f for f in os.listdir(folder) if f'{name}_epoch' in f]
    if len(checkpoint_file) == 0:
        return []
    modified_time = [os.path.getmtime(f) for f in checkpoint_file]
    return checkpoint_file[np.argmax(modified_time)]


class AttentionModel:
    def __init__(self,
                 model_name,
                 x_max_len, x_vocab_len,
                 y_max_len, y_vocab_len,
                 hidden_dim, layer_num, learning_rate, dropout,
                 embedding_dim=0):

        self.model_name = model_name

        self.X_max_len = x_max_len
        self.X_vocab_len = x_vocab_len

        self.y_max_len = y_max_len
        self.y_vocab_len = y_vocab_len

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.learning_rate = learning_rate
        self.dropout = dropout

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
                                            bidirectional=True,
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

        prev_acc = 0.0
        epoch = k_start
        while epoch <= epochs and self.model.optimizer.lr.read_value() > 0.0000001:
        # for epoch in range(k_start, epochs + 1):
            acc = self._train_epoch(batch_size, mem_size, test_data, x_train, y_test_array, y_train)
            print('Accuracy', acc)
            if acc >= prev_acc:
                self.model.save_weights(f'{self.model_name}_epoch_{epoch}_{acc}_{self.embedding_dim}_{self.hidden_dim}_{self.layer_num}_{self.dropout}_{self.learning_rate}.hdf5')
                epoch += 1
                prev_acc = acc
            else:
                saved_weights = find_checkpoint_file('.', self.model_name)
                self.model.load_weights(saved_weights)
                self.model.optimizer.lr.assign(self.model.optimizer.lr.read_value() * 0.1)

                # model.save(f'model_epoch_{epoch}_{acc}.hdf5')

    def _train_epoch(self, batch_size, mem_size, test_data, x_train, y_test_array, y_train):
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
                           batch_size=batch_size, epochs=1, verbose=2)
        y_pred_array = np.argmax(self.model.predict(test_data[0]), axis=2)
        acc = np.mean(np.all(y_pred_array == y_test_array, axis=1))
        return acc

    def test(self, X_test):
        if self.model is None:
            saved_weights = find_checkpoint_file('.', self.model_name)

            if len(saved_weights) == 0:
                print("The network hasn't been trained! Program will exit...")
                sys.exit()
            self.model.load_weights(saved_weights)

        return np.argmax(self.model.predict(self._vectorize(X_test, self.X_vocab_len), batch_size=1), axis=2)

    def load_weights(self, modelpath):
        self.model.load_weights(modelpath)
