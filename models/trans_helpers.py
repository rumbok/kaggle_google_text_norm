from keras.models import Sequential, load_model
from keras.layers import TimeDistributed, Dense, RepeatVector, Embedding
from keras.optimizers import RMSprop
from keras.layers.recurrent import LSTM
from scipy.sparse import csr_matrix
import os
import numpy as np
import sys
import seq2seq
from seq2seq.models import AttentionSeq2Seq


LAYER_NUM = 2
HIDDEN_DIM = 64
EMBEDDING_DIM = 32
BATCH_SIZE = 32
MEM_SIZE = 10000
NB_EPOCH = 25


def create_attention_model(X_vocab_len, X_max_len, y_vocab_len, y_max_len, embedding_dim, hidden_dim, layer_num):
    model = Sequential()

    model.add(AttentionSeq2Seq(output_dim=y_vocab_len, hidden_dim=hidden_dim,
                               output_length=y_max_len,
                               input_shape=(X_max_len, X_vocab_len),
                               bidirectional=False, depth=layer_num))

    # model.add(Dense(y_vocab_len, activation='softmax'))
    model.add(TimeDistributed(Dense(y_vocab_len, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.03), metrics=['accuracy'])

    return model


def create_model(X_vocab_len, X_max_len, y_vocab_len, y_max_len, embedding_dim, hidden_dim, layer_num):
    model = Sequential()

    # Creating encoder network
    # model.add(Embedding(X_vocab_len, embedding_dim, input_length=X_max_len, mask_zero=True))
    model.add(LSTM(hidden_dim, input_shape=(X_max_len, X_vocab_len)))
    model.add(RepeatVector(y_max_len))

    # Creating decoder network
    for _ in range(layer_num):
        model.add(LSTM(hidden_dim, return_sequences=True))
    model.add(TimeDistributed(Dense(y_vocab_len, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


def find_checkpoint_file(folder):
    checkpoint_file = [f for f in os.listdir(folder) if 'checkpoint' in f]
    if len(checkpoint_file) == 0:
        return []
    modified_time = [os.path.getmtime(f) for f in checkpoint_file]
    return checkpoint_file[np.argmax(modified_time)]


def vectorize_data(words: csr_matrix, char_to_ix):
    sequences = np.zeros((words.shape[0], words.shape[1], len(char_to_ix)))
    for i, word in enumerate(words.toarray()):
        for j, c in enumerate(word):
            sequences[i, j, char_to_ix[c]] = 1.
    return sequences


def train_model(X_train, X_char_to_ix, y_train, y_char_to_ix, X_test, y_test):
    print('[INFO] Compiling model...')
    model = create_attention_model(len(X_char_to_ix), X_train.shape[1], len(y_char_to_ix), y_train.shape[1], EMBEDDING_DIM, HIDDEN_DIM, LAYER_NUM)

    saved_weights = find_checkpoint_file('.')

    k_start = 1
    if len(saved_weights) != 0:
        print('[INFO] Saved weights found, loading...')
        epoch = saved_weights[saved_weights.rfind('_') + 1:saved_weights.rfind('.')]
        model.load_weights(saved_weights)
        k_start = int(epoch) + 1

    valid_data = (vectorize_data(X_test, X_char_to_ix), vectorize_data(y_test, y_char_to_ix))

    for k in range(k_start, NB_EPOCH + 1):
        # Shuffling the training data every epoch to avoid local minima
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train = X_train[indices, :]
        y_train = y_train[indices, :]

        # Training MEM_SIZE sequences at a time
        for i in range(0, X_train.shape[0], MEM_SIZE):
            i_end = min(i+MEM_SIZE, X_train.shape[0])
            X_sequences = vectorize_data(X_train[i:i_end, :], X_char_to_ix)
            y_sequences = vectorize_data(y_train[i:i_end, :], y_char_to_ix)

            print(f'[INFO] Training model: epoch {k}th {i}/{X_train.shape[0]} samples')
            model.fit(X_sequences, y_sequences, batch_size=BATCH_SIZE, validation_data=valid_data, epochs=1, verbose=2)
        model.save_weights('checkpoint_epoch_{}.hdf5'.format(k))
        model.save('model_epoch_{}.hdf5'.format(k))


def test_model(X_test, X_char_to_ix, y_char_to_ix, y_ix_to_char, y_max_len):
    print('[INFO] Compiling model...')
    model = create_attention_model(len(X_char_to_ix), X_test.shape[1], len(y_char_to_ix), y_max_len, EMBEDDING_DIM, HIDDEN_DIM, LAYER_NUM)

    saved_weights = find_checkpoint_file('.')

    if len(saved_weights) == 0:
        print("The network hasn't been trained! Program will exit...")
        sys.exit()
    else:
        model.load_weights(saved_weights)

        predictions = np.argmax(model.predict(vectorize_data(X_test, X_char_to_ix)), axis=2)
        sequences = []
        for prediction in predictions:
            sequences.append([y_ix_to_char[ix] for ix in prediction])
        return sequences
