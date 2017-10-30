from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent, Embedding
from keras.layers.recurrent import LSTM
import os
import numpy as np
import sys


BATCH_SIZE = 100
LAYER_NUM = 3
HIDDEN_DIM = 256
NB_EPOCH = 20


def create_model(X_vocab_len, X_max_len, y_vocab_len, y_max_len, hidden_size, num_layers):
    model = Sequential()

    # Creating encoder network
    model.add(Embedding(X_vocab_len, 128, input_length=X_max_len, mask_zero=True))
    model.add(LSTM(hidden_size))
    model.add(RepeatVector(y_max_len))

    # Creating decoder network
    for _ in range(num_layers):
        model.add(LSTM(hidden_size, return_sequences=True))
    model.add(TimeDistributed(Dense(y_vocab_len)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])
    return model


def find_checkpoint_file(folder):
    checkpoint_file = [f for f in os.listdir(folder) if 'checkpoint' in f]
    if len(checkpoint_file) == 0:
        return []
    modified_time = [os.path.getmtime(f) for f in checkpoint_file]
    return checkpoint_file[np.argmax(modified_time)]


def vectorize_data(words, char_to_ix):
    sequences = np.zeros((words.shape[0], words.shape[1], len(char_to_ix)))
    chars = words.tocoo()
    for r, c, d in zip(chars.row, chars.col, chars.data):
        sequences[r, c, char_to_ix[d]]
    return sequences


def train_model(X_train, X_char_to_index, x_max_len, y_train, y_char_to_index, y_max_len):
    print('[INFO] Compiling model...')
    model = create_model(len(X_char_to_index), x_max_len, len(y_char_to_index), y_max_len, HIDDEN_DIM, LAYER_NUM)

    saved_weights = find_checkpoint_file('.')

    k_start = 1
    if len(saved_weights) != 0:
        print('[INFO] Saved weights found, loading...')
        epoch = saved_weights[saved_weights.rfind('_') + 1:saved_weights.rfind('.')]
        model.load_weights(saved_weights)
        k_start = int(epoch) + 1

    for k in range(k_start, NB_EPOCH + 1):
        # Shuffling the training data every epoch to avoid local minima
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train = X_train[indices, :]
        y_train = y_train[indices, :]

        # Training 1000 sequences at a time
        for i in range(0, X_train.shape[0], 1000):
            i_end = min(i+1000, X_train.shape[0])
            y_sequences = vectorize_data(y_train[i:i_end, :], y_char_to_index)

            print(f'[INFO] Training model: epoch {k}th {i}/{X_train.shape[0]} samples')
            model.fit(X_train[i:i_end, :].toarray(), y_sequences, batch_size=BATCH_SIZE, epochs=1, verbose=2)
        model.save_weights('checkpoint_epoch_{}.hdf5'.format(k))


def test_model(X_test, X_index_to_char, x_max_len, y, y_index_to_char, y_char_to_index, y_max_len):
    print('[INFO] Compiling model...')
    model = create_model(len(X_index_to_char), x_max_len, len(y_index_to_char), y_max_len, HIDDEN_DIM, LAYER_NUM)

    saved_weights = find_checkpoint_file('.')

    if len(saved_weights) == 0:
        print("The network hasn't been trained! Program will exit...")
        sys.exit()
    else:
        model.load_weights(saved_weights)

        predictions = np.argmax(model.predict(X_test), axis=2)
        sequences = []
        # for prediction in predictions:
        #     sequence = ' '.join([y_ix_to_word(index) for index in prediction if index > 0])
        #     print(sequence)
        #     sequences.append(sequence)
        # np.savetxt('test_result', sequences, fmt='%s')
