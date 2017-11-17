from __future__ import print_function
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import sys
import argparse
from models.cardinal.seq2seq_utils import *

MAX_LEN = 16
INPUT_VOCAB_SIZE = 20000
OUTPUT_VOCAB_SIZE = 158
BATCH_SIZE = 32
LAYER_NUM = 3
HIDDEN_DIM = 1000
NB_EPOCH = 20


if __name__ == '__main__':
    # Loading input sequences, output sequences and the necessary mapping dictionaries
    print('[INFO] Loading data...')
    X, X_vocab_len, X_word_to_ix, X_ix_to_word, y, y_vocab_len, y_word_to_ix, y_ix_to_word = load_data(MAX_LEN, INPUT_VOCAB_SIZE, OUTPUT_VOCAB_SIZE)

    # Finding the length of the longest sequence
    X_max_len = max([len(sentence) for sentence in X])
    y_max_len = max([len(sentence) for sentence in y])
    print(X_max_len, y_max_len)

    # Padding zeros to make all sequences have a same length with the longest one
    print('[INFO] Zero padding...')
    X = pad_sequences(X, maxlen=X_max_len, dtype='int32')
    y = pad_sequences(y, maxlen=y_max_len, dtype='int32')

    # Creating the network model
    print('[INFO] Compiling model...')
    model = create_model(X_vocab_len, X_max_len, y_vocab_len, y_max_len, HIDDEN_DIM, LAYER_NUM)

    # Finding trained weights of previous epoch if any
    saved_weights = find_checkpoint_file('.')

    # Training only if we chose training mode
    k_start = 1

    # If any trained weight was found, then load them into the model
    if len(saved_weights) != 0:
        print('[INFO] Saved weights found, loading...')
        epoch = saved_weights[saved_weights.rfind('_') + 1:saved_weights.rfind('.')]
        model.load_weights(saved_weights)
        k_start = int(epoch) + 1

    i_end = 0
    for k in range(k_start, NB_EPOCH + 1):
        # Shuffling the training data every epoch to avoid local minima
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

        # Training 1000 sequences at a time
        for i in range(0, len(X), 1000):
            if i + 1000 >= len(X):
                i_end = len(X)
            else:
                i_end = i + 1000
            y_sequences = process_data(y[i:i_end], y_max_len, y_word_to_ix)

            print('[INFO] Training model: epoch {}th {}/{} samples'.format(k, i, len(X)))
            model.fit(X[i:i_end], y_sequences, batch_size=BATCH_SIZE, epochs=1, verbose=1)
        model.save_weights('checkpoint_epoch_{}.hdf5'.format(k))

    # Only performing test if there is any saved weights
    if len(saved_weights) == 0:
        print("The network hasn't been trained! Program will exit...")
        sys.exit()
    else:
        X_test = load_test_data('test', X_word_to_ix, MAX_LEN)
        X_test = pad_sequences(X_test, maxlen=X_max_len, dtype='int32')
        model.load_weights(saved_weights)

        predictions = np.argmax(model.predict(X_test), axis=2)
        sequences = []
        for prediction in predictions:
            sequence = ' '.join([y_ix_to_word(index) for index in prediction if index > 0])
            print(sequence)
            sequences.append(sequence)
        np.savetxt('test_result', sequences, fmt='%s')
