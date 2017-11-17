from keras.models import Sequential, load_model
from keras.layers import TimeDistributed, Dense, RepeatVector, Embedding, ConvLSTM2D, Activation
from keras.optimizers import RMSprop, SGD
from keras.layers.recurrent import LSTM
from scipy.sparse import csr_matrix
import os
import numpy as np
import sys
from seq2seq.models import AttentionSeq2Seq, Seq2Seq


LAYER_NUM = 1
HIDDEN_DIM = 64
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MEM_SIZE = 1000
NB_EPOCH = 100
DROPOUT = 0.01


def create_attention_model(X_vocab_len, X_max_len, y_vocab_len, y_max_len, hidden_dim, layer_num, learning_rate,
                           dropout):
    model = Sequential()

    model.add(AttentionSeq2Seq(output_dim=y_vocab_len,
                               hidden_dim=hidden_dim,
                               output_length=y_max_len,
                               input_shape=(X_max_len, X_vocab_len),
                               bidirectional=False,
                               depth=layer_num,
                               dropout=dropout))
    model.add(TimeDistributed(Dense(y_vocab_len, activation='softmax')))
    opt = RMSprop(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def create_model(X_vocab_len, X_max_len, y_vocab_len, y_max_len, hidden_dim, layer_num, learning_rate, dropout):
    model = Sequential()

    model.add(Seq2Seq(output_dim=y_vocab_len,
                      hidden_dim=hidden_dim,
                      output_length=y_max_len,
                      input_shape=(X_max_len, X_vocab_len),
                      depth=layer_num,
                      dropout=dropout))
    model.add(TimeDistributed(Dense(y_vocab_len, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=learning_rate), metrics=['accuracy'])
    return model


def create_model_with_emb(X_vocab_len, X_max_len, y_vocab_len, y_max_len, hidden_dim, layer_num, learning_rate,
                          dropout):
    model = Sequential()
    # Creating encoder network
    model.add(Embedding(X_vocab_len, 100, input_length=X_max_len, mask_zero=True))
    model.add(LSTM(hidden_dim, dropout=dropout, recurrent_dropout=dropout))
    model.add(RepeatVector(y_max_len))

    # Creating decoder network
    for _ in range(layer_num):
        model.add(LSTM(hidden_dim, return_sequences=True, dropout=dropout, recurrent_dropout=dropout))
    model.add(TimeDistributed(Dense(y_vocab_len)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(lr=learning_rate),
                  metrics=['accuracy'])
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
        for j, ix in enumerate(word):
            sequences[i, j, ix] = 1.
    return sequences


def train_model(X_train, X_char_to_ix, y_train, y_char_to_ix, y_ix_to_char, X_test, y_test):
    print('[INFO] Compiling model...')
    model = create_attention_model(len(X_char_to_ix), X_train.shape[1], len(y_char_to_ix), y_train.shape[1],
                                   HIDDEN_DIM, LAYER_NUM, LEARNING_RATE, DROPOUT)

    saved_weights = find_checkpoint_file('.')

    k_start = 1
    if len(saved_weights) != 0:
        print('[INFO] Saved weights found, loading...')
        epoch = saved_weights[
                saved_weights.find('epoch_') + 6:saved_weights.find('_', saved_weights.find('epoch_') + 6)]
        model.load_weights(saved_weights)
        k_start = int(epoch) + 1

    test_data = (vectorize_data(X_test, X_char_to_ix), vectorize_data(y_test, y_char_to_ix))
    preds = []
    for prediction in y_test.toarray():
        preds.append(' '.join([y_ix_to_char[ix] for ix in prediction]))
    y_test_words = np.array(preds)

    for epoch in range(k_start, NB_EPOCH + 1):
        # Shuffling the training data every epoch to avoid local minima
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train = X_train[indices, :]
        y_train = y_train[indices, :]

        # Training MEM_SIZE sequences at a time
        for i in range(0, X_train.shape[0], MEM_SIZE):
            i_end = min(i + MEM_SIZE, X_train.shape[0])
            X_sequences = vectorize_data(X_train[i:i_end, :], X_char_to_ix)
            y_sequences = vectorize_data(y_train[i:i_end, :], y_char_to_ix)

            print(f'[INFO] Training model: epoch {epoch}th {i}/{X_train.shape[0]} samples')
            model.fit(X_sequences, y_sequences, batch_size=BATCH_SIZE, validation_data=test_data, epochs=1, verbose=1)

        predictions = np.argmax(model.predict(test_data[0]), axis=2)
        preds = []
        for prediction in predictions:
            preds.append(' '.join([y_ix_to_char[ix] for ix in prediction]))
        pred_words = np.array(preds)

        acc = np.mean(pred_words == y_test_words)
        print('Accuracy', acc)

        model.save_weights(f'checkpoint_epoch_{epoch}_{acc}_{HIDDEN_DIM}_{LAYER_NUM}_{DROPOUT}.hdf5')
        # model.save(f'model_epoch_{epoch}_{acc}.hdf5')
    return model


def test_model(X_test, X_char_to_ix, y_char_to_ix, y_ix_to_char, y_max_len, model=None):
    print('[INFO] Compiling model...')
    if model is None:
        model = create_attention_model(len(X_char_to_ix), X_test.shape[1], len(y_char_to_ix), y_max_len,
                                       HIDDEN_DIM, LAYER_NUM, LEARNING_RATE, 0.0)
        saved_weights = find_checkpoint_file('.')

        if len(saved_weights) == 0:
            print("The network hasn't been trained! Program will exit...")
            sys.exit()
        model.load_weights(saved_weights)

    predictions = np.argmax(model.predict(vectorize_data(X_test, X_char_to_ix)), axis=2)
    preds = []
    for prediction in predictions:
        preds.append(' '.join([y_ix_to_char[ix] for ix in prediction]))
    return preds
