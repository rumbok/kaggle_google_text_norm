import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from transformers.string_to_chars import StringToChar
from models.trans_helpers import train_model, test_model, create_attention_model, int_to_str, vectorize_data
from tqdm import tqdm
from numpy.core.defchararray import add


LAYER_NUM = 2
HIDDEN_DIM = 64
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MEM_SIZE = 10000
NB_EPOCH = 3
DROPOUT = 0.0

X_MAX_LEN = 32
Y_MAX_LEN = 32

ENG_CHARS = [0, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117,
             118, 119, 120, 121, 122]
ENG_INDEXES = {chr: ix for ix, chr in enumerate(ENG_CHARS)}
RUS_CHARS = [0, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088,
             1089, 1090, 1091, 1092, 1093, 1094, 1095, 1096, 1098, 1099, 1100, 1101, 1102, 1103]
RUS_INDEXES = {chr: ix for ix, chr in enumerate(RUS_CHARS)}
ENG_REGEXP = '^[a-zA-Z]+$'



class LSTMTransliterator(TransformerMixin, BaseEstimator):
    def __init__(self,
                 modelpath='',
                 layer_num=LAYER_NUM, hidden_dim=HIDDEN_DIM, batch_size=BATCH_SIZE,
                 learning_rate=LEARNING_RATE, mem_size=MEM_SIZE, num_epochs=NB_EPOCH, dropout=DROPOUT):
        self.dropout = dropout
        self.modelpath = modelpath
        self.num_epochs = num_epochs
        self.mem_size = mem_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num

    def fit(self, X: pd.DataFrame, y=None, *args, **kwargs):
        self.model = create_attention_model(len(ENG_INDEXES), X_MAX_LEN, len(RUS_INDEXES), Y_MAX_LEN,
                                            self.hidden_dim, self.layer_num, self.learning_rate, self.dropout)

        if self.modelpath:
            self.model.load_weights(self.modelpath)

        return self

    def transform(self, X: pd.DataFrame, y=None, *args, **kwargs):
        trans_ixs = X[(X['before'].str.match(ENG_REGEXP)) & (X['class'] == 'TRANS')].index
        X_data = StringToChar(X_MAX_LEN, to_coo=True).fit_transform(X.loc[trans_ixs, 'before'].str.lower()).tocsr()

        predictions = np.argmax(self.model.predict(vectorize_data(X_data, ENG_INDEXES)), axis=2)
        sequences = []
        for prediction in tqdm(predictions, f'{self.__class__.__name__} transform stage 1'):
            sequences.append([RUS_CHARS[ix] for ix in prediction])
        strs_predict = int_to_str(np.array(sequences))
        translit = [' '.join([c + '_trans' for c in str]) for str in tqdm(strs_predict, f'{self.__class__.__name__} transform stage 2')]

        if 'after' in X.columns:
            return X.assign(after=X['after'].combine_first(pd.Series(translit, index=trans_ixs)))
        else:
            return X.assign(after=pd.Series(translit, index=trans_ixs, name='after'))


if __name__ == '__main__':
    df = pd.SparseDataFrame({'before': 'Smells like teen spirit вот ya.ru'.split(),
                             'class': ['TRANS', 'TRANS', 'TRANS', 'TRANS', 'PLAIN', 'TRANS']})
    print(df)

    ct = LSTMTransliterator(modelpath='../checkpoint_epoch_30_0.8001_64_2_0.0.hdf5')

    print(ct.fit_transform(df))
