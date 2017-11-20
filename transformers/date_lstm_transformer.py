import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from models.lstm.models import AttentionModel
from models.lstm.trainer import prepare_matrix, words_list, load_index
import os.path

INPUT_MAX_LEN = 47
OUTPUT_MAX_LEN = 12


class DateLSTMTransformer(TransformerMixin, BaseEstimator):
    def __init__(self,
                 modelpath,
                 input_max_len=INPUT_MAX_LEN, output_max_len=OUTPUT_MAX_LEN):

        self.modelpath = modelpath

        meta = os.path.basename(self.modelpath).split('_')
        self.model_name = meta[0]
        self.x_max_len = input_max_len
        self.y_max_len = output_max_len
        self.x_ix_to_word = load_index(f'{self.model_name}_input_index.csv')
        self.y_ix_to_word = load_index(f'{self.model_name}_output_index.csv')
        self.embedding_dim = int(meta[4])
        self.hidden_dim = int(meta[5])
        self.layer_num = int(meta[6])
        self.learning_rate = float(meta[8][0:-5])
        self.dropout = float(meta[7])

        self.model = AttentionModel(self.model_name,
                                    self.x_max_len, len(self.x_ix_to_word), self.y_max_len, len(self.y_ix_to_word),
                                    self.hidden_dim, self.layer_num, self.learning_rate, self.dropout,
                                    self.embedding_dim)
        self.model.load_weights(self.modelpath)

    def fit(self, X: pd.DataFrame, y=None, *args, **kwargs):
        return self

    def transform(self, df: pd.DataFrame, y=None, *args, **kwargs):
        cardinal_ixs = df[df['class'] == 'DATE'].index

        x_series = (df.loc[cardinal_ixs, 'prev_prev'].map(str) + ' ' \
                    + df.loc[cardinal_ixs, 'prev'].map(str) + ' ' \
                    + df.loc[cardinal_ixs, 'before'].map(lambda s: ' '.join(list(s))) + ' ' \
                    + df.loc[cardinal_ixs, 'next'].map(str) + ' ' \
                    + df.loc[cardinal_ixs, 'next_next'].map(str)).str.lower()

        x, _, _ = prepare_matrix(x_series,
                                 self.x_max_len,
                                 len(self.x_ix_to_word),
                                 f'{self.model_name}_input_index.csv')
        del x_series

        y_predict = words_list(self.model.test(x), self.y_ix_to_word)
        del x

        if 'after' in df.columns:
            return df.assign(after=df['after'].combine_first(pd.Series(y_predict, index=cardinal_ixs)))
        else:
            return df.assign(after=pd.Series(y_predict, index=cardinal_ixs, name='after'))


if __name__ == '__main__':
    df = pd.DataFrame({'before': '123 в 1960 году 56 пехотинцев'.split(),
                       'class': ['CARDINAL', 'PLAIN', 'DATE', 'PLAIN', 'CARDINAL', 'PLAIN']})
    df['prev_prev'] = df['before'].shift(2)
    df['prev'] = df['before'].shift(1)
    df['next'] = df['before'].shift(-1)
    df['next_next'] = df['before'].shift(-2)
    df = df.fillna('')
    print(df)

    ct = DateLSTMTransformer(modelpath='date_epoch_17_0.9237390719569604_0_64_2_0.0_1.0000000000000002e-07.hdf5')

    print(ct.fit_transform(df))
