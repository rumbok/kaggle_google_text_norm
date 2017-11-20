import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from models.lstm.models import AttentionModel
from models.lstm.trainer import prepare_matrix, words_list, load_index
import os.path

INPUT_MAX_LEN = 32
OUTPUT_MAX_LEN = 32
ENG_REGEXP = '^[a-zA-Z]+$'


class TransLSTMTransformer(TransformerMixin, BaseEstimator):
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
        trans_ixs = df[(df['before'].str.match(ENG_REGEXP)) & (df['class'] == 'TRANS')].index

        x_series = df['before'].str.lower().map(lambda s: ' '.join(list(s)))

        x, _, _ = prepare_matrix(x_series,
                                 self.x_max_len,
                                 len(self.x_ix_to_word),
                                 f'{self.model_name}_input_index.csv')

        y_predict = words_list(self.model.test(x), self.y_ix_to_word)
        translit = [' '.join([c + '_trans' for c in str]) for str in
                    tqdm(strs_predict, f'{self.__class__.__name__} transform stage 2')]

        if 'after' in df.columns:
            return df.assign(after=df['after'].combine_first(pd.Series(y_predict, index=trans_ixs)))
        else:
            return df.assign(after=pd.Series(y_predict, index=trans_ixs, name='after'))


if __name__ == '__main__':
    df = pd.DataFrame({'before': '123 в 1960 году 56 пехотинцев'.split(),
                       'class': ['CARDINAL', 'PLAIN', 'DATE', 'PLAIN', 'CARDINAL', 'PLAIN']})
    df['prev_prev'] = df['before'].shift(2)
    df['prev'] = df['before'].shift(1)
    df['next'] = df['before'].shift(-1)
    df['next_next'] = df['before'].shift(-2)
    df = df.fillna('')
    print(df)

    ct = TransLSTMTransformer('measure_epoch_18_0.6424546023794615_0_64_2_0.01_1.0000000000000002e-07.hdf5')

    print(ct.fit_transform(df))
