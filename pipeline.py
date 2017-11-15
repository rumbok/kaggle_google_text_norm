import numpy as np

from transformers.cardinal_transformer import CardinalTransformer
from transformers.dash_transformer import DashTransformer
from transformers.dict_nbhd_transformer import DictNBHDTransformer
from transformers.dict_transformer import DictTransformer
from transformers.digit_transformer import DigitTransformer
from transformers.flat_transformer import FlatTransformer
from transformers.dict_class_transformer import DictClassTransformer
from transformers.latin_transliterator import LatinTransliterator
from transformers.letters_transformer import LettersTransformer
from transformers.lstm_transliterator import LSTMTransliterator
from transformers.rome_transformer import RomeTransformer
from transformers.scored_chain import ScoredChain
from transformers.self_transformer import SelfTransformer
from time import gmtime, strftime
from transformers.add_class_transformer import AddClassTransformer


transform_chain = ScoredChain([
        ('class', AddClassTransformer('models/class.model.train_1190428_0.00101_0.3_500_6')),
        ('digit', DigitTransformer()),
        ('dash', DashTransformer()),
        ('punсt', DictClassTransformer(u'PUNCT', 1.0)),
        ('verbatim', DictClassTransformer(u'VERBATIM', 1.0)),
        ('latin', LatinTransliterator()),
        ('self', SelfTransformer(threshold=0.5, modelpath='models/self.model.train_9517064_0.00117_0.3_500_6')),
        ('rome', RomeTransformer()),
        ('dict_nbhd', DictNBHDTransformer(0.5)),
        ('date', DictClassTransformer(u'DATE', 1.0)),
        ('cardinal_dict', DictClassTransformer(u'CARDINAL', 1.0)),
        ('cardinal', CardinalTransformer()),
        ('letters', LettersTransformer()),
        ('dict', DictTransformer(0.5)),
        ('translit', LSTMTransliterator('check point_epoch_36_0.8127_64_2_0.0.hdf5')),
        ('flat', FlatTransformer())
    ])


def transform(x_train, x_test, y_train, y_test=None):
    if y_test is not None:
        transform_chain.metrics=[
            ('tp', lambda X: np.mean((X["after"] == y_test)) if 'after' in X.columns else 0.0),
            ('fp', lambda X: np.mean((X["after"].notnull()) & ~(X["after"] == y_test)) if 'after' in X.columns else 0.0)
        ]

    transform_chain.fit(x_train, y_train)
    predicted = transform_chain.transform(x_test)

    if y_test is not None:
        with open('pipeline_results', 'a+') as f:
            f.write(strftime("%Y-%m-%d %H:%M:%S\n", gmtime()))
            f.write(f'test data {len(y_test)}\n')
            for step in transform_chain.steps:
                f.write(f'{step[0]}: {step[1].get_params()}\n')
            f.write('Chain:\n')
            for step_result in transform_chain.results:
                f.write(f'  {step_result}\n')
            f.write('\n')
    return predicted
