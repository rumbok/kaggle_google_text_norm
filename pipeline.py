import numpy as np

from transformers.add_case_transformer import AddCaseTransformer
from transformers.add_number_transformer import AddNumberTransformer
from transformers.cardinal_lstm_transformer import CardinalLSTMTransformer
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
        # # ('case', AddCaseTransformer('case.model.train_502554_0.02781_0.3_500_7')),
        # # ('number', AddNumberTransformer('number.model.train_502554_0.00849_0.3_500_6')),
        # ('digit', DigitTransformer()),
        # ('trans', DictClassTransformer(u'TRANS', 0.5)),
        # ('dash', DashTransformer()),
        # ('punсt', DictClassTransformer(u'PUNCT', 1.0)),
        # ('verbatim', DictClassTransformer(u'VERBATIM', 1.0)),
        # ('latin', LatinTransliterator()),
        ('self', SelfTransformer(threshold=0.5, modelpath='models/self.model.train_9517064_0.00117_0.3_500_6')),
        # ('rome', RomeTransformer()),
        # ('dict_nbhd', DictNBHDTransformer(0.5)),
        # ('date', DictClassTransformer(u'DATE', 1.0)),
        # ('cardinal_dict', DictClassTransformer(u'CARDINAL', 1.0)),
        ('cardinal', CardinalLSTMTransformer('cardinal_epoch_8_0.8895194854332198_0_64_2_0.0_0.0001.hdf5')),
        # ('cardinal', CardinalTransformer(use_case=False, use_number=False)),
        # ('letters', LettersTransformer(by_class=False)),
        # ('dict', DictTransformer(0.5)),
        # ('translit', LSTMTransliterator('check point_epoch_36_0.8127_64_2_0.0.hdf5')),
        # ('letters', LettersTransformer(by_class=True)),
        # ('flat', FlatTransformer())
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
