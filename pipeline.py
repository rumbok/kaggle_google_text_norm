import numpy as np
from transformers.dash_transformer import DashTransformer
from transformers.dict_nbhd_transformer import DictNBHDTransformer
from transformers.dict_transformer import DictTransformer
from transformers.digit_transformer import DigitTransformer
from transformers.flat_transformer import FlatTransformer
from transformers.dict_class_transformer import DictClassTransformer
from transformers.letters_transformer import LettersTransformer
from transformers.lstm_transliterator import LSTMTransliterator
from transformers.scored_chain import ScoredChain
from transformers.self_transformer import SelfTransformer
from time import gmtime, strftime
from transformers.add_class_transformer import AddClassTransformer
from transformers.simple_transliterator import SimpleTransliterator


transform_chain = ScoredChain([
        ('class', AddClassTransformer('models/class.model.train_1190428_0.00101_0.3_500_6')),
        ('digit', DigitTransformer()),
        ('dash', DashTransformer()),
        ('punсt', DictClassTransformer(u'PUNCT', 1.0)),
        ('verbatim', DictClassTransformer(u'VERBATIM', 1.0)),
        ('self', SelfTransformer(threshold=0.5, modelpath='models/self.model.train_9517064_0.00117_0.3_500_6')),
        ('dict_nbhd', DictNBHDTransformer(0.5)),
        ('letters', LettersTransformer()),
        ('date', DictClassTransformer(u'DATE', 1.0)),
        ('cardinal', DictClassTransformer(u'CARDINAL', 1.0)),
        ('dict', DictTransformer(0.5)),
        ('translit', LSTMTransliterator('checkpoint_epoch_36_0.8127_64_2_0.0.hdf5')),
        ('translit', SimpleTransliterator('cyrtranslit')),
        ('flat', FlatTransformer())
    ])


def transform(x_train, x_test, y_train, y_test=None):
    transform_chain.metrics=[
        ('tp', lambda X: np.mean((X["after"] == y_test)) if 'after' in X.columns and y_test is not None else 0.0),
        ('fp', lambda X: np.mean((X["after"].notnull()) & ~(X["after"] == y_test)) if 'after' in X.columns and y_test is not None else 0.0)
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
