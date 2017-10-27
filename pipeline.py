import os
import numpy as np
import pandas as pd
from loaders.loading import load_train
from transformers.dash_transformer import DashTransformer
from transformers.dict_nbhd_transformer import DictNBHDTransformer
from transformers.dict_transformer import DictTransformer
from transformers.digit_transformer import DigitTransformer
from transformers.flat_transformer import FlatTransformer
from transformers.dict_class_transformer import DictClassTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from transformers.letters_transformer import LettersTransformer
from transformers.scored_chain import ScoredChain
from transformers.self_transformer import SelfTransformer
from time import gmtime, strftime
from transformers.simple_transliterator import SimpleTransliterator

SUBM_PATH = r'../input/norm_challenge_ru'

df = load_train(['class', 'before', 'after'])
df['before_prev'] = df['before'].shift(1)
df['before_next'] = df['before'].shift(-1)
df = df.fillna('')
print(df.info())

classes = ['PLAIN', 'DATE', 'PUNCT', 'ORDINAL', 'VERBATIM', 'LETTERS', 'CARDINAL', 'MEASURE', 'TELEPHONE', 'ELECTRONIC',
           'DECIMAL', 'DIGIT', 'FRACTION', 'MONEY', 'TIME']

x_train, x_test, y_train, y_test = train_test_split(df.drop(['after'], axis=1), df['after'], test_size=0.1, random_state=2017)

# test_size = int(len(df.index)*0.1)
# x_train = df.drop(['after'], axis=1).head(len(df.index)-test_size)
# y_train = df['after'].head(len(df.index)-test_size)
# x_test = df.drop(['after'], axis=1).tail(test_size)
# y_test = df['after'].tail(test_size)

transform_chain = ScoredChain([
    # ('dash', DashTransformer()),
    # ('digit', DigitTransformer()),
    # ('punсt', DictClassTransformer(u'PUNCT', 1.0)),
    # ('verbatim', DictClassTransformer(u'VERBATIM', 1.0)),
    ('self', SelfTransformer(threshold=0.5, modelpath='models/self.model.train')),
    # ('dict_nbhd', DictNBHDTransformer(0.5)),
    # ('letters', LettersTransformer()),
    # ('dict', DictTransformer(0.5)),
    # ('translit', SimpleTransliterator('cyrtranslit')),
    # ('flat', FlatTransformer())
], metrics=[
    ('tp', lambda X: np.mean((X["after"] == y_test))),
    ('fp', lambda X: np.mean((X["after"].notnull()) & ~(X["after"] == y_test)))
])

transform_chain.fit(x_train, y_train)
predicted = transform_chain.transform(x_test)

with open('pipeline_results', 'a+') as f:
    f.write(strftime("%Y-%m-%d %H:%M:%S\n", gmtime()))
    f.write(f'test data {len(y_test)}\n')
    for step in transform_chain.steps:
        f.write(f'{step[0]}: {step[1].get_params()}\n')
    f.write('Chain:\n')
    for step_result in transform_chain.results:
        f.write(f'  {step_result}\n')
    f.write('\n')