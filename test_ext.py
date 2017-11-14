from loaders.loading import load_train, load_batch, load_test
from pipeline import transform_chain
import gc
import os
from datetime import datetime
import csv


SUBM_PATH = r'../input/norm_challenge_ru'
INPUT_PATH = r'../input/norm_challenge_ru'
DATA_INPUT_PATH = r'../input/norm_challenge_ru/ru_with_types'

df = load_train(columns=['class', 'before', 'after'], input_path=INPUT_PATH)
df['before_prev'] = df['before'].shift(1)
df['before_next'] = df['before'].shift(-1)
df = df.fillna('')

x_train = df
y_train = x_train['after']
del df, x_train['after']

x_test = load_test(INPUT_PATH)
x_test['before_prev'] = x_test['before'].shift(1)
x_test['before_next'] = x_test['before'].shift(-1)
x_test = x_test.fillna('')

transform_chain.fit(x_train, y_train)
del x_train, y_train

for x_train in load_batch(columns=['class', 'before', 'after'],
                     batch_size=10,
                     input_path=DATA_INPUT_PATH):
    x_train['before_prev'] = x_train['before'].shift(1)
    x_train['before_next'] = x_train['before'].shift(-1)
    x_train = x_train.fillna('')

    y_train = x_train['after']
    del x_train['after']

    transform_chain.fit(x_train, y_train)

    del x_train, y_train
    gc.collect()

predict = transform_chain.transform(x_test)

predict['id'] = predict['sentence_id'].map(str) + '_' + predict['token_id'].map(str)
del predict['before']
del predict['sentence_id']
del predict['token_id']

predict.to_csv(os.path.join(SUBM_PATH, f'ru_predict_{datetime.now().strftime("%Y-%m-%d-%H-%M")}.csv'),
               quoting=csv.QUOTE_NONNUMERIC,
               index=False,
               columns=['id', 'after'])


