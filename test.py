from loaders.loading import load_train, load_test, load_external
from pipeline import transform
import os
from datetime import datetime
import csv

SUBM_PATH = r'../input/norm_challenge_ru'
INPUT_PATH = r'../input/norm_challenge_ru'
DATA_INPUT_PATH = r'../input/norm_challenge_ru/ru_with_types'

x_train = load_train(columns=['class', 'before', 'after'], input_path=INPUT_PATH)
#x_train = load_external(columns=['class', 'before', 'after'], only_diff=False, input_path=DATA_INPUT_PATH)
x_train['before_prev'] = x_train['before'].shift(1)
x_train['before_next'] = x_train['before'].shift(-1)
x_train = x_train.fillna('')
print(x_train.info())
y_train = x_train['after']
x_train = x_train.drop(['after'], axis=1)

x_test = load_test(INPUT_PATH)
x_test['before_prev'] = x_test['before'].shift(1)
x_test['before_next'] = x_test['before'].shift(-1)
x_test = x_test.fillna('')

predict = transform(x_train, x_test, y_train)

predict['id'] = predict['sentence_id'].map(str) + '_' + predict['token_id'].map(str)
del predict['before']
del predict['sentence_id']
del predict['token_id']

predict.to_csv(os.path.join(SUBM_PATH, f'ru_predict_{datetime.now().strftime("%Y-%m-%d-%H-%M")}.csv'),
               quoting=csv.QUOTE_NONNUMERIC,
               index=False,
               columns=['id', 'after'])
