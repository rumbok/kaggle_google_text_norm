from loaders.loading import load_batch
from pipeline import transform_chain
from time import gmtime, strftime
import numpy as np
import gc

SUBM_PATH = r'../input/norm_challenge_ru'
INPUT_PATH = r'../input/norm_challenge_ru'
DATA_INPUT_PATH = r'../input/norm_challenge_ru/ru_with_types'

first = True
for x_train in load_batch(columns=['class', 'before', 'after'],
                          batch_size=10,
                          input_path=DATA_INPUT_PATH):
    x_train['prev_prev'] = x_train['before'].shift(2)
    x_train['prev'] = x_train['before'].shift(1)
    x_train['next'] = x_train['before'].shift(-1)
    x_train['next_next'] = x_train['before'].shift(-2)
    x_train = x_train.fillna('')

    if first:
        x_test = x_train
        y_test = x_test['after']
        del x_test['after'], x_test['class'], x_train
        gc.collect()
        first = False
    else:
        y_train = x_train['after']
        del x_train['after']

        transform_chain.fit(x_train, y_train)

        del x_train, y_train
        gc.collect()

transform_chain.metrics = [
    ('tp', lambda X: np.mean((X["after"] == y_test)) if 'after' in X.columns and y_test is not None else 0.0),
    ('fp', lambda X: np.mean(
        (X["after"].notnull()) & ~(X["after"] == y_test)) if 'after' in X.columns and y_test is not None else 0.0)
]
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
