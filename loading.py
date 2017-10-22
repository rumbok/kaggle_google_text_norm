import os
import pandas as pd


INPUT_PATH = r'../input/norm_challenge_ru'
DATA_INPUT_PATH = r'../input/norm_challenge_ru/ru_with_types'
SUBM_PATH = INPUT_PATH


def load_train(columns: list) -> pd.DataFrame:
    return pd.read_csv(os.path.join(INPUT_PATH, 'ru_train.csv'),
                       encoding='utf-8',
                       index_col=False,
                       usecols=columns)


def load_external(usecols: list) -> pd.DataFrame:
    pass