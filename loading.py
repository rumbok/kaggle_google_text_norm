import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob

INPUT_PATH = r'../input/norm_challenge_ru'
DATA_INPUT_PATH = r'../input/norm_challenge_ru/ru_with_types'
SUBM_PATH = INPUT_PATH


def load_train(columns: list) -> pd.DataFrame:
    return pd.read_csv(os.path.join(INPUT_PATH, 'ru_train.csv'),
                       encoding='utf-8',
                       index_col=False,
                       usecols=columns)


def load_external(columns: list) -> pd.DataFrame:
    files = glob(os.path.join(DATA_INPUT_PATH, "*"))
    np_array_list = []
    for file in tqdm(files[:2], 'load files'):
        np_array_list.append(np.genfromtxt(file,
                                           dtype="U,U,U",
                                           delimiter='\t',
                                           names=['class', 'before', 'after'],
                                           usecols=columns,
                                           loose=True))
        # df = pd.read_csv(file,
        #                  encoding='utf-8',
        #                  index_col=False,
        #                  sep='\t',
        #                  header=None,
        #                  names=['class', 'before', 'after'],
        #                  usecols=columns,
        #                  error_bad_lines=False)
        # np_array_list.append(df.as_matrix())
    comb_np_array = np.vstack(np_array_list)
    big_frame = pd.DataFrame(comb_np_array)
    big_frame.columns = columns

    return big_frame


if __name__ == '__main__':
    load_external(['before','class']).info()
