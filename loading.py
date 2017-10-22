import os
import pandas as pd
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
    res = []
    files = glob(os.path.join(DATA_INPUT_PATH, "*"))
    for file in tqdm(files[:10], 'load files'):
        chunk = open(file, encoding='UTF8')
        while 1:
            line = chunk.readline().strip()
            if line == '':
                break
            arr = line.split('\t')
            if len(arr) < 3 or arr[0] == '<eos>':
                continue
            cls = arr[0]
            before = arr[1]
            if arr[2] == '<self>' or arr[2] == 'sil':
                after = arr[1]
            else:
                after = arr[2]
            res.append((cls, before, after))
        chunk.close()
    big_frame = pd.DataFrame(res, columns=['class', 'before', 'after'])
    del res
    return big_frame[columns]


if __name__ == '__main__':
    load_external(['class', 'before']).info()
