import os
import pandas as pd
from tqdm import tqdm
from glob import glob

INPUT_PATH = r'../input/norm_challenge_ru'
DATA_INPUT_PATH = r'../input/norm_challenge_ru/ru_with_types'
SUBM_PATH = INPUT_PATH


def load_train(columns: list, input_path=INPUT_PATH) -> pd.DataFrame:
    return pd.read_csv(os.path.join(input_path, 'ru_train.csv'),
                       encoding='utf-8',
                       index_col=False,
                       usecols=columns)


def load_external(columns: list, head=0, only_diff=True, input_path=DATA_INPUT_PATH) -> pd.DataFrame:
    res = []
    files = glob(os.path.join(input_path, "*"))
    if head > 0:
        files = files[:head]
    for file in tqdm(files, 'load files'):
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
            if only_diff and before == after:
                continue
            res.append((cls, before, after))
        chunk.close()
    big_frame = pd.DataFrame(res, columns=['class', 'before', 'after'])
    del res
    return big_frame[columns]


def load_test() -> pd.DataFrame:
    return pd.read_csv(os.path.join(INPUT_PATH, 'ru_test.csv'),
                       encoding='utf-8',
                       index_col=False)


if __name__ == '__main__':
    load_train(['class', 'before'], INPUT_PATH).info()
