import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
from scipy.sparse import coo_matrix
import pymorphy2
from typing import Union


class MorphologyExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, to_coo=False, multi_words=False):
        self.to_coo = to_coo
        self.multi_words = multi_words
        self.morph = pymorphy2.MorphAnalyzer()
        self.word_rows = {}

        tag_values = set()
        for (_, tag, _, _, _) in tqdm(self.morph.dictionary.iter_known_words(),
                                      f'{self.__class__.__name__} create index'):
            tag_values.update(tag.grammemes)
        self.tag_indexes = {x: i + 4 for i, x in enumerate(tag_values)}

    def fit(self, X, y=None):
        return self

    def transform(self, words) -> Union[pd.SparseDataFrame, coo_matrix]:
        rows = []
        cols = []
        datas = []
        for i, word in tqdm(enumerate(words),
                            f'{self.__class__.__name__} transform',
                            total=len(words)):
            if word in self.word_rows:
                col_data = self.word_rows[word]
            else:
                is_first_upper = len(word) > 0 and word[0].isupper()
                is_upper = word.isupper()
                length = len(word)
                num_words = len(word.split())

                if self.multi_words and ' ' in word:
                    tags = set()
                    for w in word.split():
                        tags.update(self.morph.parse(w)[0].tag.grammemes)
                        print(w, tags)
                    indexes = [self.tag_indexes[k] for k in tags if k in self.tag_indexes]
                else:
                    indexes = [self.tag_indexes[k] for k in self.morph.parse(word)[0].tag.grammemes
                               if k in self.tag_indexes]
                print(indexes)

                col = [0, 1, 2, 3] + indexes
                data = [is_first_upper, is_upper, length, num_words] + [1] * len(indexes)
                col_data = (col, data)
                self.word_rows[word] = col_data

            rows += [i] * len(col_data[0])
            cols += col_data[0]
            datas += col_data[1]

            # p.tag.POS  # Part of Speech, часть речи
            # p.tag.animacy  # одушевленность
            # p.tag.aspect  # вид: совершенный или несовершенный
            # p.tag.case  # падеж
            # p.tag.gender  # род (мужской, женский, средний)
            # p.tag.involvement  # включенность говорящего в действие
            # p.tag.mood  # наклонение (повелительное, изъявительное)
            # p.tag.number  # число (единственное, множественное)
            # p.tag.person  # лицо (1, 2, 3)
            # p.tag.tense  # время (настоящее, прошедшее, будущее)
            # p.tag.transitivity  # переходность (переходный, непереходный)

        coo_matr = coo_matrix((datas, (rows, cols)), shape=(len(words), 4 + len(self.tag_indexes)),
                              dtype=np.uint8)
        del datas, rows, cols

        if self.to_coo:
            return coo_matr
        else:
            return pd.SparseDataFrame(coo_matr,
                                      columns=['is_first_upper', 'is_upper', 'length', 'num_words'] + list(self.tag_indexes.keys()),
                                      default_fill_value=0)


if __name__ == '__main__':
    data = [u'В 1905 году'] + u'съешь ещё этих мягких французских булок , ДА выпей чаю брюки брючные'.split()
    print(data)

    morph = MorphologyExtractor()
    res = morph.transform(data)
    print(res.info())
    print(res.shape)
    print(res.density)

    morph.to_coo = True
    res_coo = morph.transform(data)
    print(res_coo.shape)
    print(res_coo.nnz / res_coo.shape[0] / res_coo.shape[1])