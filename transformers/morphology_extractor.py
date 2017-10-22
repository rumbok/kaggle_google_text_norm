import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
import pymorphy2


class MorphologyExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.morph = pymorphy2.MorphAnalyzer()
        self.tags = {}

    def fit(self, X, y=None):
        return self

    def transform(self, words):
        res = []
        for word in tqdm(words, f'{self.__class__.__name__} transform'):
            is_first_upper = len(word) > 0 and word[0].isupper()
            is_upper = word.isupper()
            length = len(word)
            num_words = word.count(' ')

            if word.lower() in self.tags:
                p = self.tags[word.lower()]
            else:
                p = self.morph.parse(word)[0]
                self.tags[word.lower()] = p

            pos = p.tag.POS  # Part of Speech, часть речи
            animacy = p.tag.animacy  # одушевленность
            aspect = p.tag.aspect  # вид: совершенный или несовершенный
            case = p.tag.case  # падеж
            gender = p.tag.gender  # род (мужской, женский, средний)
            involvement = p.tag.involvement  # включенность говорящего в действие
            mood = p.tag.mood  # наклонение (повелительное, изъявительное)
            number = p.tag.number  # число (единственное, множественное)
            person = p.tag.person  # лицо (1, 2, 3)
            tense = p.tag.tense  # время (настоящее, прошедшее, будущее)
            transitivity = p.tag.transitivity  # переходность (переходный, непереходный)
            voice = p.tag.voice  # залог (действительный, страдательный)

            res.append((is_first_upper, is_upper, length, num_words,
                        pos, animacy, aspect, case, gender, involvement, mood,
                        number, person, tense, transitivity, voice))
        return pd.DataFrame(res, columns=['is_first_upper','is_upper', 'length', 'num_words',
                                          'pos','animacy', 'aspect','case','gender','involvement', 'mood','number',
                                          'person','tense','transitivity','voice'])
        # dt = [('is_first_upper', '?'), ('is_upper', '?'), ('pos', 'U4'), ('animacy', 'U4'),
        #       ('aspect', 'U4'), ('case', 'U4'), ('gender', 'U4'), ('mood', 'U4'), ('number', 'U4'),
        #       ('person', 'U4'), ('tense', 'U4'), ('transitivity', 'U4'), ('voice', 'U4')]
        # arr = np.array(res, dtype=dt)
        # return arr


if __name__ == '__main__':
    data = [u'в 1905 году'] + u'Определение частей речи работает не так как задумывалось'.split()
    print(data)
    res = MorphologyExtractor().transform(data)
    print(res)
    print(res.shape)
