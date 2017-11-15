import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
import pymorphy2
from collections import defaultdict
from pandas.api.types import CategoricalDtype
from loaders.loading import load_train


class CaseExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, multi_words=False):
        self.multi_words = multi_words
        self.morph = pymorphy2.MorphAnalyzer()
        self.word_rows = {}

        self.case_type = CategoricalDtype(categories=['none', 'nomn', 'gent', 'datv', 'accs', 'ablt', 'loct', 'voct', 'gen2', 'acc2', 'loc2'])
        self.number_type = CategoricalDtype(categories=['none', 'sing', 'plur'])

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, words, y=None, **fit_params) -> pd.DataFrame:
        data = []
        for word in tqdm(words,
                         f'{self.__class__.__name__} transform',
                         total=len(words)):
            case = None
            number = None
            if word.lower() in self.word_rows:
                (case, number) = self.word_rows[word.lower()]
            else:
                if self.multi_words:
                    cases = defaultdict(lambda: 0.0)
                    numbers = defaultdict(lambda: 0.0)
                    for w in word.split():
                        for p in self.morph.parse(w):
                            if p.tag.case is not None:
                                cases[p.tag.case] += p.score
                            if p.tag.number is not None:
                                numbers[p.tag.number] += p.score
                    if cases:
                        case = max(cases, key=cases.get)
                    if numbers:
                        number = max(numbers, key=numbers.get)
                else:
                    tag = self.morph.parse(word)[0].tag
                    case = tag.case
                    number = tag.number
                self.word_rows[word.lower()] = (case, number)

            data.append((case, number))

            # p.tag.case  # падеж
            # p.tag.number  # число (единственное, множественное)
        res = pd.DataFrame(data, columns=['case', 'number']).fillna('none')
        del data
        res['case'] = res['case'].astype(self.case_type)
        res['number'] = res['number'].astype(self.number_type)
        return res


if __name__ == '__main__':
    data = [u'В 1905 году'] + u'съешь ещё этих мягких французских булок , ДА выпей чаю брюки брючные'.split()
    print(data)

    morph = CaseExtractor()
    res = morph.transform(data)
    print(res.info())
    print(res)

    morph.multi_words = True
    morph.word_rows = {}
    res = morph.transform(data)
    print(res.info())
    print(res)

    df = load_train(columns=['after'], input_path=r'../../input/norm_challenge_ru')
    morph.word_rows = {}
    res = morph.transform(df.sample(100000)['after'])
    print(res.info())
    print(res)
