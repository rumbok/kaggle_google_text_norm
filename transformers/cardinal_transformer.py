import pandas as pd
import pymorphy2
from sklearn.base import TransformerMixin, BaseEstimator
from tqdm import tqdm
from num2words import num2words
import re

CARDINAL_REGEXP = '^-?\d+$'
regexp = re.compile(CARDINAL_REGEXP)


class CardinalTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, use_case=False, use_number=False):
        self.use_case = use_case
        self.use_number = use_number
        self.morph = pymorphy2.MorphAnalyzer()
        self.numbers = {}

    def fit(self, X, y=None, *args, **kwargs):
        return self

    def transform(self, X: pd.DataFrame, y=None, *args, **kwargs):
        data = []
        if self.use_case:
            # for (before, cls, case, number) in tqdm(zip(X['before'], X['class'], X['case'], X['number']),
            #                                         f'{self.__class__.__name__} transform',
            #                                         total=len(X)):
            for (before, cls) in tqdm(zip(X['before'], X['class']),
                                      f'{self.__class__.__name__} transform',
                                      total=len(X)):
                if cls == 'CARDINAL' and regexp.match(before):
                    words = num2words(before, lang='ru')
                    inflected_words = []
                    for w in words.split():
                        # grammems = set()
                        # if self.use_case and case not in {'none', 'nomn', 'accs'}:
                        #     grammems.add(case)
                        # if self.use_number and number != 'none':
                        #     grammems.add(number)
                        #
                        # if len(grammems):
                        #     if w in self.numbers:
                        #         parsed = self.numbers[w]
                        #     else:
                        #         parsed = self.morph.parse(w)[0]
                        #         self.numbers[w] = parsed
                        #     inflected = parsed.inflect(grammems)
                        #
                        #     if inflected:
                        #         inflected_words.append(inflected.word)
                        #     else:
                        #         inflected_words.append(w)
                        # else:
                            inflected_words.append(w)

                    data.append(' '.join(inflected_words))
                else:
                    data.append(None)
        else:
            for (before, cls) in tqdm(zip(X['before'], X['class']), f'{self.__class__.__name__} transform',
                                      total=len(X)):
                if cls == 'CARDINAL' and regexp.match(before):
                    data.append(num2words(before, lang='ru'))
                else:
                    data.append(None)

        if 'after' in X.columns:
            return X.assign(after=X['after'].combine_first(pd.Series(data, index=X.index)))
        else:
            return X.assign(after=data)


if __name__ == '__main__':
    df = pd.DataFrame([[u'0', 'CARDINAL', 'gent', 'plur'],
                       [u'56', 'CARDINAL', 'nomn', 'plur'],
                       [u'-05665', 'CARDINAL', 'datv', 'plur'],
                       [u'7478', 'CARDINAL', 'nomn', 'sing'],
                       [u'7478', 'CARDINAL', 'gent', 'sing'],
                       [u'7478', 'CARDINAL', 'datv', 'none'],
                       [u'7478', 'CARDINAL', 'accs', 'sing'],
                       [u'7478', 'CARDINAL', 'ablt', 'sing'],
                       [u'7478', 'CARDINAL', 'loct', 'sing'],
                       [u'1001', 'CARDINAL', 'nomn', 'sing'], ],
                      columns=['before', 'class', 'case', 'number'])
    dt = CardinalTransformer(use_case=True)

    print(dt.fit_transform(df).head(20))
