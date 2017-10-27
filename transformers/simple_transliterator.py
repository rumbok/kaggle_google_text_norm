from transliterate import translit
import cyrtranslit
import pytils
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from tqdm import tqdm
import re

ENG_REGEXP = '[a-zA-Z\'_]'
re_eng = re.compile(ENG_REGEXP, re.IGNORECASE | re.UNICODE)


class SimpleTransliterator(TransformerMixin, BaseEstimator):
    def __init__(self, algo='cyrtranslit'):
        self.algo = algo

    def fit(self, X, y=None, *args, **kwargs):
        return self

    def transform(self, X: pd.DataFrame, y=None, *args, **kwargs):
        data = []
        for w in tqdm(X['before'], f'{self.__class__.__name__} transform', total=len(X)):
            if re_eng.match(w):
                if self.algo == 'translit':
                    rus_w = translit(w, language_code='ru').lower()
                elif self.algo == 'cyrtranslit':
                    rus_w = cyrtranslit.to_cyrillic(w, lang_code='ru').lower()
                elif self.algo == 'pytils':
                    rus_w = pytils.translit.detranslify(w).lower()
                data.append(' '.join([c + u'_trans' for c in rus_w]))
            else:
                data.append(None)
        if 'after' in X.columns:
            return X.assign(after=X['after'].combine_first(pd.Series(data, index=X.index)))
        else:
            return X.assign(after=data)


if __name__ == '__main__':
    df = pd.DataFrame.from_dict({"Tiberius": "т_trans и_trans б_trans е_trans р_trans и_trans у_trans с_trans",
                                 "Julius": "д_trans ж_trans у_trans л_trans и_trans у_trans с_trans",
                                 "Pollienus": "п_trans о_trans л_trans л_trans и_trans е_trans н_trans у_trans с_trans",
                                 "Auspex": "о_trans с_trans п_trans е_trans к_trans с_trans",
                                 "Half": "х_trans а_trans л_trans ф_trans",
                                 "Armor": "а_trans р_trans м_trans о_trans р_trans",
                                 "Sbrinz": "с_trans б_trans р_trans и_trans н_trans с_trans",
                                 "Kase": "к_trans е_trans й_trans с_trans",
                                 "GmbH": "г_trans м_trans б_trans",
                                 "The": "з_trans э_trans",
                                 "next": "н_trans е_trans к_trans с_trans т_trans",
                                 "supermoon": "с_trans у_trans п_trans е_trans р_trans м_trans у_trans н_trans",
                                 "in": "и_trans н_trans",
                                 "is": "и_trans с_trans",
                                 "July": "д_trans ж_trans у_trans л_trans и_trans",
                                 "русское": "русское"}, orient='index')\
        .reset_index()\
        .rename(columns={'index': 'before', 0: 'actual'})

    res = SimpleTransliterator('translit').transform(df)
    print('translit', np.mean(res['actual']==res['after']))

    res = SimpleTransliterator('cyrtranslit').transform(df)
    print('cyrtranslit', np.mean(res['actual']==res['after']))

    res = SimpleTransliterator('pytils').transform(df)
    print('pytils', np.mean(res['actual']==res['after']))
