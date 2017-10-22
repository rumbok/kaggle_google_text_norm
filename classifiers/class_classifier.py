from sklearn.base import BaseEstimator, ClassifierMixin
from itertools import chain
import xgboost


class ClassClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model = xgboost.XGBClassifier(max_depth=10, objective='multi:softmax', silent=1, n_jobs=2)

    def fit(self, X, y):
        for (word, cls) in chain(X,y):
            if cls == self.classname:
                self.puncts.add(word)

    def predict(self, X):
        result = []
        for word in X:
            if word in self.puncts:
                result.append(self.classname)
            else:
                result.append(None)
        return result


if __name__ == '__main__':
    pass