from sklearn.base import BaseEstimator, ClassifierMixin
from itertools import chain


class DictClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classname):
        self.classname = classname
        self.puncts = set()

    def predict(self, X):
        result = []
        for word in X:
            if word in self.puncts:
                result.append(self.classname)
            else:
                result.append(None)
        return result

    def fit(self, X, y):
        for (word, cls) in chain(X,y):
            if cls == self.classname:
                self.puncts.add(word)


if __name__ == '__main__':
    pass