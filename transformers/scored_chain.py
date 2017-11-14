from sklearn.base import BaseEstimator, TransformerMixin
import gc


class ScoredChain(BaseEstimator, TransformerMixin):
    def __init__(self, steps, metrics=[]):
        self.steps = steps
        self.metrics = metrics
        self.results = []

    def fit(self, X, y=None):
        for step in self.steps:
            step[1].fit(X, y)
        return self

    def transform(self, X):
        transformed_X = X
        self.results = []
        for step in self.steps:
            transformed_X = step[1].transform(transformed_X)
            gc.collect()

            step_result = f'{step[0]}:'
            for metric in self.metrics:
                step_result += f'\t{metric[0]}={metric[1](transformed_X):0.6}'
            self.results.append(step_result)
        return transformed_X
