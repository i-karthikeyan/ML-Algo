import numpy as np


class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):
        n_sample, n_feature = X.shape
        X_col = X[:, self.feature_idx]
        predictions = np.ones(n_sample)
        if self.polarity == 1:
            predictions[X_col < self.threshold] = -1
        else:
            predictions[X_col > self.threshold] = -1
        return predictions


class Adaboost:
    def __init__(self, n_clf):
        self.n_clf = n_clf
        self.clfs = []

    def fit(self, X, y):
        n_sample, n_feature = X.shape
        w = np.full(n_sample, (1 / n_sample))  # weight
        self.clfs = []
        for _ in range(self.n_clf):
            clf = DecisionStump()
            mini_error = float('inf')
            for feature_i in range(n_feature):
                X_col = X[:, feature_i]
                thresholds = np.unique(X_col)
                for thr in thresholds:
                    pola = 1
                    predictions = np.ones(n_sample)
                    predictions[X_col < thr] = -1
                    missclassified = w[y != predictions]
                    error = sum(missclassified)
                    if error > 0.5:
                        error = 1 - error
                        pola = -1
                    if error < mini_error:
                        mini_error = error
                        clf.polarity = pola
                        clf.threshold = thr
                        clf.feature_idx = feature_i

        eps = 1e-10
        clf.alpha = 0.5 * np.log((1 - mini_error +eps)/ (mini_error + eps))
        predictions = clf.predict(X)
        w *= np.exp(-clf.alpha * y * predictions)
        w /= np.sum(w)
        self.clfs.append(clf)

    def predict(self, X):
        cld_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sum(cld_preds, axis=0)
        y_pred = np.sign(y_pred)
        return y_pred
