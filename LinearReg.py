import numpy as np
class LinearRegression:
    def __init__(self,lr=0.001,n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weight = None
        self.bias = None
    def fit(self,X,y):
        n_sample,n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bias=0
        for _ in range(self.n_iters):
            y_pred = np.dot(X,self.weight)+self.bias
            dw = (1/n_sample)*np.dot(X.T,(y_pred-y))
            db = (1/n_sample)*np.sum(y_pred-y)
            self.weight-= self.lr*dw
            self.bias -= self.lr*db
    def predict(self,X):
        y_pred = np.dot(X,self.weight)+self.bias
        return y_pred

    def mse(self,y_test, predictions):
        return np.mean((y_test - predictions) ** 2)