import numpy as np
from collections import *
from DecisionTree import DecisionTree
class RandomForest:
    def __init__(self,n_tree=10,max_depth = 10,min_sample_split=2,n_feature = None):
        self.n_tree=n_tree
        self.max_depth = max_depth
        self .min_sample_split = min_sample_split
        self.n_feature = n_feature
        self.trees = []
    def fit(self,X,y):
        self.trees = []
        for _ in range(self.n_tree):
            tree = DecisionTree(max_depth=self.max_depth,
                                min_sample_split=self.min_sample_split,
                                n_feature=self.n_feature)
            x_sample,y_sample = self.booststrap_sample(X,y)
            tree.fit(x_sample,y_sample)
            self.trees.append(tree)

    def booststrap_sample(self,X,y):
        n_sample = X.shape[0]
        idsx = np.random.choice(n_sample,n_sample,replace=True)
        return X[idsx],y[idsx]
    def predict(self,X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self.most_common(pred) for pred in tree_preds])
        return predictions
    def most_common(self,pred):
        count = Counter(pred)
        mc = count.most_common(1)[0][0]
        return mc


