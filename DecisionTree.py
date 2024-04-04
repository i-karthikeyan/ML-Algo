from collections import *
import numpy as np
class Node :
    def __init__(self,feature = None,threshold = None,left=None ,right=None,*,value = None):
        self.feature=feature
        self.threshold =threshold
        self.left = left
        self.right=right
        self.value = value
    def is_leaf_node(self):
        return self.value is not None
class DecisionTree:
    def __init__(self,min_sample_split=2,max_depth=100,n_feature = None):
        self.min_sample_split=min_sample_split
        self.max_depth=max_depth
        self .n_feature = n_feature
        self.root = None
    def fit(self,X,y):
        self.n_feature = X.shape[1] if not self.n_feature else min(X.shape[1],self.n_feature)
        self.root = self.grow_tree(X,y)
    def grow_tree(self, X, y, depth=0):
        n_sample,n_feats = X.shape
        n_lables = len(np.unique(y))
        if depth>= self.max_depth or n_lables == 1 or n_sample < self.min_sample_split:
            left_value = self.most_common_label(y)
            return Node(value=left_value)
        feat_idx = np.random.choice(n_feats,self.n_feature,replace=False) #no repeated values
        best_feature,best_thres = self.best_split(X,y,feat_idx)
        left_idxs,right_idxs = self.split(X[:,best_feature],best_thres)
        left = self.grow_tree(X[left_idxs,:],y[left_idxs],depth+1)
        right = self.grow_tree(X[right_idxs,:],y[right_idxs],depth+1)
        return Node(best_feature,best_thres,left,right)

    def best_split(self,X,y,feat_idx):
        best_gain = -1
        split_idx = split_threshold = None
        for f_idx in feat_idx:
            X_col = X[:,f_idx]
            threshold = np.unique(X_col)
            for thr in threshold:
                gain = self.information_gain(y,X_col,thr)
                if gain>best_gain:
                    best_gain = gain
                    split_idx = f_idx
                    split_threshold = thr
        return split_idx,split_threshold
    def information_gain(self,y,X_col,thr):
        parent = self.entropy(y)
        left_idx,right_idx = self.split(X_col,thr)
        if len(left_idx) == 0 or len(right_idx) == 0:
            return  0
        n = len(y)
        n_l,n_r = len(left_idx),len(right_idx)
        e_l,e_r = self.entropy(y[left_idx]),self.entropy(y[right_idx])
        child_e = (n_l/n)*e_l + (n_r/n)*e_r
        information_gain = parent-child_e
        return information_gain

    def split(self,X_col,thr):
        left_idx = np.argwhere(X_col<=thr).flatten()
        right_idx = np.argwhere(X_col > thr).flatten()
        return left_idx,right_idx
    def entropy(self,y):
        hist = np.bincount(y)
        ps = hist/len(y)
        return -np.sum([p*np.log(p) for p in ps if p>0])


    def most_common_label(self,y):
        count = Counter(y)
        value = count.most_common(1)[0][0]
        return value

    def predict(self,X):
        return np.array([self.traverse(x,self.root) for x in X])
    def traverse(self,x,node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature]<=node.threshold:
            return self.traverse(x,node.left)
        else:
            return self.traverse(x,node.right)




