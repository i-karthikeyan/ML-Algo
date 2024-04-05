import numpy as np
import matplotlib.pyplot as plt
def eulicidean_distance(x1,x2):
    # return np.sqrt(np.sum((x1-x2)**2))
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KMeans:
    def __init__(self,K=5,max_iters = 100,plot_steps = False):
        self.K=K
        self.max_iters=max_iters
        self.plot_steps=plot_steps
        self.clusters = [[]for _ in range(self.K)]
        self.centroids = []
    def predict(self,X):
        self.X= X
        self.n_sample,self.n_feature = X.shape
        random_sample_idx = np.random.choice(self.n_sample,self.K,replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idx]   # random centroids for frist time
        for _ in range(self.max_iters):
            self.clusters = self.create_clusters(self.centroids)
            if self.plot_steps:
                self.plot()
            old_centroids = self.centroids
            self.centroids = self.get_centroids(self.clusters)
            if self.is_same(old_centroids,self.centroids):
                break
            if self.plot_steps:
                self.plot()
        return self.get_cluster(self.clusters)
    def get_cluster(self,clusters):
        labels = np.empty(self.n_sample)
        for idx,cluster in enumerate(clusters):
            for sample_id in cluster:
                labels[sample_id] = idx
        return labels
    def create_clusters(self,centroids):
        clusters = [[] for _ in range(self.K)]
        for idx,sample in enumerate(self.X):
            centroid_idx = self.closest_centroid(sample,centroids)
            clusters[centroid_idx].append(idx)
        return clusters
    def closest_centroid(self,sample,centroids):
        distances = [eulicidean_distance(sample,point) for point in centroids]
        close = np.argmin(distances)
        return close


    def get_centroids(self,clusters):
        centroids = np.zeros((self.K,self.n_feature))
        for clidx,cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster],axis=0)
            centroids[clidx] = cluster_mean
        return centroids
    def is_same(self,old_centroids, centroids):
        distance = [eulicidean_distance(old_centroids[i],centroids[i]) for i in range(self.K)]
        return sum(distance) == 0
    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()

