import matplotlib.pyplot as plt

from k_means_clustering import KMeansClustering


class ClusteringPlotter:
    def __init__(self, clustering: KMeansClustering):
        self.clustering = clustering

    def run(self):
        colors = ["lightblue", "pink", "lightgreen"]
        centr_colors = ["blue", "red", "green"]

        for idx in range(0, self.clustering.k):
            clust = self.clustering.clusters[idx]
            centr = self.clustering.centroids[idx]
            color = colors[idx]
            centr_color = centr_colors[idx]

            x = clust[:, 0]
            y = clust[:, 1]
            plt.scatter(x, y, c=color)
            plt.scatter(centr[0], centr[1], c=centr_color, s=50)
