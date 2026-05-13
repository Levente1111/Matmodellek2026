import numpy as np

from pairwise_distance_calculator import PairwiseDistanceCalculator

class KMeansClustering:
    def __init__(self, data: np.ndarray, k: int, n_iter: int):
        self.data = data
        self.k = k
        self.n_iter = n_iter

        self.__clusters = None
        self.__centroids = None
        self.prepare()

    def prepare(self):
        n = self.data.shape[0]
        indexes = np.random.choice(n, self.k, replace=False)
        centroids = dict(zip(range(0, self.k), self.data[indexes]))

        clusters = dict()
        for cl in range(0, self.k):
            clusters[cl] = []

        self.__centroids = centroids
        self.__clusters = clusters

    @property
    def cluster(self) -> dict:
        return self.__clusters

    @property
    def centroids(self) ->dict:
        return self.__centroids

    def run(self):
        for step in range(0, self.n_iter):
            self.calculate_clusters()
            self.calculate_centroids()

    def calculate_clusters(self):
        calculate_distances = self.calculate_pairwise_distances()
        nearest_centroid_index = self.calculate_nearest_centroid_index(distances = calculate_distances)
        clusters = self.group_by_index(nearest_centroid_index)
        self.__clusters = clusters

    def calculate_centroids(self):
        centroids= dict()
        for cl in range(0, self.k):
            centroid = np.mean(self.__clusters[cl], axis=0)
            centroids[cl] = centroid
        self.__centroids = centroids

    def calculate_pairwise_distances(self) -> np.ndarray:
        vectors1 = self.data
        vectors2 = np.array(list(self.centroids.values()))

        return PairwiseDistanceCalculator(vectors1, vectors2).compute_distance_no_loop()

    @staticmethod
    def calculate_nearest_centroid_index(distances: np.ndarray) -> np.ndarray:
        nearest_centroid_index = np.argmin(distances, axis=1)
        return nearest_centroid_index

    def group_by_index(self, indexes: np.ndarray) -> dict:
        clusters = dict()
        for cl in range(0, self.k):
            clusters[cl] = self.data[indexes == cl]

        return clusters