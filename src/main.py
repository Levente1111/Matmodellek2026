import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA


from google_data_downloader import GoogleDataDownloader
from src.clustering_plotter import ClusteringPlotter
from src.k_means_clustering import KMeansClustering


def main():
    url = "https://drive.google.com/file/d/1d66BhqNfbBhNmQSswYeR_gMJyruqUS1x/view?usp=sharing"
    f_name = "iris.csv"
    g_ddl = GoogleDataDownloader(file_url=url, file_name=f_name)

    iris_data = pd.read_csv(filepath_or_buffer=g_ddl.file_path)

    # vetítés a legjobb 2D síkra
    # (bővebben: A gépi tanulás matematikai alapjai kurzuson)
    pca = PCA(n_components=2)
    iris_2d = pca.fit_transform(X=iris_data.drop("variety", axis=1))

    k_means = KMeansClustering(data=iris_2d, k=3, n_iter=50)
    k_means.run()

    plotter = ClusteringPlotter(clustering=k_means)
    plotter.run()
    plt.show
if __name__ == "__main__":
    main()
