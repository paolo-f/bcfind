from sklearn.manifold import Isomap


class IsomapEmbedder:
    """
    Class that computes the 2-dimensional ISOMAP embedding of a set of points.

    Parameters
    ----------

    n_neighbors : int
        number of nearest neighbors to use to compute the kNN graph of the data
    """

    def __init__(self, n_neighbors):
        self._n_neighbors = n_neighbors
        self._n_components = 2
        self._embedder = Isomap(self._n_neighbors, self._n_components)

    def compute(self, X):
        """
        Method that actually computes the embedding of the data points.

        Parameters
        ----------

        X : numpy array of shape (n_samples, n_dimensions) 
            holding the points of the dataset

        Returns
        -------

        2d_points : numpy array of shape (n_samples, 2) 
            holding the embedding of the points of the dataset
        """

        return self._embedder.fit_transform(X)
