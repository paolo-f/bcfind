import numpy as np


class SurfaceCleaner:
    """
    Class that computes distances between original points and reconstructed points.

    Parameters
    ----------

    metric : object
        EuclideanMetric() object to compute distances
    """

    def __init__(self, metric):
        self._metric = metric

    def compute_distances(self, X, X_reb):
        """
        Method that actually computes reconstruction distances.

        Parameters
        ----------

        X : numpy array of shape (n_samples, n_dimensions)
            original data points
        X_reb : numpy array of shape (n_samples, n_dimensions)
            reconstructed data points

        Returns
        -------

        distances : numpy array of shape (n_samples, ) 
            reconstruction distances
        """

        rows = X.shape[0]
        distances = np.zeros(rows)
        for row in xrange(rows):
            distance = self._metric.compute_single(X[row, :], X_reb[row, :])
            distances[row] = distance
        return distances
