import numpy as np


class GaussianKernel:
    """
    Class that computes Gaussian kernel values.

    Parameters

    sigma: the sigma parameter for the Gaussian kernel, e.g. the standard deviation
    metric: an EuclideanMetric() object to compute distances
    """

    def __init__(self, sigma, metric):
        self._sigma = sigma
        self._metric = metric

    def compute_single(self, x1, x2):
        """
        Takes as input two points and compute Gaussian kernel value.

        Parameters

        x1: numpy array of first point
        x2: numpy array of second point

        Returns

        value: Gaussian kernel value between the points
        """

        distance = self._metric.compute_single(x1, x2)
        return np.exp(-np.power(distance, 2) / self._sigma ** 2)

    def compute_multiple(self, X):
        """
        Takes as input a Dataset matrix and returns a Gaussian kernel values matrix.

        Parameters

        X: numpy array of shape (n_samples, n_dimensions) holding the points of the dataset

        Returns

        values_matrix: numpy array of shape (n_samples, n_samples) holding Gaussian kernel values between dataset points
        """

        pairwise_distances = self._metric.compute_pairwise(X)
        return np.exp(-np.power(pairwise_distances, 2) / self._sigma ** 2)
