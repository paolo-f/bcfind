from scipy.spatial.distance import pdist, squareform, euclidean


class EuclideanMetric:
    """
    Class that simply computes Euclidean distances
    """

    def __init__(self):
        pass

    def compute_single(self, x1, x2):
        """
        Takes as input two points and computes their euclidean distance.

        Parameters
        ----------

        x1 : numpy array 
            first point
        x2 : numpy array
            second point

        Returns
        -------

        distance : 
            euclidean distance between the points
        """

        return euclidean(x1, x2)

    def compute_pairwise(self, X):
        """
        Takes as input a Dataset matrix and returns a distance matrix.

        Parameters
        ----------

        X : numpy array of shape (n_samples, n_dimensions) 
            holding the points of the dataset

        Returns
        -------

        distance_matrix : numpy array of shape (n_samples, n_samples) 
            holding distances between dataset points
        """

        return squareform(pdist(X, 'euclidean', 2))
