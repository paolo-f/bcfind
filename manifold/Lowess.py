import numpy as np
from SurfaceCleaner import SurfaceCleaner
from scipy.linalg import lstsq
import parameters


class Lowess:
    """
    Class that computes Lowess regression, which is a kind of non-parametric regression method.
    Basically, for every point in the dataset it computes a linear regression, using data points
    weighted with user specified weights. In a second step, weights are modified with a Polynomial
    kernel designed to spot outliers and Lowess is iterated for robustness.

    Parameters
    ----------

    metric : object
        an EuclideanMetric() object to compute distances in the Polynomial kernel
    robust_iter : int
        number of extra iteration of Lowess regression in order to smooth outliers

    References
    ----------
    Cleveland, W. S. (1979). Robust locally weighted regression and smoothing
    scatterplots. J Am Stat Assoc, 74(368), 829-836.
    """

    def __init__(self, metric, robust_iter):
        self._weight_threshold = parameters.weight_threshold
        self._robust_iter = robust_iter
        self._metric = metric

    def fit_transform(self, X, Y, weights):
        """
        Method that actually fit the points with Lowess regression.

        Parameters
        ----------

        X : numpy array of shape (n_samples, n_expl_dim)
            explanatory data
        Y : numpy array of shape (n_samples, n_dep_dim)
            dependent data
        weights : numpy array of shape (n_samples, n_samples)
            holds the weights for the fitting of each point

        Returns
        -------

        ret : numpy array of shape (n_samples, n_dep_dim)
            holds the fitted data
        """

        n_samples = X.shape[0]
        n_targets = Y.shape[1]
        ret = np.zeros((n_samples, n_targets))
        for i in xrange(n_samples):
            try:
                ret[i, :] = self._fit_point(X, Y, weights, i)
            except RuntimeError:
                ret[i, :] = Y[i, :]
        for j in xrange(self._robust_iter):
            delta = self._compute_delta(Y, ret)
            weights *= delta #FIXME weights matrix is now non-symmetric!
            for i in xrange(n_samples):
                try:
                    ret[i, :] = self._fit_point(X, Y, weights, i)
                except RuntimeError:
                    continue
        return ret, delta







    def fit_new_points(self, X, Y, X_new, delta, new_weights):
        """
        Method that actually fit the points with Lowess regression.

        Parameters
        ----------

        X : numpy array of shape (n_samples, n_expl_dim)
            explanatory data
        Y : numpy array of shape (n_samples, n_dep_dim)
            dependent data
        weights : numpy array of shape (n_samples, n_samples)
            holds the weights for the fitting of each point

        Returns
        -------

        ret : numpy array of shape (n_samples, n_dep_dim)
            holds the fitted data
        """

        n_targets = Y.shape[1]
        n_new_samples = X_new.shape[0]
        new_ret = np.zeros((n_new_samples, n_targets))
        for i in xrange(n_new_samples):
            point = X_new[i, :]
            point_weights = new_weights[i, :]
            indexes = point_weights > self._weight_threshold
            if np.any(indexes):
                weights_less = point_weights[indexes]
                X_less = X[indexes, :]
                Y_less = Y[indexes, :]
                X_less, Y_less, X_less_mean, Y_less_mean = self._center_data(X_less, Y_less, weights_less)
                coef = self._compute_coef(X_less, Y_less)
                intercept = self._compute_intercept(coef, X_less_mean, Y_less_mean)
                new_ret[i,:] = np.dot(point, coef) + intercept
            else:
                raise RuntimeError('No points have relevant weights')

        return new_ret



    def _compute_delta(self, Y, ret):
        """
        Method that computes the Polynomial kernel which lower the weights of probable outliers.

        Parameters

        Y: numpy array of shape (n_samples, n_dep_dim) that represent the dependent data
        ret: numpy array of shape (n_samples, n_dep_dim) that represent the fitted data

        Returns

        delta: numpy array of shape (n_samples, ) that holds the modification weights
        """

        cleaner = SurfaceCleaner(self._metric)
        residuals = cleaner.compute_distances(Y, ret)
        s = np.median(residuals)
        delta = np.clip(residuals / (6 * s), -1, 1)
        delta = 1 - delta * delta
        delta *= delta
        return delta

    def _fit_point(self, X, Y, weights, i):
        """
        Method that fits a single point given its smoothing weights.

        Parameters

        X: numpy array of shape (n_samples, n_expl_dim) that represent the explanatory data
        Y: numpy array of shape (n_samples, n_dep_dim) that represent the dependent data
        weights: numpy array of shape (n_samples, n_samples) which holds the weights for the fitting of each point
        i: index (in the X and Y numpy array) of the point to be fitted

        Returns

        fitted: numpy array of shape (n_dep_dim, ) with the fitted point
        """

        point = X[i, :]
        point_weights = weights[i, :]
        indexes = point_weights > self._weight_threshold
        if np.any(indexes):
            weights_less = point_weights[indexes]
            X_less = X[indexes, :]
            Y_less = Y[indexes, :]
            X_less, Y_less, X_less_mean, Y_less_mean = self._center_data(X_less, Y_less, weights_less)
            coef = self._compute_coef(X_less, Y_less)
            intercept = self._compute_intercept(coef, X_less_mean, Y_less_mean)
            return np.dot(point, coef) + intercept
        else:
            raise RuntimeError('No points have relevant weights')

    def _compute_coef(self, X, Y):
        """
        Method that actually computes the linear regression. Need centered data

        Parameters

        X: numpy array of shape (n_samples, n_expl_dim) that represent the explanatory data
        Y: numpy array of shape (n_samples, n_dep_dim) that represent the dependent data

        Returns

        coef: numpy array of shape (n_expl_dim, n_dep_dim) holding the coefficients of the linear fit
        """

        coef, residuals, rank, s = lstsq(X, Y)
        return coef

    def _compute_intercept(self, coef, X_mean, Y_mean):
        """
        Method that computes the intercept of the fit.

        Parameters

        coef: numpy array of shape (n_expl_dim, n_dep_dim) holding the coefficients of the linear fit
        X_mean: numpy array mean vector of shape (n_expl_dim, ) of the explanatory data
        Y_mean: numpy array mean vector of shape (n_dep_dim, ) of the dependent data

        Returns

        intercept: numpy array of shape (n_dep_dim, ) holding intercept coordinates
        """

        intercept = Y_mean - np.dot(X_mean, coef)
        return intercept

    def _center_data(self, X, Y, weights):
        """
        Method that centers the data.

        Parameters

        X: numpy array of shape (n_samples, n_expl_dim) that represent the explanatory data
        Y: numpy array of shape (n_samples, n_dep_dim) that represent the dependent data
        weights: numpy array of shape (n_samples, n_samples) which holds the weights for the fitting of each point

        Returns:

        X: numpy array of shape (n_samples, n_expl_dim) that represent the centered explanatory data
        Y: numpy array of shape (n_samples, n_dep_dim) that represent the centered dependent data
        X_mean: numpy array mean vector of shape (n_expl_dim, ) of the explanatory data
        Y_mean: numpy array mean vector of shape (n_dep_dim, ) of the dependent data
        """

        X_mean = (np.sum(X * weights[:, np.newaxis], 0) / np.sum(weights))
        X = X - X_mean
        Y_mean = (np.sum(Y * weights[:, np.newaxis], 0) / np.sum(weights))
        Y = Y - Y_mean
        return X, Y, X_mean, Y_mean
