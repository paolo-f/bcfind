import numpy as np
from scipy.spatial import cKDTree
from Queue import PriorityQueue


class PatchMaker:
    """
    Class that computes a patch given the set of data points and a seed.

    Parameters
    ----------

    X : numpy array of shape (n_samples, n_dimensions)
        whole dataset
    seed : int
        index of a row of X representing the point from which the patch making process starts
    n_neighbors : int
        number of nearest neighbors to use to build the kNN graph of the data to make the patch
    max_distance : float
        maximum geodesic distance at which Uniform Cost Search stops expanding nodes
    """

    def __init__(self, X, seed, n_neighbors, max_distance):
        self._X = X
        self._kdtree = cKDTree(self._X)
        self._seed = seed
        self._n_neighbors = n_neighbors
        self._max_distance = max_distance

    def _init_open_points(self, start_index):
        """
        Method that inits the Priority Queue needed in the Uniform Cost Search algorithm.

        Parameters
        ----------

        start_index : int
            index of the seed point

        Returns
        -------

        open_points : object
            a Priority Queue holding only the seed point with cost 0
        """

        open_points = PriorityQueue()
        item = [0, start_index]
        open_points.put(item)
        return open_points

    def _init_open_points_dict(self, start_index):
        """
        Method that inits a dictionary needed to speed up the Uniform Cost Search algorithm.

        Parameters
        ----------

        start_index : int
            index of the seed point

        Returns
        -------

        open_points_dict : dict
            a dictionary holding only the seed point with cost 0
        """

        open_points_dict = dict()
        open_points_dict[start_index] = 0
        return open_points_dict

    def _init_closed_points(self):
        """
        Method that inits the set of visited points

        Returns

        closed_points: empty set of visited points
        """

        return set()

    def visit_results(self, seed):
        """
        Method that uses the Uniform Cost Search algorithm to create a patch of points.
        Actually, this is just an implementation of the Uniform Cost Search algorithm,
        using as cost function the geodesic distance between two data points.

        Parameters
        ----------

        seed : int
            index of a point in the dataset from which the patch making process starts

        Returns
        -------

        closed_points : set
            set of visited points which have a geodesic distance from the seed
            under the maximum specified by self._max_distance
        """

        open_points = self._init_open_points(seed)
        open_points_dict = self._init_open_points_dict(seed)
        closed_points = self._init_closed_points()
        while not open_points.empty():
            item = open_points.get()
            distance = item[0]
            if distance > self._max_distance:
                continue
            point = np.int64(item[1])
            del open_points_dict[point]
            if point in closed_points:
                continue
            closed_points.add(point)
            neighbors_distances, neighbors = self._kdtree.query(self._X[point, :], self._n_neighbors)
            for i in xrange(len(neighbors)):
                neighbor = neighbors[i]
                neighbor_distance = neighbors_distances[i]
                if neighbor not in closed_points:
                    if neighbor in open_points_dict:
                        queue_priority = open_points_dict[neighbor]
                        test_priority = neighbor_distance + distance
                        if test_priority < queue_priority:
                            index = np.where(np.array(open_points.queue)[:, 1] == neighbor)[0][0]
                            open_points.queue[index][0] = test_priority
                            open_points_dict[neighbor] = test_priority
                            if (index - 1) >= 0 and (open_points.queue[index - 1][0] > test_priority):
                                open_points.queue.sort()
                    else:
                        priority = neighbor_distance + distance
                        item = [priority, neighbor]
                        open_points.put(item)
                        open_points_dict[neighbor] = priority
        return closed_points

    def patch_data(self):
        """
        Method that wraps the indexes of the points belonging to the patch in a numpy array.

        Returns
        -------

        closed_points : numpy array
            holding the indexes of the points belonging to the patch;
            can be used for fancy indexing
        """

        patch = self.visit_results(self._seed)
        return np.array(list(patch))
