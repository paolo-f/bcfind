import numpy as np
import networkx as nx
from . import parameters


class ClassificationScores:
    """
    Class that evaluates performances of a group of markers.
    The normal use is with GT and predictions from a substack.
    Performances are evaluated by solving a Max Weight Matching problem.
    Markers not inside the margin of a substack are discarded in the evaluation process.

    Parameters
    ----------

    X_true : numpy array with shape (n_samples, n_dimensions)
        holds coordinates of ground truth markers
    X_pred : numpy array with shape (n_samples, n_dimensions) 
        holds prediction markers
    metric : object
        an EuclideanMetric() object to compute distances
    margin : int
        number of margin voxels not to consider when evaluating performances
    width : int
        width of the substack in voxel
    height : int
        height of the substack in voxel
    depth: int
        depth of the substack in voxel
    """

    def __init__(self, X_true, X_pred, metric, margin, width, height, depth):
        self._X_true = X_true
        self._X_pred = X_pred
        self._purkinje_radius = parameters.purkinje_radius
        self._metric = metric
        self._margin = margin
        self._width = width
        self._height = height
        self._depth = depth
        self._X = np.vstack((self._X_true, self._X_pred))
        self._distance_matrix = self._metric.compute_pairwise(self._X)
        self._graph, self._mate = self._max_weight_matching()

    def _max_weight_matching(self):
        """
        Solves the Max Weight Matching problem for evaluating performances.
        """

        graph = nx.Graph()
        for count in xrange(self._X_true.shape[0]):
            node = 'true_' + repr(count)
            graph.add_node(node, coord=self._X_true[count, :])
        for count in xrange(self._X_pred.shape[0]):
            node = 'pred_' + repr(count)
            graph.add_node(node, coord=self._X_pred[count, :])
        for ni in [n for n in graph.nodes() if n[0] == 't']:
            for nj in [n for n in graph.nodes() if n[0] == 'p']:
                row = int(ni[5:])
                column = self._X_true.shape[0] + int(nj[5:])
                distance = self._distance_matrix[row, column]
                if distance < 2 * self._purkinje_radius:
                    w = 1.0 / np.maximum(0.001, distance)
                    graph.add_edge(ni, nj, weight=w)
        mate = nx.algorithms.matching.max_weight_matching(graph, maxcardinality=False)
        return graph, mate

    def graph_based_performances(self):
        """
        Returns the performances of the predictions.

        Returns
        -------

        precision : float
            precision value
        recall : float
            recall value
        f1 : float
            f1 value
        tp_inside : int
            number of true positives inside the margin
        fn_inside : int
            number of false negatives inside the margin
        fp_inside : int
            number of false positives inside the margin
        tp_pred_in : set
            indices of true positives inside the margin
        fp_pred_in : set
            indices of false positives inside the margin
        """

        true_positives_true = set()
        true_positives_pred = set()
        tp_inside = 0
        fn_inside = 0
        fp_inside = 0
        for k1, k2 in self._mate.iteritems():
            row = int(k1[5:])
            column = self._X_true.shape[0] + int(k2[5:])
            if k1[0] == 'p':
                continue
            distance = self._distance_matrix[row, column]
            if 2 * distance < self._purkinje_radius:
                true_positives_true.add(int(k1[5:]))
                true_positives_pred.add(int(k2[5:]))
                if self.is_inside(self._graph.node[k1]['coord'],
                                  self._margin,
                                  self._width,
                                  self._height,
                                  self._depth) is True:
                    tp_inside += 1
        for ind in xrange(self._X_true.shape[0]):
            if self.is_inside(self._X_true[ind, :], self._margin, self._width, self._height, self._depth):
                if ind not in true_positives_true:
                    fn_inside += 1
        tp_pred_in = set()
        fp_pred_in = set()
        for ind in xrange(self._X_pred.shape[0]):
            if self.is_inside(self._X_pred[ind, :], self._margin, self._width, self._height, self._depth):
                if ind not in true_positives_pred:
                    fp_inside += 1
                    fp_pred_in.add(ind)
                else:
                    tp_pred_in.add(ind)
        precision = self.precision(tp_inside, fp_inside)
        recall = self.recall(tp_inside, fn_inside)
        f1 = self.f1(precision, recall)
        return precision, recall, f1, tp_inside, fn_inside, fp_inside, tp_pred_in, fp_pred_in

    @staticmethod
    def is_inside(point, margin, width, height, depth):
        """
        Static Method: returns True if point is inside margin, False otherwise

        Parameters
        ----------

        point : numpy array
            the point to test
        margin : int 
            number of margin voxels not to consider when evaluating performances
        width : int
            width of the substack in voxel
        height : int
            height of the substack in voxel
        depth : int
            depth of the substack in voxel
        """

        m = margin / 2
        if point[0] < m \
                or point[1] < m \
                or point[2] < m \
                or point[0] > width - m \
                or point[1] > height - m \
                or point[2] > depth - m:
            return False
        return True

    @staticmethod
    def precision(tp, fp):
        """
        Static method: computes precision

        Parameters
        ----------

        tp : int
            number of true positives
        fp : int
            number of false positives

        Returns
        -------

        precision : float
            precision computed with given values
        """

        if tp > 0:
            return float(tp) / (float(tp) + float(fp))
        else:
            return 0

    @staticmethod
    def recall(tp, fn):
        """
        Static method: computes recall

        Parameters
        ----------
        tp : int
            number of true positives
        fn : int
            number of false negatives

        Returns
        -------

        recall : float
            recall computed with given values
        """

        if tp > 0:
            return float(tp) / (float(tp) + float(fn))
        else:
            return 0

    @staticmethod
    def f1(precision, recall):
        """
        Static method: computes F1 score

        Parameters
        ----------

        precision : float
            the precision value
        recall : float
            the recall value

        Returns
        -------

        f1 : float
            F1 score computed with given values
        """
        if not (precision == 0 and recall == 0):
            return 2 * (precision * recall) / (precision + recall)
        else:
            return 0
