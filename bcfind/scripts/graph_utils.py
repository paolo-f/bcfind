"""
Various helpers functions to deal with graphs.
"""

from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import connected_components
from scipy.sparse import issparse
import numpy as np


def compute_adjacency_matrix(X, n_neighbors):
    """
    Computes the (sparse) adjacency matrix of a kNN graph of a dataset.

    Parameters

    X: numpy array of shape (n_samples, n_dimensions) holding the points of the dataset
    n_neighbors: number of nearest neighbors used to compute the kNN graph

    Returns

    adjacency_matrix: sparse representation of the adjacency matrix of the kNN graph
    """

    adjacency_matrix = kneighbors_graph(X, n_neighbors, 'connectivity')
    return adjacency_matrix

def compute_connected_components(adjacency_matrix):
    """
    Given an adjacency matrix of a graph, computes the number of connected components.

    Parameters

    adjacency_matrix: adjacency matrix of the graph

    Returns

    cc: number of connected components in the graph represented by the adjacency matrix
    """

    if issparse(adjacency_matrix):
        difference_matrix = adjacency_matrix - adjacency_matrix.transpose()
        is_symmetric_p = np.all(1e-10 > difference_matrix.data)
        is_symmetric_n = np.all(difference_matrix.data > -1e-10)
    else:
        difference_matrix = adjacency_matrix - adjacency_matrix.T
        is_symmetric_p = np.all(1e-10 > difference_matrix)
        is_symmetric_n = np.all(difference_matrix > -1e-10)
    if is_symmetric_p and is_symmetric_n:
        return connected_components(adjacency_matrix, False)
    else:
        return connected_components(adjacency_matrix, True)

def compute_minimum_nearest_neighbors(X):
    """
    Given a dataset, computes the minimum number of nearest neighbors to have a fully connected kNN graph.

    Parameters

    X: numpy array of shape (n_samples, n_dimensions) holding the points of the dataset

    Returns

    n_neighbors: minimum number of nearest neighbors to have a fully connected kNN graph.
    """

    n_neighbors = 1
    adjacency_matrix = compute_adjacency_matrix(X, n_neighbors)
    n_components, _ = compute_connected_components(adjacency_matrix)
    while n_components > 1:
        n_neighbors += 1
        adjacency_matrix = compute_adjacency_matrix(X, n_neighbors)
        n_components, _ = compute_connected_components(adjacency_matrix)
    return n_neighbors
