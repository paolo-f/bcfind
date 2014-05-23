"""
Various general helpers functions.
"""

import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def add_trailing_slash(path):
    """
    Function that adds a trailing slash to a string (likely a path).

    Parameters

    path: path to which the trailing slash will be added

    Returns

    path: path with the trailing slash added
    """

    if path[-1] != '/':
        path += '/'
    return path


def remove_trailing_slash(path):
    """
    Function that removes a trailing slash to a string (likely a path).

    Parameters

    path: path to wahich the trailing slash will be removed

    Returns

    path: path with the trailing slash removed
    """

    if path[-1] == '/':
        path = path[:-1]
    return path


def make_dir(path):
    """
    Function that creates a directory given a path.

    Parameters

    path: path of the directory to create
    """

    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError:
            print path + ' already existing, skipping creation...'


def points_to_substack(points_matrix, name):
    """
    Function that groups point by substack

    Parameters

    points_matrix: numpy array of shape (n_samples, n_dimensions) representing the dataset
    name: numpy array of shape (n_samples, ) representing the IDs of the points;
            should be something like "MS_CENTER ??(??????)"

    Returns

    data_substacks: dictionary indexed with substacks IDs holding dataset matrices of each substack
    """

    if points_matrix.shape[0] != len(name):
        raise RuntimeError("Number of points and their names should have the same length!")
    data_substacks = dict()
    for index in xrange(len(name)):
        substack = extract_substack(name[index])
        if substack not in data_substacks:
            data_substacks[substack] = list()
        data_substacks[substack].append(points_matrix[index, :])
    return data_substacks


def data_to_substack(data_matrix, substacks):
    """
    Function that extracts and groups data by given substacks

    Parameters

    data_matrix: numpy array of shape (n_samples, n_data_dimensions) representing the data,
                    most likely a pandas.DataFrame.as_matrix(); last column must be
                    a substack column
    substacks: set of substacks for which the function should extract and group points

    Returns

    data_substacks: dictionary indexed with substacks IDs holding data matrices of each substack
    """

    data_substacks = dict()
    for index in xrange(data_matrix.shape[0]):
        substack = extract_substack(data_matrix[index, -1])
        if substack in substacks:
            if substack not in data_substacks:
                data_substacks[substack] = list()
            data_substacks[substack].append(data_matrix[index, :-1])
    return data_substacks


def extract_substack(name):
    """
    Function that extracts substack ID given a name like "MS_CENTER ??(??????)"

    Parameters

    name: a string which is an ID of a point, like "MS_CENTER ??(??????)"

    Returns

    substack: a string which is the substack ID extracted from name
    """

    temp = name.split(' ')
    temp = temp[1].split('(')
    substack = temp[1][:-1]
    return substack


def get_filenames(path):
    """
    Function that extracts all the files in a specified path.

    Parameters

    path: path in which lists the files

    Returns

    filenames: list with all the filenames in the given path
    """

    filenames = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    return filenames


def plot_curve(x, y, xlabel, ylabel, outdir):
    """
    Function that saves a *.png file of a plot. It automatically sorts the values for the abscissa
    and adjust the order of the values for the ordinate.

    Parameters

    x: numpy array of not necessarily sorted values for the abscissa
    y: numpy array of not necessarily sorted values for the ordinate
    xlabel: string which is the label for the abscissa
    ylabel: string which is the label for the ordinate
    outdir: string which represent a path to where the plot will be saved
    """

    indexes = np.argsort(x)
    ordered_x = np.zeros(len(x))
    ordered_y = np.zeros(len(y))
    for i in xrange(len(x)):
        ordered_x[i] = x[indexes[i]]
        ordered_y[i] = y[indexes[i]]
    plt.figure()
    plt.plot(ordered_x, ordered_y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(which='both', axis='both')
    plt.savefig(outdir + xlabel + '-' + ylabel)
