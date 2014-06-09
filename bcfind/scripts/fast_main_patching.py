#!/usr/bin/env python
"""
Invoke this script to find minimum number of nearest neighbors for a connected graph and to find seeds
for the subsequent step of creating patches of the data
"""

import sys
import numpy as np
import manifold.graph_utils as graph_utils
from scipy.spatial import cKDTree
import pandas as pd
import manifold.utils as utils
import manifold.parameters as parameters
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("data_file", help="path to the markers file")
    parser.add_argument("outdir", help="where to save minimum nearest neighbors and seeds")

    return parser


def main(args):
    data_file = args.data_file
    outdir = utils.add_trailing_slash(args.outdir)
    
    outdir_seeds = outdir + 'seeds/'
    outdir_nn = outdir + 'nn/'
    
    utils.make_dir(outdir)
    utils.make_dir(outdir_seeds)
    utils.make_dir(outdir_nn)
    
    for folder in xrange(parameters.jobs):
        utils.make_dir(outdir_seeds + repr(folder))
    
    data_frame = pd.read_csv(data_file)
    
    points_matrix = data_frame.as_matrix([parameters.x_col, parameters.y_col, parameters.z_col])
    name = data_frame[parameters.name_col]
    
    data_substacks = utils.points_to_substack(points_matrix, name)
    
    seeds = list()
    global_kdtree = cKDTree(points_matrix)
    for substack, data in data_substacks.iteritems():
        X = np.vstack(data)
        X = np.float64(X)
        kdtree = cKDTree(X)
        _, index = kdtree.query(np.mean(X, axis=0))
        _, centroid = global_kdtree.query(X[index, :])
        seeds.append(centroid)
    
    n_neighbors = graph_utils.compute_minimum_nearest_neighbors(points_matrix)
    
    with open(outdir_nn + repr(n_neighbors), 'w') as nn_file:
        nn_file.close()
    
    folder = 0
    while len(seeds) > 0:
        seed = seeds.pop()
        with open(outdir_seeds + repr(folder) + '/' + repr(seed), 'w') as seed_file:
            seed_file.close()
        folder = (folder + 1) % parameters.jobs


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
