"""
Invoke this script to compute reconstruction distances for a set of patches.
"""

import sys
import numpy as np
from PatchMaker import PatchMaker
import graph_utils
from IsomapEmbedder import IsomapEmbedder
from EuclideanMetric import EuclideanMetric
from GaussianKernel import GaussianKernel
from Lowess import Lowess
from SurfaceCleaner import SurfaceCleaner
import pandas as pd
import utils
import parameters


def usage():
    print "python " + sys.argv[0] + " data_file outdir max_distance n_neighbors seeds_folder sigma debug"
    print "data_file: path to the markers file"
    print "outdir: where to save patches markers files"
    print "max_distance: maximum geodesic radius for creating patches with uniform cost search"
    print "n_neighbors: nearest neighbors used to build the nearest neighbors graph of the data"
    print "seeds_folder: path to a folder containing seeds to start uniform cost search"
    print "sigma: sigma parameter of the gaussian kernel used in lowess regression"
    print "debug: decides if the script saves csv files for CloudCompare"

if len(sys.argv) != 8:
    usage()
    sys.exit(1)

data_file = sys.argv[1]
outdir = utils.add_trailing_slash(sys.argv[2])

utils.make_dir(outdir)

data_frame = pd.read_csv(data_file)

max_distance = float(sys.argv[3])
n_neighbors = int(sys.argv[4])
seeds_folder = utils.add_trailing_slash(sys.argv[5])
sigma = float(sys.argv[6])
debug = int(sys.argv[7])

if debug:
    outdir_embeddings = outdir + 'embeddings/'
    outdir_reconstructions = outdir + 'reconstructions/'
    outdir_csvs = outdir + 'csv_patches/'
    utils.make_dir(outdir_embeddings)
    utils.make_dir(outdir_reconstructions)
    utils.make_dir(outdir_csvs)

seeds = utils.get_filenames(seeds_folder)

for seed in seeds:
    X = data_frame.as_matrix([parameters.x_col, parameters.y_col, parameters.z_col])
    name = data_frame[parameters.name_col]

    patch_maker = PatchMaker(X, int(seed), n_neighbors, max_distance)
    patch = patch_maker.patch_data()

    if len(patch) == 1:
        print "There is one point in patch from seed " + seed + " with geodesic radius " + repr(max_distance)
        print "Most likely a false positive, skipping..."
        continue

    data_frame_patch = data_frame[data_frame.index.isin(patch)]

    X_patch = data_frame_patch.as_matrix([parameters.x_col, parameters.y_col, parameters.z_col])
    name_patch = data_frame_patch[parameters.name_col]

    n_neighbors_patch = graph_utils.compute_minimum_nearest_neighbors(X_patch)

    iso = IsomapEmbedder(n_neighbors_patch)
    try:
        points_2d = iso.compute(X_patch)
    except ValueError:
        print "Processing seed " + seed + "..."
        print "Got a strange ValueError due to sparse representation, skipping the patch..."
        continue

    metric = EuclideanMetric()
    kernel = GaussianKernel(sigma, metric)

    weights = kernel.compute_multiple(points_2d)
    np.fill_diagonal(weights, 0)

    low = Lowess(metric, parameters.robust_iter)

    points_3d_rebuilt = low.fit_transform(points_2d, X_patch, weights)

    surface_cleaner = SurfaceCleaner(metric)
    surface_distance_penalty = surface_cleaner.compute_distances(X_patch, points_3d_rebuilt)

    data_frame_patch[parameters.distance_col] = surface_distance_penalty
    data_frame_patch.to_csv(outdir + seed + '.marker', index=False)

    if debug:
        embed_df = pd.DataFrame(data={parameters.x_col: points_2d[:, 0],
                                      parameters.y_col: points_2d[:, 1],
                                      parameters.distance_col: surface_distance_penalty})
        embed_df.to_csv(outdir_embeddings + seed + '.csv', index=False)

        rebuild_df = pd.DataFrame(data={parameters.x_col: points_3d_rebuilt[:, 0],
                                        parameters.y_col: points_3d_rebuilt[:, 1],
                                        parameters.z_col: points_3d_rebuilt[:, 2],
                                        parameters.distance_col: surface_distance_penalty})
        rebuild_df.to_csv(outdir_reconstructions + seed + '.csv', index=False)

        data_frame_patch.to_csv(outdir_csvs + seed + '.csv', cols=[parameters.x_col,
                                                                   parameters.y_col,
                                                                   parameters.z_col,
                                                                   parameters.distance_col], index=False)
