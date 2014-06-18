#!/usr/bin/env python
"""
Invoke this script to compute reconstruction distances for a set of patches.
"""

import sys
import numpy as np
from manifold.PatchMaker import PatchMaker
import manifold.graph_utils as graph_utils
from manifold.IsomapEmbedder import IsomapEmbedder
from manifold.EuclideanMetric import EuclideanMetric
from manifold.GaussianKernel import GaussianKernel
from manifold.Lowess import Lowess
from manifold.SurfaceCleaner import SurfaceCleaner
import pandas as pd
import manifold.utils as utils
import manifold.parameters as parameters
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("data_file", help="path to the markers file")
    parser.add_argument("outdir", help="where to save patches markers files")
    parser.add_argument("max_distance", help="maximum geodesic radius for creating patches with uniform cost search", type=float)
    parser.add_argument("n_neighbors", help="nearest neighbors used to build the nearest neighbors graph of the data", type=int)
    parser.add_argument("seeds_folder", help="path to a folder containing seeds to start uniform cost search")
    parser.add_argument("sigma", help="sigma parameter of the gaussian kernel used in lowess regression", type=float)
    parser.add_argument("--debug", help="saves csv debug files", action="store_true")

    return parser


def main(args):
    data_file = args.data_file
    outdir = utils.add_trailing_slash(args.outdir)
    
    utils.make_dir(outdir)
    
    data_frame = pd.read_csv(data_file)
    
    max_distance = args.max_distance
    n_neighbors = args.n_neighbors
    seeds_folder = utils.add_trailing_slash(args.seeds_folder)
    sigma = args.sigma
    debug = args.debug
    
    if debug:
        outdir_embeddings = outdir + 'embeddings/'
        outdir_reconstructions = outdir + 'reconstructions/'
        outdir_csvs = outdir + 'csv_patches/'
        outdir_single_points = outdir + 'single_points/'
        outdir_faulty = outdir + 'faulty/'
        utils.make_dir(outdir_embeddings)
        utils.make_dir(outdir_reconstructions)
        utils.make_dir(outdir_csvs)
        utils.make_dir(outdir_single_points)
        utils.make_dir(outdir_faulty)
    
    seeds = utils.get_filenames(seeds_folder)
    
    for seed in seeds:
        X = data_frame.as_matrix([parameters.x_col, parameters.y_col, parameters.z_col])
        name = data_frame[parameters.name_col]
        
        patch_maker = PatchMaker(X, int(seed), n_neighbors, max_distance)
        patch = patch_maker.patch_data()
        
        if len(patch) == 1:
            print "There is one point in patch from seed " + seed + " with geodesic radius " + repr(max_distance)
            print "Most likely a false positive, skipping..."
            if debug:
                print "Saving patch with one point for debug purposes..."
                single_frame_patch = data_frame[data_frame.index.isin(patch)]
                single_frame_patch.to_csv(outdir_single_points + seed + '.csv', index=False)
            continue
        elif len(patch) == 2:
            print "There are two points in patch from seed " + seed + " with geodesic radius " + repr(max_distance)
            print "Most likely two false positives, skipping..."
            if debug:
                print "Saving patch with two points for debug purposes..."
                single_frame_patch = data_frame[data_frame.index.isin(patch)]
                single_frame_patch.to_csv(outdir_single_points + seed + '.csv', index=False)
            continue
        
        data_frame_patch = data_frame[data_frame.index.isin(patch)]
        
        X_patch = data_frame_patch.as_matrix([parameters.x_col, parameters.y_col, parameters.z_col])
        
        n_neighbors_patch = graph_utils.compute_minimum_nearest_neighbors(X_patch)
        
        iso = IsomapEmbedder(n_neighbors_patch)
        
        try:
            points_2d = iso.compute(X_patch)
        except ValueError:
            print "Processing seed " + seed + "..."
            print "Got a strange ValueError due to sparse representation, skipping the patch..."
            if debug:
                print "Saving faulty patch for debug purposes..."
                data_frame_patch.to_csv(outdir_faulty + seed + '.csv', index=False)
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
            
            data_frame_patch.to_csv(outdir_csvs + seed + '.csv', columns=[parameters.x_col,
                parameters.y_col,
                parameters.z_col,
                parameters.distance_col], index=False)

            
if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
