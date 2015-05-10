#!/usr/bin/env python
"""
Invoke this script to merge the results from each patch produced by the Manifold filter.
"""

import sys
import pandas as pd
import manifold.utils as utils
import manifold.parameters as parameters
import numpy as np
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("path", help="where all patches markers files were saved")
    parser.add_argument("outdir", help="where to save merged markers file with reconstruction distances")
    parser.add_argument("--debug", help="saves csv debug files", action="store_true")

    return parser


def main(args):
    path = utils.add_trailing_slash(args.path)
    outdir = utils.add_trailing_slash(args.outdir)
    debug = args.debug

    utils.make_dir(outdir)

    files = utils.get_filenames(path)

    data_frames_list = [pd.read_csv(path + f) for f in files]
    big_data_frame = pd.concat(data_frames_list)
    sorted_big_data_frame = big_data_frame.sort(parameters.distance_col)
    final_data_frame = sorted_big_data_frame.drop_duplicates(parameters.name_col)
    final_data_frame.to_csv(outdir + 'cleaned.marker', index=False)

    indices = np.setdiff1d(np.arange(len(sorted_big_data_frame)), np.unique(sorted_big_data_frame, return_index=True)[1])
    duplicates = np.unique(sorted_big_data_frame[indices])
    #sorted_big_data_frame.to_csv(outdir + 'cleaned.marker', index=False)

    if debug:
        debug_df = pd.read_csv(outdir + 'cleaned.marker')
        substacks = np.empty(len(debug_df.index))
        names = debug_df[parameters.name_col]
        lut = dict()
        count = 0
        for index in debug_df.index:
            substack = utils.extract_substack(names[index])
            if substack not in lut:
                count += 1
                lut[substack] = count
            substacks[index] = lut[substack]
        #debug_df[parameters.substack_col] = substacks
        #debug_df.to_csv(outdir + 'debug_cleaned.csv',
                #col=[parameters.x_col,
                    #parameters.y_col,
                    #parameters.z_col,
                    #parameters.density_col,
                    #parameters.distance_col,
                    #parameters.substack_col],
                #index=False)
        cols = debug_df.as_matrix([parameters.x_col, parameters.y_col, parameters.z_col,parameters.density_col,parameters.distance_col])
        cleaned_df = pd.DataFrame(data={parameters.x_col: cols[:, 0],
            parameters.y_col: cols[:, 1],
            parameters.z_col: cols[:, 2],
            parameters.density_col: cols[:, 3],
            parameters.distance_col: cols[:, 4]})
        cleaned_df.to_csv(outdir + 'debug_cleaned.csv', index=False)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
