"""
Invoke this script to delete all points above the given threshold in a markers file
"""

import sys
import pandas as pd
import manifold.utils
import manifold.parameters
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("data_file", help="path to the markers file with saved reconstruction distances")
    parser.add_argument("outdir", help="where to save filtered markers file")
    parser.add_argument("threshold", type=float, help="float value used to delete points with reconstruction distances > threshold")

    return parser


def main(args):
    data_file = args.data_file
    outdir = utils.add_trailing_slash(args.outdir)
    hreshold = args.threshold
    
    utils.make_dir(outdir)
    
    data_frame = pd.read_csv(data_file)
    filtered_data_frame = data_frame[data_frame[parameters.distance_col] <= threshold].drop(parameters.distance_col, axis=1)
    filtered_data_frame.to_csv(outdir + 'threshold_' + repr(threshold) + '.marker', index=False)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
