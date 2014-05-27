"""
Invoke this script to add a distance column, filled with zeros, to an existing markers file
"""

import pandas as pd
import numpy as np
import sys
import manifold.parameters
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("infile", help="input markers file without the distance column")
    parser.add_argument("outfile", help="output markers file with a column named distance with all zero values added")

    return parser


def main(args):
    infile = args.infile
    outfile = args.outfile
    
    df = pd.read_csv(infile)
    df[parameters.distance_col] = np.zeros((df.shape[0], ))
    df.to_csv(outfile, index=False)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
