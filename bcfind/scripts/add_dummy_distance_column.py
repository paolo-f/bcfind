"""
Invoke this script to add a distance column, filled with zeros, to an existing markers file
"""

import pandas as pd
import numpy as np
import sys
import parameters
import argparse


parser = argparse.ArgumentParser(description="add a distance column, filled with zeros, to an existing markers file")
parser.add_argument("infile", help="input markers file without the distance column")
parser.add_argument("outfile", help="output markers file with a column named distance with all zero values added")
args = parser.parse_args()

infile = args.infile
outfile = args.outfile

df = pd.read_csv(infile)
df[parameters.distance_col] = np.zeros((df.shape[0], ))
df.to_csv(outfile, index=False)
