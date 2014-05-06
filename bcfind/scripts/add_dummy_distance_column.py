"""
Invoke this script to add a distance column, filled with zeros, to an existing markers file
"""

import pandas as pd
import numpy as np
import sys
import parameters


def usage():
    print "python " + sys.argv[0] + " infile outfile"
    print "infile: markers file without the distance column"
    print "outfile: markers file with a column named distance with all zero values added"

if len(sys.argv) != 3:
    usage()
    sys.exit(1)

infile = sys.argv[1]
outfile = sys.argv[2]

df = pd.read_csv(infile)
df[parameters.distance_col] = np.zeros((df.shape[0], ))
df.to_csv(outfile, index=False)
