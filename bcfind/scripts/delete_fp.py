"""
Invoke this script to delete all points in a markers file above the given threshold
"""

import sys
import pandas as pd
import utils
import parameters


def usage():
    print "python " + sys.argv[0] + " data_file outdir threshold"
    print "data_file: path to the markers file with saved reconstruction distances"
    print "outdir: where to save filtered markers file"
    print "threshold: numeric value used to delete points with reconstruction distances > threshold"

if len(sys.argv) != 4:
    usage()
    sys.exit(1)

data_file = sys.argv[1]
outdir = utils.add_trailing_slash(sys.argv[2])
threshold = sys.argv[3]

utils.make_dir(outdir)

data_frame = pd.read_csv(data_file)
filtered_data_frame = data_frame[data_frame[parameters.distance_col] <= float(threshold)].drop(parameters.distance_col, axis=1)
filtered_data_frame.to_csv(outdir + 'threshold_' + threshold + '.marker', index=False)
