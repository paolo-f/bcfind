"""
Invoke this script to merge the results from every single patch produced by the Manifold filter.
"""

import sys
import pandas as pd
import utils
import parameters
import numpy as np


def usage():
    print "python " + sys.argv[0] + " path outdir debug"
    print "path: where all patches markers files where saved"
    print "outdir: where to save merged markers file with reconstruction distances"
    print "debug: decides if the script saves csv files for CloudCompare"

if len(sys.argv) != 4:
    usage()
    sys.exit(1)

path = utils.add_trailing_slash(sys.argv[1])
outdir = utils.add_trailing_slash(sys.argv[2])
debug = int(sys.argv[3])

utils.make_dir(outdir)

files = utils.get_filenames(path)

data_frames_list = [pd.read_csv(path + f) for f in files]
big_data_frame = pd.concat(data_frames_list)
sorted_big_data_frame = big_data_frame.sort(parameters.distance_col)
final_data_frame = sorted_big_data_frame.drop_duplicates(parameters.name_col)
final_data_frame.to_csv(outdir + 'cleaned.marker', index=False)

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
    debug_df[parameters.substack_col] = substacks
    debug_df.to_csv(outdir + 'debug_cleaned.csv', col=[parameters.x_col,
                                                               parameters.y_col,
                                                               parameters.z_col,
                                                               parameters.distance_col,
                                                               parameters.substack_col], index=False)
