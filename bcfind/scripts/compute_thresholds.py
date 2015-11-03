#!/usr/bin/env python
"""
Scripts that computes binarization threshold using the maximum entropy approach (Kapur et al. 1985)
"""

from __future__ import print_function
import time
import datetime
import platform
import argparse
from bcfind.utils import mkdir_p
from bcfind.log import tee
from bcfind import mscd
from bcfind import threshold
from bcfind import volume
from bcfind.volume import SubStack,Center
import numpy as np
from os.path import basename


def main(args):
    substack = SubStack(args.indir,args.substack_id)
    patch = substack.get_volume()
    histogram = np.histogram(patch, bins=256,range=(0,256))[0]
    thresholds = threshold.multi_kapur(histogram, 2)
    outfile=args.outdir+'/'+args.substack_id+'/'+basename(args.indir)
    mkdir_p(args.outdir+'/'+args.substack_id)
    f=open(outfile,'w')
    f.write(str(thresholds[0])+','+str(thresholds[1])+'\n')
    f.close()
    

def get_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('indir', metavar='indir', type=str,
                        help="""Directory contaning the collection of substacks, e.g. indir/100905 etc.
                        Should also contain indir/info.plist""")
    parser.add_argument('substack_id', metavar='substack_id', type=str,
                        help='Substack identifier, e.g. 100905')
    parser.add_argument('outdir', metavar='outdir', type=str,
                        help="""Directory where prediction results will be saved, e.g. outdir/100905/ms.marker.
                        Will be created or overwritten""")
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
