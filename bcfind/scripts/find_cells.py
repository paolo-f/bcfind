#!/usr/bin/env python
"""
Main script for finding cell soma using the mean shift algorithm
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


def main(args):
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    tee.log('find_cells.py running on',platform.node(),st)
    mkdir_p(args.outdir+'/'+args.substack_id)
    tee.logto('%s/%s/log.txt' % (args.outdir, args.substack_id))

    timers = [mscd.pca_analysis_timer, mscd.mean_shift_timer, mscd.ms_timer, mscd.patch_ms_timer]
    timers.extend([volume.save_vaa3d_timer, volume.save_markers_timer])
    timers.extend([threshold.multi_kapur_timer])
    for t in timers:
        t.reset()

    substack = volume.SubStack(args.indir, args.substack_id)

    substack.load_volume()
    if args.local:
        mscd.pms(substack, args)
    else:
        mscd.ms(substack, args)
    for t in timers:
        if t.n_calls > 0:
            tee.log(t)
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    tee.log('find_cells.py finished on',platform.node(),st)


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
    parser.add_argument('-l', '--local', dest='local', action='store_true',
                        help='Perform local processing by dividing the volume in 8 parts.')
    # parser.set_defaults(local=True)
    parser.add_argument('-r', '--hi_local_max_radius', metavar='r', dest='hi_local_max_radius',
                        action='store', type=float, default=6,
                        help='Radius of the seed selection ball (r)')
    parser.add_argument('-R', '--mean_shift_bandwidth', metavar='R', dest='mean_shift_bandwidth',
                        action='store', type=float, default=5.5,
                        help='Radius of the mean shift kernel (R)')
    parser.add_argument('--save_image', dest='save_image', action='store_true',
                        help='Save debugging substack for visual inspection (voxels above threshold and colorized clusters).')
    parser.add_argument('--floating_point', dest='floating_point', action='store_true',
                        help='If true, cell centers are saved in floating point.')
    parser.add_argument('--min_second_threshold', metavar='min_second_threshold', dest='min_second_threshold',
                        action='store', type=int, default=15,
                        help="""If the foreground (second threshold in multi-Kapur) is below this value
                        then the substack is too dark and assumed to contain no soma""")
    parser.set_defaults(save_image=False)
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
