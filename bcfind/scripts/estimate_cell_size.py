#!/usr/bin/env python
"""
Estimates the size of somata in a substack, saving results into an apo file
"""
from __future__ import print_function
import numpy as np
import argparse
from bcfind import volume
from bcfind.utils import Struct
from bcfind.semadec import cellsize as cs

np.set_printoptions(linewidth=160, precision=4, suppress=True, threshold=5000)


def main(args):
    substack = volume.SubStack(args.indir, args.substack_id)

    substack.load_volume()
    centers = substack.load_markers(args.indir + '/' + args.substack_id + '-GT.marker', from_vaa3d=True)

    radii = np.arange(0.8, 2.5, step=0.1)
    box = Struct(an_x=args.an_x, an_y=args.an_y, an_z=args.an_z, size=args.size, shift=args.shift)
    correlations = cs.estimate_sizes(radii, centers, substack, box, args.debug_images)
    df, radiimax, radiiassigned = cs.make_dataframe(radii, correlations, centers, args)
    outputfile = args.indir + '/' + args.substack_id + '-size.apo'
    cs.save_apo_file(df, outputfile)
    if args.plot_correlations:
        cs.plot_correlations(radii, correlations, centers, radiimax, radiiassigned, args)


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('indir', metavar='indir', type=str,
                        help="""Directory contaning the collection of substacks, e.g. indir/100905 etc.
                        Should also contain indir/info.plist""")
    parser.add_argument('substack_id', metavar='substack_id', type=str,
                        help='Substack identifier, e.g. 100905')
    parser.add_argument('-s', '--size', metavar='s', dest='size',
                        action='store', type=float, default=3.0,
                        help='A box of size X(2s+1) * Z(2s+1) * Z(2s+1) around each marker will be analized')
    parser.add_argument('-S', '--shift', metavar='S', dest='shift',
                        action='store', type=int, default=2,
                        help='The best correlation is sought shifting the box from -S to S along all axes')
    parser.add_argument('-X', '--x-anisotropy', metavar='X', dest='an_x',
                        action='store', type=float, default=1.0,
                        help='Anisotropy along the X axis')
    parser.add_argument('-Y', '--y-anisotropy', metavar='Y', dest='an_y',
                        action='store', type=float, default=1.0,
                        help='Anisotropy along the Y axis')
    parser.add_argument('-Z', '--z-anisotropy', metavar='Z', dest='an_z',
                        action='store', type=float, default=1.0,
                        help='Anisotropy along the Z axis')
    parser.add_argument('-g', '--gain', metavar='gain', dest='gain',
                        action='store', type=float, default=0.01,
                        help='gain parameter when computing the radius with max correlation')
    parser.add_argument('-C', '--min-correlation', metavar='mincorr', dest='mincorr',
                        action='store', type=float, default=0.6,
                        help='Fallback to defaultr if correlation for a certain marker is below this value')
    parser.add_argument('-R', '--default-radius', metavar='defaultr', dest='defaultr',
                        action='store', type=float, default=1.2,
                        help='Default radius when correlation is below mincorr')
    parser.add_argument('-D', '--debug_images', dest='debug_images', action='store_true',
                        help='Save debugging images for visual inspection.')
    parser.set_defaults(save_images=False)
    parser.add_argument('-P', '--plot_correlations', dest='plot_correlations', action='store_true',
                        help='Plot correlation graphs.')
    parser.set_defaults(plot_correlations=False)
    return parser


# import cProfile
# import pstats
# parser = get_parser()
# args = parser.parse_args()

# substack = volume.SubStack(args.indir, args.substack_id)
# substack.load_volume()
# tensor = tensor_from_substack(substack)
# histogram = np.histogram(tensor,bins=256,range=(0,256))[0]
# cProfile.run('two_kapur(histogram)', 'kapur.stats')
# stats = pstats.Stats('kapur.stats')
# stats.sort_stats('cumulative')
# stats.print_stats()

# cProfile.run('main(args)', 'run.stats')
# stats = pstats.Stats('run.stats')
# stats.sort_stats('cumulative')
# stats.print_stats()

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
