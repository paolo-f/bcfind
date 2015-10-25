#!/usr/bin/env python
"""
Scripts that computs the rigid transformation between two 3D point clouds and then rotates a 3D input tensor to a reference volume
"""

import numpy as np
import timeit
import argparse

from clsm_registration.estimate_registration import * 



def main(args):
    total_start= timeit.default_timer()
    R, t  = findBestRigidBodyEstimation(args.markers_input,args.markers_target)
    print('R: ', R)
    print('t: ', t)
    total_stop = timeit.default_timer()
    print "Rigid Transformation estimated in %s secs."%(str(timeit.default_timer() - total_start))
    if args.transform:
	total_start= timeit.default_timer()
        transform_tensor(args.input_view, args.target_view, args.transformed_view, R, t)
	print "Rigid Transformation applied to the input tensor in %s secs."%(str(timeit.default_timer() - total_start))



def get_parser():
    parser = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_view', metavar='input_view', type=str,
                        help='tiff folder or hdf5 file of the input tensor')
    parser.add_argument('target_view', metavar='target_view', type=str,
                        help='tiff folder or hdf5 file  of the reference tensor')
    parser.add_argument('transformed_view', metavar='transformed_view', type=str,
                        help='tiff folder or hdf5 file( if it ends with .h5 extension) of the transformed tensor')
    parser.add_argument('markers_input', metavar='markers_input', type=str,
                        help='markers of the input tensor')
    parser.add_argument('markers_target', metavar='markers_target', type=str,
                        help='markers of the reference tensor')
    parser.add_argument('--transform', dest='transform', action='store_true',
                        help='transform the input tensor according to the estimated rigid transformation.')
    parser.set_defaults(transform=False)

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
