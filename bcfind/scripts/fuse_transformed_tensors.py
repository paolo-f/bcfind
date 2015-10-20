#!/usr/bin/env python
"""
Script that transforms an input tensor and fuses it with a reference volume
"""
from __future__ import print_function

import sys
import cPickle as pickle
import tables
import timeit
import numpy as np
import argparse

from bcfind.volume import SubStack
from bcfind.semadec import deconvolver
from bcfind.semadec import imtensor
from clsm_registration.rigid_transformation import * 


def main(args):

    total_start = timeit.default_timer()
    print('Starting transformation and fusion of views of volume %s ...'%(args.substack_id))

    ss = SubStack(args.first_view_dir, args.substack_id)
    minz = int(ss.info['Files'][0].split("/")[-1].split('_')[-1].split('.tif')[0])
    prefix = '_'.join(ss.info['Files'][0].split("/")[-1].split('_')[0:-1])+'_'

    np_tensor_3d_first_view,_  = imtensor.load_nearby(args.tensorimage_first_view, ss, 0)
    if args.transformation_file is not None:
	R, t = parse_transformation_file(args.transformation_file)
        np_tensor_3d_second_view = transform_substack(args.second_view_dir, args.tensorimage_second_view, args.substack_id, R, t, 0, invert=True)
    else:
        np_tensor_3d_second_view,_  = imtensor.load_nearby(args.tensorimage_second_view, ss, 0)
    fuse_tensors(args.outdir, np_tensor_3d_first_view,np_tensor_3d_second_view,np.zeros_like(np_tensor_3d_first_view).astype(np.uint8))
    print ("total time transformation and fusion: %s" %(str(timeit.default_timer() - total_start)))


def get_parser():
    parser = argparse.ArgumentParser(description="""
    Preprocess a substack using a neural network model
    """, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('first_view_dir', metavar='first_view_dir', type=str,
                        help='must contain indir/info.json, substacks, e.g. indir/000, e.g. indir/000-GT.marker')
    parser.add_argument('second_view_dir', metavar='second_view_dir', type=str,
                        help='must contain indir/info.json, substacks, e.g. indir/000, e.g. indir/000-GT.marker')
    parser.add_argument('tensorimage_first_view', metavar='tensorimage_first_view', type=str,
                        help='path to the tensor image .h5 file of the first view')
    parser.add_argument('tensorimage_second_view', metavar='tensorimage_second_view', type=str,
                        help='path to the tensor image .h5 file of the first view')
    parser.add_argument('substack_id', metavar='substack_id', type=str,
                        help='substack identifier, e.g. 010608')
    parser.add_argument('outdir', metavar='outdir', type=str,
                        help='where fused rgb tensor will be saved')
    parser.add_argument('--transformation_file', metavar='transformation_file', type=str,
                        help='Transformation log file')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
