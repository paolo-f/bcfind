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
from bcfind.scripts import transform_views
from fuse_tensors import fuse_tensors


def main(args):

    total_start = timeit.default_timer()
    print('Starting transformation and fusion of views of volume %s ...'%(args.substack_id))

    ss = SubStack(args.first_view_dir, args.substack_id)
    minz = int(ss.info['Files'][0].split("/")[-1].split('_')[-1].split('.tif')[0])
    prefix = '_'.join(ss.info['Files'][0].split("/")[-1].split('_')[0:-1])+'_'

    np_tensor_3d_first_view,_  = imtensor.load_nearby(args.tensorimage_first_view, ss, 0)

    args_transf=argparse.Namespace()
    args_transf.indir=args.second_view_dir
    args_transf.tensorimage=args.tensorimage_second_view
    args_transf.log_file=args.log_file
    args_transf.outdir=''
    args_transf.substack_id=args.substack_id
    args_transf.extramargin=0
    args_transf.invert=True
    args_transf.save_tiff=False
    args_transf.get_tensor=True
    R, t = transform_views.parse_transformation_file(args_transf)
    np_tensor_3d_second_view = transform_views.transform_substack(args_transf,R,t)

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
    parser.add_argument('log_file', metavar='log_file', type=str,
                        help='Transformation log file')
    parser.add_argument('outdir', metavar='outdir', type=str,
                        help='where fused rgb tensor will be saved')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
