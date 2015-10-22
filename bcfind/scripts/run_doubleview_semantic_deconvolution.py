#!/usr/bin/env python
"""
Script that creates a training set for semantic deconvolution.
"""

from __future__ import print_function
import sys
import cPickle as pickle
import tables
import timeit
import numpy as np
import argparse
from os.path import basename

from bcfind.volume import SubStack
from bcfind.semadec import deconvolver
from bcfind.semadec import imtensor
from clsm_registration.rigid_transformation import * 


def main(args):

    total_start = timeit.default_timer()
    print('Starting reconstruction of volume %s ...'%(args.substack_id))

    ss = SubStack(args.first_view_dir, args.substack_id)
    minz = int(ss.info['Files'][0].split("/")[-1].split('_')[-1].split('.tif')[0])
    prefix = '_'.join(ss.info['Files'][0].split("/")[-1].split('_')[0:-1])+'_'

    np_tensor_3d_first_view,_  = imtensor.load_nearby(args.tensorimage_first_view, ss, args.extramargin)
    if args.transformation_file is not None:
	R, t = parse_transformation_file(args.transformation_file)
        np_tensor_3d_second_view = transform_substack(args.second_view_dir, args.tensorimage_second_view, args.substack_id, R, t, args.extramargin, invert=True)
    else:
        np_tensor_3d_second_view,_  = imtensor.load_nearby(args.tensorimage_second_view, ss, args.extramargin)

    print('Loading model...')
    model = pickle.load(open(args.model))
    
    if not args.local_mean_std:
        h5 = tables.openFile(args.trainfile)
        Xmean = h5.root.Xmean[:].astype(np.float32)
        Xstd = h5.root.Xstd[:].astype(np.float32)
        h5.close()
    else:
        Xmean=None
        Xstd=None

    reconstruction = deconvolver.filter_volume([np_tensor_3d_first_view,np_tensor_3d_second_view], Xmean, Xstd,
                                               args.extramargin, model, args.speedup, do_cython=args.do_cython,trainfile=args.trainfile)

    pair_id = basename(args.first_view_dir)+ '_' +basename(args.second_view_dir)
    outdir=args.outdir+'/'+args.substack_id+'/'+pair_id

    imtensor.save_tensor_as_tif(reconstruction, outdir, minz, prefix='slice_')


    print ("total time reconstruction: %s" %(str(timeit.default_timer() - total_start)))


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
    parser.add_argument('model', metavar='model', type=str,
                        help='pickle file containing a trained network')
    parser.add_argument('trainfile', metavar='trainfile', type=str,
                        help='HDF5 file on which the network was trained (should contain mean/std arrays)')
    parser.add_argument('outdir', metavar='outdir', type=str,
                        help='where preprocessed volume will be saved')
    parser.add_argument('--transformation_file', metavar='transformation_file', type=str,
                        help='Transformation log file')
    parser.add_argument('--extramargin', dest='extramargin',
                        action='store', type=int, default=4,
                        help='Input and output patches are cubes of side (2*size+1)**3')
    parser.add_argument('--speedup', metavar='speedup', dest='speedup',
                        action='store', type=int, default=4,
                        help='convolution stride (isotropic along X,Y,Z)')
    parser.add_argument('--local_mean_std', dest='local_mean_std', action='store_true', help='use the compiled cython modules in deconvolver.py')
    parser.add_argument('--do_cython', dest='do_cython', action='store_true', help='use the compiled cython modules in deconvolver.py')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
