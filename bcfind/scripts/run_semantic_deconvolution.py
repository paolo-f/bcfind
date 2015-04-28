#!/usr/bin/env python
"""
Script that creates a training set for semantic deconvolution.
"""
from __future__ import print_function
import argparse
import cPickle as pickle
import tables
import numpy as np

from bcfind.volume import SubStack
from bcfind.semadec import imtensor
from bcfind.semadec import deconvolver


def main(args):
    substack = SubStack(args.indir,args.substack_id)
    np_tensor_3d, minz = imtensor.load_nearby(args.tensorimage, substack, args.extramargin)

    # Standardize volume according to mean and std found in the training set
    print('Reading standardization data from', args.trainfile)
    h5 = tables.openFile(args.trainfile)
    Xmean = h5.root.Xmean[:].astype(np.float32)
    Xstd = h5.root.Xstd[:].astype(np.float32)
    h5.close()
    print('Starting semantic devonvolution of volume', args.substack_id)
    model = pickle.load(open(args.model))
    minz = int(substack.info['Files'][0].split('full_')[1].split('.tif')[0])

    reconstruction = deconvolver.filter_volume(np_tensor_3d, Xmean, Xstd,
                                               args.extramargin, model, args.speedup, args.do_cython)

    imtensor.save_tensor_as_tif(reconstruction, args.outdir+'/'+args.substack_id, minz)


def get_parser():
    parser = argparse.ArgumentParser(description="""
    Preprocess a substack using a neural network model
    """, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('indir', metavar='indir', type=str,
                        help='needs indir/info.json, substacks, e.g. indir/000, and ground truth, e.g. indir/000-GT.marker')
    parser.add_argument('substack_id', metavar='substack_id', type=str,
                        help='substack identifier, e.g. 010608')
    parser.add_argument('tensorimage', metavar='tensorimage', type=str,
                        help='path to the tensor image .h5 file')
    parser.add_argument('model', metavar='model', type=str,
                        help='pickle file containing a trained network')
    parser.add_argument('trainfile', metavar='trainfile', type=str,
                        help='HDF5 file on which the network was trained (should contain mean/std arrays)')
    parser.add_argument('outdir', metavar='outdir', type=str,
                        help='where preprocessed volume will be saved')
    parser.add_argument('--extramargin', metavar='extramargin', dest='extramargin',
                        action='store', type=int, default=6,
                        help='Extra margin for convolution. Should be equal to (filter_size - 1)/2')
    parser.add_argument('--speedup', metavar='speedup', dest='speedup',
                        action='store', type=int, default=4,
                        help='convolution stride (isotropic along X,Y,Z)')
    parser.add_argument('--do_cython', dest='do_cython', action='store_true', help='use the compiled cython modules in deconvolver.py')
    parser.set_defaults(do_cython=False)
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)

