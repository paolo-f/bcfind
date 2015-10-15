#!/usr/bin/env python

"""
This script executes a rigid transformation to a 3D tensor

"""
__author__ = 'paciscopi'

import numpy as np
import os
import sys
from pyparsing import Word, alphanums,nums, restOfLine, Suppress, Group, Combine, Optional
import argparse
from multiview.rigid_transformation import * 


def get_parser():
    parser = argparse.ArgumentParser(description="""
    This script applies a rigid transformation to a 3D tensor
    """, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('indir', metavar='indir', type=str,
                        help='needs indir/info.json, substacks, e.g. indir/000, and ground truth, e.g. indir/000-GT.marker')
    parser.add_argument('tensorimage', metavar='tensorimage', type=str,
                        help='path to the tensor image .h5 file')
    parser.add_argument('transformation_file', metavar='transformation_file', type=str,
                        help='File ascii that stores the estimated rigid transformation. It consists of a single line, formatted as follows:'
                        ' substack_id,R00,R01,R02,R10,R11,R12,R20,R21,R22,tx,ty,yz,comment')
    parser.add_argument('substack_id', metavar='substack_id', type=str,
                        help='Substack identifier, e.g. 100905')
    parser.add_argument('outdir', metavar='outdir', type=str,
                        help='folder where the transformed substack will be saved')
    parser.add_argument('--extramargin', metavar='extramargin', dest='extramargin',
                        action='store', type=int, default=6,
                        help='Extra margin for convolution. Should be equal to (filter_size - 1)/2')
    parser.add_argument('--invert', dest='invert', action='store_true', help='If it\'s false the transformation is applied considering indir as moving tensor. Otherwise indir is considered as reference and the transformation will be inverted   ')
    parser.add_argument('--savetiff', dest='save_tiff', action='store_true', help='save the transformed tiff images')
    return parser



def main(args):
    R, t = parse_transformation_file(args.transformation_file)
    transform_substack(args.indir, args.tensorimage, args.substack_id, R, t, args.extramargin, args.outdir, args.invert, args.save_tiff, save_hdf5=True)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)


