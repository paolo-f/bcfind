#!/usr/bin/env python

"""
This script executes a rigid transformation to a 3D tensor

"""
__author__ = 'paciscopi'

import numpy as np
import os
import sys
from PIL import Image
import glob
from pyparsing import Word, alphanums,nums, restOfLine, Suppress, Group, Combine, Optional
import timeit
import math
from itertools import combinations, chain
from scipy.misc import comb
from scipy import special
import cv2

import shutil

import tables
import argparse
from progressbar import *

from bcfind.volume import *
from bcfind.semadec import imtensor

def parse_transformation_file(args):

    scientific_notation = "e-" + Word(nums)
    float_number = Combine(Optional("-") + Word(nums) + Optional("." + Word(nums) + Optional(scientific_notation) ))
    row_matrix =  Group(float_number + float_number + float_number)
    comma = Suppress(",")
    grammar = ('#' + restOfLine).suppress() | Word(alphanums + "_" + alphanums) + comma + row_matrix + comma + row_matrix + comma + row_matrix + comma + row_matrix + restOfLine.suppress()

    try:
        f = open(args.log_file, 'r')
    except IOError:
        print('file not existing')
        sys.exit(1)

    line = f.readline()
    parsed_line = grammar.parseString(line)
    R = np.zeros((3,3))
    R[0, :]=np.array(parsed_line[1][:]).astype(np.float)
    R[1, :]=np.array(parsed_line[2][:]).astype(np.float)
    R[2, :]=np.array(parsed_line[3][:]).astype(np.float)
    t=np.array(parsed_line[4][:]).astype(np.float)

    return R, t


def transform_substack(args, R, t):


    # Create the output directory if it doesn't exist
    if not os.path.exists(args.outdir):
                os.makedirs(args.outdir)


    ss = SubStack(args.indir, args.substack_id)

    input_stack_file = args.tensorimage

    hf5 = tables.openFile(input_stack_file, 'r')

    full_D, full_H, full_W = hf5.root.full_image.shape

    X0,Y0,Z0 = ss.info['X0'], ss.info['Y0'], ss.info['Z0']
    origin = (Z0, Y0, X0)
    H,W,D = ss.info['Height'], ss.info['Width'], ss.info['Depth']
    or_ss_shape = (D,H,W)
    #print('Loading data for substack', ss.substack_id)
    offset_W=int((3**(1/2.0)*W + W/2.0 - W)/2.0)
    offset_H=int((3**(1/2.0)*H + H/2.0 - H)/2.0)
    offset_D=int((3**(1/2.0)*D + D/2.0 - D)/2.0)


    if offset_W < args.extramargin:
        offset_W = args.extramargin
    if offset_H < args.extramargin:
        offset_H = args.extramargin
    if offset_D < args.extramargin:
        offset_D = args.extramargin

    offset_D_left = offset_D if int(origin[0] - offset_D) > 0 else origin[0]
    offset_H_left = offset_H if int(origin[1] - offset_H) > 0 else origin[1]
    offset_W_left = offset_W if int(origin[2] - offset_W) > 0 else origin[2]
    offset_D_right = offset_D if int(origin[0] + or_ss_shape[0] + offset_D) <= full_D else full_D - (origin[0] + or_ss_shape[0])
    offset_H_right = offset_H if int(origin[1] + or_ss_shape[1] + offset_H) <= full_H else full_H - (origin[1] + or_ss_shape[1])
    offset_W_right = offset_W if int(origin[2] + or_ss_shape[2] + offset_W) <= full_W else full_W - (origin[2] + or_ss_shape[2])



    pixels_input = hf5.root.full_image[origin[0] - offset_D_left:origin[0] + or_ss_shape[0] + offset_D_right,
                                       origin[1] - offset_H_left:origin[1] + or_ss_shape[1] + offset_H_right,
                                       origin[2] - offset_W_left:origin[2] + or_ss_shape[2] + offset_W_right]


    exmar_D_left = 0 if offset_D_left == origin[0] else args.extramargin
    exmar_H_left  = 0 if offset_H_left == origin[1] else args.extramargin
    exmar_W_left  = 0 if offset_W_left == origin[2] else args.extramargin

    depth_target, height_target, width_target = or_ss_shape[0] + 2 * args.extramargin, or_ss_shape[1] + 2 * args.extramargin, or_ss_shape[2] + 2 * args.extramargin #new
    depth_input, height_input, width_input = pixels_input.shape[0], pixels_input.shape[1],  pixels_input.shape[2]
    pixels_transformed_input = np.zeros((depth_target,height_target,width_target), dtype=np.uint8)



    total_start = timeit.default_timer()

    coords_2d_target = np.vstack(np.indices((width_target,height_target)).swapaxes(0,2).swapaxes(0,1))
    invR = R.T

    if args.invert:
        t = -np.dot(invR, t)
        invR = R

    invR_2d_transpose = np.transpose(np.dot(invR[:, 0:2], np.transpose(coords_2d_target - t[0:2])))

    offset_coords = np.array([[offset_W_left - exmar_W_left, offset_H_left - exmar_H_left, offset_D_left - exmar_D_left]]*invR_2d_transpose.shape[0])#new


    for z in xrange(0, depth_target, 1):
        R_t_3d = np.transpose(invR_2d_transpose + invR[:, 2] * (z - t[2]) + offset_coords)
        good_indices = np.array(range(R_t_3d.shape[1]))
        good_indices = good_indices[(R_t_3d[0, :] > 0) * (R_t_3d[1, :] > 0) * (R_t_3d[2, :] > 0) * (R_t_3d[0, :] < (width_input - 1)) * (R_t_3d[1, :] < (height_input - 1)) * (R_t_3d[2, :] < (depth_input - 1))]

        R_t_3d = R_t_3d.take(good_indices,axis=1)
        R_t_3d = np.round(R_t_3d).astype(int)
        coords_2d_target_tmp = coords_2d_target.take(good_indices, axis=0)


        coords_3d_target_tmp = np.hstack((coords_2d_target_tmp, np.ones((coords_2d_target_tmp.shape[0], 1)).astype(int)*z))


        pixels_transformed_input[coords_3d_target_tmp[:, 2], coords_3d_target_tmp[:, 1], coords_3d_target_tmp[:, 0]] = pixels_input[R_t_3d[2, :], R_t_3d[1, :], R_t_3d[0, :]]



    if args.save_tiff:
        minz = int(ss.info['Files'][0].split("/")[-1].split('_')[-1].split('.tif')[0])
        _prefix = '_'.join(ss.info['Files'][0].split("/")[-1].split('_')[0:-1])+'_'
        substack_outdir = args.outdir + '/' + args.substack_id
        if os.path.exists(substack_outdir):
            shutil.rmtree(substack_outdir)
        mkdir_p(substack_outdir)
        out_tensor = np.array(pixels_transformed_input, dtype=np.uint8)
        out_tensor = out_tensor[args.extramargin:depth_target-args.extramargin,args.extramargin:height_target-args.extramargin,args.extramargin:width_target-args.extramargin]
        imtensor.save_tensor_as_tif(out_tensor, substack_outdir, minz, prefix=_prefix)

    total_stop = timeit.default_timer()
    print ("total time transformation stack:%s "%(str(total_stop - total_start)))

    if args.get_tensor:
        return np.array(pixels_transformed_input, dtype=np.uint8)
    else:
        target_shape = (depth_target, height_target, width_target)
        atom = tables.UInt8Atom()
        h5f = tables.openFile(args.outdir + '/' + ss.substack_id + '.h5', 'w')
        ca = h5f.createCArray(h5f.root, 'full_image', atom, target_shape)
        for z in xrange(0, depth_target, 1):
            ca[z, :, :] = pixels_transformed_input[z,:,:]
        h5f.close()





def get_parser():
    parser = argparse.ArgumentParser(description="""
    This script applies a rigid transformation to a 3D tensor
    """, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('indir', metavar='indir', type=str,
                        help='needs indir/info.json, substacks, e.g. indir/000, and ground truth, e.g. indir/000-GT.marker')
    parser.add_argument('tensorimage', metavar='tensorimage', type=str,
                        help='path to the tensor image .h5 file')
    parser.add_argument('log_file', metavar='log_file', type=str,
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
    parser.add_argument('--get_tensor',dest='get_tensor', action='store_true', help='get the transformed tensor')
    return parser



def main(args):
    R, t = parse_transformation_file(args)
    transform_substack(args, R, t)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)


