#!/usr/bin/env python
"""
Scripts that fuse 3D greyscale tensors in a 3D RGB tensor
"""

import numpy as np
import os
from PIL import Image
import glob
import cv2
import argparse

from clsm_registration.rigid_transformation import fuse_tensors


def read_tensors(args):

    suffix = ".tif"
    convert_to_gray = True

    filesRedChannel = sorted([args.redChannel_dir + '/' + f for f in os.listdir(args.redChannel_dir) if f[0] != '.' and f.endswith(suffix)])
    img_z = np.asarray(Image.open(filesRedChannel[0]))
    height_redChannel, width_redChannel = img_z.shape
    depth_redChannel = len(filesRedChannel)
    pixels_redChannel = np.empty(shape=(depth_redChannel, height_redChannel, width_redChannel), dtype=np.uint8)

    for z, image_file in enumerate(filesRedChannel):
        img_z = Image.open(image_file)
        if convert_to_gray:
            img_z = img_z.convert('L')
        pixels_redChannel[z, :, :] = np.asarray(img_z)
    print('...read the first tensor (%s slices)' %z)


    filesGreenChannel = sorted([args.greenChannel_dir + '/' + f for f in os.listdir(args.greenChannel_dir) if f[0] != '.' and f.endswith(suffix)])
    img_z = np.asarray(Image.open(filesGreenChannel[0]))
    height_greenChannel, width_greenChannel = img_z.shape
    depth_greenChannel = len(filesGreenChannel)
    pixels_greenChannel = np.empty(shape=(depth_greenChannel, height_greenChannel, width_greenChannel), dtype=np.uint8)

    for z, image_file in enumerate(filesGreenChannel):
        img_z = Image.open(image_file)
        if convert_to_gray:
            img_z = img_z.convert('L')
        pixels_greenChannel[z, :, :] = np.asarray(img_z)
    print('...read the second tensor (%s slices)' %z)


    if not args.blueChannel_dir:
        pixels_blueChannel = np.zeros((depth_redChannel, height_redChannel, width_redChannel), dtype=np.uint8)
    else:
        filesBlueChannel = sorted([args.blueChannel_dir + '/' + f for f in os.listdir(args.blueChannel_dir) if f[0] != '.' and f.endswith(suffix)])
        img_z = np.asarray(Image.open(filesBlueChannel[0]))
        height_blueChannel, width_blueChannel = img_z.shape
        depth_blueChannel = len(filesBlueChannel)
        pixels_blueChannel = np.empty(shape=(depth_blueChannel, height_blueChannel, width_blueChannel), dtype=np.uint8)

        for z, image_file in enumerate(filesBlueChannel):
            img_z = Image.open(image_file)
            if convert_to_gray:
                img_z = img_z.convert('L')
            pixels_blueChannel[z, :, :] = np.asarray(img_z)
        print('...read the third tensor (%s slices)' %z)

    return pixels_redChannel,pixels_greenChannel,pixels_blueChannel


def main(args):
    pixels_redChannel, pixels_greenChannel,pixels_blueChannel=read_tensors(args)
    fuse_tensors(args.outputdir,pixels_redChannel,pixels_greenChannel,pixels_blueChannel)

def get_parser():
    parser = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('outputdir', metavar='outputdir', type=str,
                        help='output folder where the fused 3D tensor will be saved')
    parser.add_argument('redChannel_dir', metavar='redChannel_dir', type=str,
                        help='path of the 3D red channel tensor')
    parser.add_argument('greenChannel_dir', metavar='greenChannel_dir', type=str,
                        help='path of the 3D green channel tensor')
    parser.add_argument('--blueChannel_dir', metavar='blueChannel_dir', dest='blueChannel_dir', action='store', type=str,
                        help='path of the 3D blue channel tensor')
    return parser



if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()
    main(args)
