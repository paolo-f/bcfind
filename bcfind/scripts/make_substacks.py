#!/usr/bin/env python
"""
Splits a large 3D image into several substacks.

Example: if the source stack is stored in /my/data/mouse1/cerebellum the
program will create substacks in /my/data/substacks/mouse1/cerebellum/
Each substack is a folder named as xxyyzz (six digits) so for example
my/data/substacks/mouse1/cerebellum/010002 is the 
2nd along X, 1st along Y, and 3rd along Z.
A dictionary summarizing the performed split is saved in 
/tmp/my/data/substacks/mouse1/cerebellum/info.json
"""
from __future__ import print_function
import argparse
import json
import sys
import os
import subprocess
import Image
import uuid

from bcfind.utils import mkdir_p

def main(args):
    """
    Files must be numbered consecutively.
    """
    prefix=None
    convert_to_gray=True
    verbose=True
    
    savewd = os.getcwd()
    os.chdir(args.indir)
    full_path = os.getcwd()
    os.chdir(savewd)
    files = os.listdir(args.indir)
    files = [f for f in files if f[0] != '.' and f.endswith(args.suffix)]
    prefix = prefix or os.path.commonprefix(files)
    if prefix == '':
        raise Exception('Files in '+args.indir+' do not have a common prefix ' + str(files))
    # trailing zero dropped -- WHY???
    # zrange = sorted([int(str(f.split(prefix)[1].split(args.suffix)[0])[:-1]) for f in files])
    zrange = sorted([int(f.split('_')[-1].split('.')[0]) for f in files])
    # n_digits = len(files[0].split(prefix)[1].split(args.suffix)[0])
    n_digits = len(f.split('_')[-1].split('.')[0])
    if zrange != range(zrange[0], zrange[-1]+1):
        raise Exception('Files in '+args.indir+' are not a complete sequence: '+str(zrange))
    print('Checking image files')
    prefix = '_'.join(prefix.split('_')[:-1])
    for z in zrange:
        image_file = ('%s_%0'+str(n_digits)+'d'+args.suffix) % (prefix, z)
        img_z = Image.open(full_path+'/'+image_file)
        if z == zrange[0]:
            width, height = img_z.size
        else:
            if (width, height) != img_z.size:
                raise Exception('OOps, file', image_file, 'has size', img_z.size, 'instead of', (width, height))
    depth = len(zrange)

    print('Volume geometry:', width, height, depth)
    print('Finding substacks')
    substacks = dict()
    # nx,ny,nz=3,8,3 # Good for V000_maggio_large_substack_blurred
    w = int(round(float((width-args.margin))/float(args.nx)))
    h = int(round(float((height-args.margin))/float(args.ny)))
    d = int(round(float((depth-args.margin))/float(args.nz)))
    for i in range(args.nx):
        for j in range(args.ny):
            for k in range(args.nz):
                x0 = w*i
                y0 = h*j
                z0 = d*k
                x1 = min(width, x0+w+args.margin)
                y1 = min(height, y0+h+args.margin)
                z1 = min(depth, z0+d+args.margin)
                identifier = "%02d%02d%02d" % (i, j, k)
                substack = dict(Width=x1-x0, Height=y1-y0, Depth=z1-z0, X0=x0, Y0=y0, Z0=z0, Files=[])
                if verbose:
                    print('Substack', (i, j, k))
                    print('X:', x0, x1, w)
                    print('Y:', y0, y1, h)
                    print('Z:', z0, z1, d)
                    print('volume:', ((x1-x0)*(y1-y0)*(z1-z0))/1000000.0, 'MVoxels')
                substacks[identifier] = substack

    print(full_path)
    full_path_of_substacks = "/".join(full_path.split('/')[:-2]) + '/substacks/' + "/".join(full_path.split('/')[-2:])
    print('Saving cropped image files into', full_path_of_substacks)
    for z_from_zero, z in enumerate(zrange):
        image_file = ('%s_%0'+str(n_digits)+'d'+args.suffix) % (prefix, z)
        img_z = Image.open(full_path+'/'+image_file)
        if verbose:
            print(full_path+'/'+image_file, end='')
#        print('==========',z,'=============')
        for substack_id, substack in substacks.iteritems():
            substack_dir = full_path_of_substacks+'/'+substack_id
            mkdir_p(substack_dir)
            X0 = substack['X0']
            Width = substack['Width']
            Y0 = substack['Y0']
            Height = substack['Height']
            Z0 = substack['Z0']
            Depth = substack['Depth']
#            print(substack)
            if z_from_zero in range(Z0, Z0+Depth):
                region = img_z.crop((X0+0, Y0+0, X0+Width, Y0+Height))
                if convert_to_gray:
                    region = region.convert('L')
                if verbose:
                    print(' ', substack_id, end='')
                # region.save(substack_dir+'/'+image_file)
                tmpfile = '/tmp/'+uuid.uuid4()+'.tif'
                region.save(tmpfile)
                subprocess.call(['tiffcp', '-clzw:2', tmpfile, substack_dir+'/'+image_file])
                os.unlink(tmpfile)
                # substack['Files'].append(substack_dir+'/'+image_file)
                # the path to files is now relative
                substack['Files'].append(substack_id+'/'+image_file)
        if verbose:
            print()

    print('Saving substack info into', full_path_of_substacks+'/info.json')
    info = dict(ImageSequenceDirectory=full_path,
                Width=width,
                Height=height,
                Depth=depth,
                Margin=args.margin,
                SubStacks=substacks,
                Full_Path_Of_Substacks=full_path_of_substacks)
    with open(full_path_of_substacks+'/info.json', 'w') as ostream:
        print(json.dumps(info), file=ostream)

def get_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('indir', metavar='indir', type=str,
                        help='Directory contaning the source stack (sequence of TIFF files)')
    parser.add_argument('nx', metavar='nx', type=int, help='# of substacks along the X dimension')
    parser.add_argument('ny', metavar='ny', type=int, help='# of substacks along the Y dimension')
    parser.add_argument('nz', metavar='nz', type=int, help='# of substacks along the Z dimension')
    parser.add_argument('--suffix', dest='suffix', type=str, default='.tif', help='Image file suffix')
    parser.add_argument('--margin', dest='margin', type=int, default=40, help='Overlap between adjacent substacks')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Verbose output.')
    parser.set_defaults(verbose=False)
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
