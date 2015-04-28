#!/usr/bin/env python
"""
Converts a TIFF image sequence to an HDF5 file
"""
from __future__ import print_function
import argparse
import numpy as np
import tables
import os
import progressbar as pb
from PIL import Image

def get_files(indir, prefix=None, suffix = '.tif'):
    savewd = os.getcwd()
    os.chdir(indir)
    full_path = os.getcwd()
    os.chdir(savewd)
    files = os.listdir(indir)
    files = [f for f in files if f[0] != '.' and f.endswith(suffix)]
    prefix = prefix or os.path.commonprefix(files)
    if prefix == '':
        raise Exception('Files in '+indir+' do not have a common prefix ' + str(files))
    zrange = sorted([int(f.split('_')[-1].split('.')[0]) for f in files])
    n_digits = len(f.split('_')[-1].split('.')[0])
    if zrange != range(zrange[0], zrange[-1]+1):
        raise Exception('Files in '+indir+' are not a complete sequence: '+str(zrange))
    img_z = Image.open(full_path+'/'+files[0])
    width, height = img_z.size
    return {'full_path':full_path, 'files':sorted(files), 'minz':zrange[0], 'maxz':zrange[-1],
            'width':width, 'height':height}



def get_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('indir', metavar='indir', type=str,
                        help='needs indir/info.plist, substacks, e.g. indir/100905, and GT files e.g. indir/100905-GT.marker')
    parser.add_argument('outfile', metavar='outfile', type=str,
                        help='where to save the HDF5 file (check free disk space!!)')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Verbose output.')
    parser.set_defaults(verbose=False)
    return parser

def main(args):
    d = get_files(args.indir)
    shape = (d['maxz']-d['minz']+1, d['height'], d['width'])
    print('Shape is', shape)
    atom = tables.UInt8Atom()

    h5f = tables.openFile(args.outfile, 'w')
    ca = h5f.createCArray(h5f.root, 'full_image', atom, shape)
    h5f.createArray(h5f.root, 'minz', np.array([d['minz']]))

    pbar = pb.ProgressBar(widgets=['Converting %d images: ' % len(d['files']),
                                   pb.Percentage(), ' ', pb.ETA()],
                          maxval=len(d['files'])).start()
    for z, image_file in enumerate(d['files']):
        print('opening', image_file)
        img_z = Image.open(d['full_path'] + '/' + image_file).convert('L')
        pix = np.array(img_z, dtype=np.uint8)
        ca[z,:,:] = pix
        pbar.update(z+1)
    pbar.finish()
    h5f.close()


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)

