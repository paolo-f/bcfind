from __future__ import print_function
import tables
import os
from PIL import Image
import numpy as np
from progressbar import *

from bcfind.utils import mkdir_p


def pad_if_out_of_range(from_shape, origin, ss_shape):
    """Take a subtensor of shape `ss_shape` from a tensor of shape
`from_shape` starting from `origin`. All are n-dim tuples.  Returns an
n-dim tuple of pairs that may be used with numpy.pad

    """
    def start(a,b):
        if b >= a:
            return 0
        else:
            return a-b

    def end(c,d):
        if d <= c:
            return 0
        else:
            return d-c

    if len(from_shape) != len(origin) or len(from_shape) != len(ss_shape):
        raise Exception('The three tuples should have the same length')
    widths = []
    for dim in range(len(from_shape)):
        widths.append((start(0,origin[dim]), end(from_shape[dim], origin[dim]+ss_shape[dim])))
    return tuple(widths)


def load_nearby(tensorimage, ss, extramargin):
    """Load a substack from a large tensor image stored in an HDF5
file. The extramargin is useful when convolut-like operations are
performed on the substack.

    """
    hf5 = tables.openFile(tensorimage, 'r')
    X0,Y0,Z0 = ss.info['X0'], ss.info['Y0'], ss.info['Z0']
    H,W,D = ss.info['Height'], ss.info['Width'], ss.info['Depth']
    from_shape = hf5.root.full_image.shape
    origin = (Z0-extramargin, Y0-extramargin, X0-extramargin)
    ss_shape = (D+2*extramargin,H+2*extramargin,W+2*extramargin)
    # np_tensor_3d = hf5.root.full_image[1200:1500,1200:1500,1200:1500]
    np_tensor_3d = hf5.root.full_image[max(0,origin[0]):min(origin[0]+ss_shape[0],from_shape[0]),
                                       max(0,origin[1]):min(origin[1]+ss_shape[1],from_shape[1]),
                                       max(0,origin[2]):min(origin[2]+ss_shape[2],from_shape[2])]
    pad = pad_if_out_of_range(from_shape, origin, ss_shape)
    print('pad', pad)
    np_tensor_3d = np.pad(np_tensor_3d,pad,mode='constant')
    print('new shape', np_tensor_3d.shape)
    minz = int(hf5.root.minz[0])
    hf5.close()
    return np_tensor_3d, minz


def save_tensor_as_tif(np_tensor_3d,path, minz, prefix='full_'):
    """Export a 3D numpy tensor as a sequence of tiff files (one for each
z coordinate). Each file is named as prefix followed by an integer id,
which ranges in minz:minz+np_tensor_3d.shape[0].  The tensor is
expected to be stored in the order Z,Y,X

    """
    import uuid
    mkdir_p(path)
    pbar = ProgressBar(widgets=['Saving %d tiff files: ' % np_tensor_3d.shape[0], Percentage(), ' ', AdaptiveETA()])
    for z in pbar(range(np_tensor_3d.shape[0])):
        out_img = Image.fromarray(np_tensor_3d[z,:,:])
        tempname = '/tmp/'+str(uuid.uuid4())+'.tif'
        out_img.save(tempname)
        destname = path+'/'+prefix+'%04d.tif' % (minz+z)
        os.system('tiffcp -clzw:2 ' + tempname + ' ' + destname)
        os.remove(tempname)

    print('Saved substack in',path)
