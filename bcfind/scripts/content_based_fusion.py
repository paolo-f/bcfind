#!/usr/bin/env python
"""
Script that fuses two 3D greyscale tensors in a unique 3D greyscale output tensor.
The algorithm is based on the local entropy estimation.

References
----------
Preibisch, S., Rohlfing, T., Hasak, M.P., Tomancak, P.: Mosaicing of single plane illumination
microscopy images using groupwise registration and fast content-based image fusion. In:
J.M. Reinhardt, J.P.W. Pluim (eds.) Proc. SPIE Conference on Medical Imaging, vol. 6914 (2008)
"""

from __future__ import print_function
from bcfind.volume import SubStack
from bcfind.semadec import imtensor
from bcfind import local_entropy
from bcfind.utils import mkdir_p
import tables
import numpy as np
import argparse
from progressbar import *
import timeit
import os.path
from clsm_registration.rigid_transformation import * 

def do_content_based_fusion(np_tensor_3d_first_view,np_tensor_3d_second_view, size_patch, extramargin, speedup=2, n_bins=256, fast_computation=True):

    sc = np_tensor_3d_first_view.shape
    fused_image = np.zeros(sc).astype(float)
    patchlen = (1+2*size_patch)**3
    entropy_mask_first_view=np.zeros(sc).astype(float)
    entropy_mask_second_view=np.zeros(sc).astype(float)

    start_ex=timeit.default_timer()
    if fast_computation:
        entropy_mask_first_view,entropy_mask_second_view=local_entropy.loop_compute_local_entropy(np_tensor_3d_first_view, np_tensor_3d_second_view, extramargin, size_patch, n_bins, speedup)
    else:
        rangez = range(extramargin, sc[0]-extramargin, speedup)
        rangey = range(extramargin, sc[1]-extramargin, speedup)
        rangex = range(extramargin, sc[2]-extramargin, speedup)
        print('Patch size: %dx%dx%d (%d)' % (1+2*size_patch, 1+2*size_patch, 1+2*size_patch, (1+2*size_patch)**3))
        print('Will subsample jumping by',speedup,'voxels')
        bar_extraction = ProgressBar(widgets=['Pre-processing %d slices (%d patches): ' % (len(rangez),len(rangex)*len(rangey)*len(rangez)),
                                    Percentage(), ' ', ETA()])

        n_points = len(rangex)*len(rangey)
        ndone = 0
        data = np.zeros((len(rangez) * n_points, patchlen*2))
        iter_z = 0
        for z0 in bar_extraction(rangez):
            i = 0
            for y0 in rangey:
                for x0 in rangex:
                    patch_first_view = np_tensor_3d_first_view[z0-size_patch:z0+size_patch+1,
                                                            y0-size_patch:y0+size_patch+1,
                                                            x0-size_patch:x0+size_patch+1]
                    patch_second_view = np_tensor_3d_second_view[z0-size_patch:z0+size_patch+1,
                                                                y0-size_patch:y0+size_patch+1,
                                                                x0-size_patch:x0+size_patch+1]

                    data[iter_z * n_points + i][0:patchlen] = patch_first_view.ravel()
                    data[iter_z * n_points + i][patchlen:2*patchlen] = patch_second_view.ravel()
                    hist_first_view,_ = np.histogram(data[iter_z * n_points + i][0:patchlen], bins=n_bins, range=(0,256), density=True)
                    hist_second_view,_ = np.histogram(data[iter_z * n_points + i][patchlen:2*patchlen], bins=n_bins, range=(0,256), density=True)
                    entropy_mask_first_view[z0,y0,x0]=-np.nansum(hist_first_view*np.log2(hist_first_view))
                    entropy_mask_second_view[z0,y0,x0]=-np.nansum(hist_second_view*np.log2(hist_second_view))
                    i += 1
            iter_z += 1

    stop_ex=timeit.default_timer()
    print ("total time computing entropy ", str(stop_ex - start_ex))


    power_entropy_first_view=np.power(100,entropy_mask_first_view)
    power_entropy_second_view=np.power(100,entropy_mask_second_view)

    fused_image=np.multiply(power_entropy_first_view,np_tensor_3d_first_view) + np.multiply(power_entropy_second_view,np_tensor_3d_second_view)
    fused_image=np.round(np.divide(fused_image, power_entropy_first_view + power_entropy_second_view)).astype(np.uint8)

    return fused_image,entropy_mask_first_view,entropy_mask_second_view




def main(args):

    total_start = timeit.default_timer()
    print('Starting Preibisch fusion', args.substack_id)

    ss = SubStack(args.first_view_dir, args.substack_id)
    minz = int(ss.info['Files'][0].split("/")[-1].split('_')[-1].split('.tif')[0])
    prefix = '_'.join(ss.info['Files'][0].split("/")[-1].split('_')[0:-1])+'_'
    np_tensor_3d_first_view,_  = imtensor.load_nearby(args.tensorimage_first_view, ss, args.size_patch)
    sc_in=np_tensor_3d_first_view.shape

    if args.transformation_file is not None:
	R, t = parse_transformation_file(args.transformation_file)
        np_tensor_3d_second_view = transform_substack(args.second_view_dir, args.tensorimage_second_view, args.substack_id, R, t, args.size_patch, invert=True)
    else:
        np_tensor_3d_second_view,_  = imtensor.load_nearby(args.tensorimage_second_view, ss, args.size_patch)

    fused_image,entropy_mask__view,entropy_mask_second_view = do_content_based_fusion(np_tensor_3d_first_view,np_tensor_3d_second_view,args.size_patch, args.size_patch, speedup=1,fast_computation=True)
   
    if args.extramargin>args.size_patch:
	args.extramargin=args.size_patch
    
    offset_margin=args.size_patch - args.extramargin
    fused_image_output=fused_image[offset_margin:sc_in[0]-offset_margin,offset_margin:sc_in[1]-offset_margin,offset_margin:sc_in[2]-offset_margin]
    atom = tables.UInt8Atom()
    mkdir_p(args.outdir)
    h5f = tables.openFile(args.outdir + '/' + args.substack_id + '.h5', 'w')
    sc_out=fused_image_output.shape
    ca = h5f.createCArray(h5f.root, 'full_image', atom, sc_out)
    for z in xrange(0, sc_out[0], 1):
        ca[z, :, :] = fused_image_output[z,:,:]
    h5f.close()

    imtensor.save_tensor_as_tif(fused_image_output, args.outdir+'/'+args.substack_id, minz,prefix=prefix)
    print ("total time Preibisch fusion: %s" %(str(timeit.default_timer() - total_start)))


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
                        help='where preprocessed volume will be saved')
    parser.add_argument('--extramargin', metavar='extramargin', dest='extramargin',
                        action='store', type=int, default=5,
                        help='Extra margin for convolution. Should be equal to (filter_size - 1)/2')
    parser.add_argument('-s', '--size_patch', dest='size_patch',
                        action='store', type=int, default=4,
                        help='local entropy will be computed inside cubic patches of side (2*size+1)')
    parser.add_argument('--transformation_file', metavar='transformation_file', type=str,
                        help='Transformation log file')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)

