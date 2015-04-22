#!/usr/bin/env python
"""
Script that fuses two 3D greyscale tensors in a unique 3D greyscale output tensor.
The algorithm is based on the local entropy estimation of both the tensors.
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
import shutil

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
                                    Percentage(), ' ', AdaptiveETA()])

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
    fused_image=fused_image[extramargin:sc[0]-extramargin,extramargin:sc[1]-extramargin,extramargin:sc[2]-extramargin]

    return fused_image,entropy_mask_first_view,entropy_mask_second_view




def main(args):

    total_start = timeit.default_timer()
    print('Starting fusion of volume', args.substack_id)

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.outdir):
        mkdir_p(args.outdir)

    # Copy the json file if it doesn't exist
    if not os.path.isfile(args.outdir+'/info.json'):
        shutil.copy2(args.first_view_dir+'/info.json', args.outdir)


    ss_first_view = SubStack(args.first_view_dir, args.substack_id)
    minz = int(ss_first_view.info['Files'][0].split("/")[-1].split('_')[-1].split('.tif')[0])
    prefix = '_'.join(ss_first_view.info['Files'][0].split("/")[-1].split('_')[0:-1])+'_'
    np_tensor_3d_first_view, _ = imtensor.load_nearby(args.first_view_dir.rstrip('//') + '.h5', ss_first_view, args.extramargin)

    hf5_second_view = tables.openFile(args.second_view_dir.rstrip('//') + '/' + str(args.substack_id) + '.h5', 'r')
    np_tensor_3d_second_view = np.array(hf5_second_view.root.full_image)
    hf5_second_view.close()


    _speedup=1
    fused_image,entropy_mask__view,entropy_mask_second_view = do_content_based_fusion(np_tensor_3d_first_view,np_tensor_3d_second_view,args.size_patch, args.extramargin, speedup=_speedup,fast_computation=True)
    #import ipdb; ipdb.set_trace()
    imtensor.save_tensor_as_tif(fused_image, args.outdir+'/'+args.substack_id+'_speedup'+str(_speedup), minz,prefix=prefix)



def get_parser():
    parser = argparse.ArgumentParser(description="""
    Preprocess a substack using a neural network model
    """, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('first_view_dir', metavar='first_view_dir', type=str,
                        help='must contain indir/info.json, substacks, e.g. indir/000, e.g. indir/000-GT.marker')
    parser.add_argument('second_view_dir', metavar='second_view_dir', type=str,
                        help='must contain indir/info.json, substacks, e.g. indir/000, e.g. indir/000-GT.marker')
    parser.add_argument('substack_id', metavar='substack_id', type=str,
                        help='substack identifier, e.g. 010608')
    parser.add_argument('outdir', metavar='outdir', type=str,
                        help='where preprocessed volume will be saved')
    parser.add_argument('--extramargin', metavar='extramargin', dest='extramargin',
                        action='store', type=int, default=5,
                        help='Extra margin for convolution. Should be equal to (filter_size - 1)/2')
    parser.add_argument('-s', '--size_patch', dest='size_patch',
                        action='store', type=int, default=4,
                        help='Input and output patches are cubes of side (2*size+1)**3')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)

