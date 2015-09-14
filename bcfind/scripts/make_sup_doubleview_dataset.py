from __future__ import print_function


import sys
sys.path.insert(1,'/home/logos_users/paciscopi/bcfind20/bcfind')

import numpy as np
import tables
import argparse
from progressbar import *
from bcfind.volume import *
from scipy.spatial import cKDTree
from bcfind.semadec import imtensor
import scipy.ndimage.filters as gfilter
import pandas as pd
import os
from bcfind.utils import Struct,mkdir_p,which
from bcfind.scripts import transform_views


def inside_margin(c, substack):
    m = substack.plist['Margin'] / 2
    return min(c.x - m, c.y - m, c.z - m, substack.info['Width'] - m - c.x, substack.info['Height'] - m - c.y,
               substack.info['Depth'] - m - c.z)

def inside_patch(c_array,x0, y0, z0,size,offset=0):
    if np.where((c_array[:,0] >= x0 - size - offset)
             & (c_array[:,0] < x0 + size +1 + offset)
             &  (c_array[:,1] >= y0 - size - offset)
             & (c_array[:,1] < y0 + size +1 + offset)
             & (c_array[:,2] >= z0 - size - offset)
                & (c_array[:,2] < z0 + size +1 + offset))[0].size:
        return True
    else:
        return False

def make_pos_neg_dataset(tensor_first_view, tensor_second_view, ss, C, view1_id, view2_id, default_sigma=0.8,size=5, save_tiff_files=False, find_negative=True):

    tensor_first_view = tensor_first_view.astype(np.float32)
    tensor_second_view = tensor_second_view.astype(np.float32)
    H, W, D = ss.info['Height'], ss.info['Width'], ss.info['Depth']
    patchlen = (1 + 2 * size) ** 3

    margin = ss.plist['Margin'] / 2

    c_array = np.array([[c.x, c.y, c.z] for c in C])


    trunc=1.5
    fixed_radiiassigned={c:default_sigma for c in C  if inside_margin(c,ss) > 0 }
    cccc=fixed_radiiassigned.copy()
    kdt = cKDTree(c_array)
    dist,ind = kdt.query(c_array,k=c_array.shape[0],distance_upper_bound=(2*default_sigma*trunc)+2.)
    for c,id_c in zip(C,xrange(c_array.shape[0])):
        if inside_margin(c, ss) > 0:
            for j in xrange(2,c_array.shape[0]):
                if not np.isinf(dist[id_c][j]):
                    if inside_margin(C[ind[id_c][j]], ss) > 0:
                        if trunc*(fixed_radiiassigned[c]+fixed_radiiassigned[C[ind[id_c][j]]]) > dist[id_c][j] - 2.:
                            new_sigma=((dist[id_c][j]-1.)/trunc)*fixed_radiiassigned[c]/(fixed_radiiassigned[c]+fixed_radiiassigned[C[ind[id_c][j]]])
                            fixed_radiiassigned[C[ind[id_c][j]]]=(new_sigma*fixed_radiiassigned[C[ind[id_c][j]]])/fixed_radiiassigned[c]
                            fixed_radiiassigned[c]=new_sigma


    target_tensor_3d = np.zeros(tensor_first_view.shape,dtype=np.uint8)
    for c in C:
        if inside_margin(c, ss) > 0:
            target_tensor_3d_tmp = np.zeros(tensor_first_view.shape)
            target_tensor_3d_tmp[c.z, c.y, c.x] = 1
            sigma=fixed_radiiassigned[c]
            target_tensor_3d_tmp = gfilter.gaussian_filter(target_tensor_3d_tmp, sigma,
                                               mode='constant', cval=0.0,
                                               truncate=trunc)  #sigma=3.5,mode='constant',cval=0.0,truncate=1.5
            target_tensor_3d_tmp = (target_tensor_3d_tmp.astype(np.float32) / np.max(target_tensor_3d_tmp))

            target_tensor_3d =  np.maximum(np.array(target_tensor_3d_tmp * 255.0, dtype=np.uint8), target_tensor_3d)

    save_tiff_files=True

    if save_tiff_files:
        debug_path='/tmp/debug/sigma_'+str(default_sigma)
        mkdir_p(debug_path)
        minz = 0
        imtensor.save_tensor_as_tif(target_tensor_3d, debug_path+'/'+ss.substack_id+'_'+view1_id+'_'+view2_id, minz)


    print('num markers =',len(C))
    target_tensor_3d = (target_tensor_3d.astype(np.float32) / np.max(target_tensor_3d))
    tensor_first_view = tensor_first_view.astype(np.float32) / 255.0
    tensor_second_view = tensor_second_view.astype(np.float32) / 255.0

    nrej_intensity = 0
    num_negative_targets = 0
    num_positive_targets = 0

    step_x = 1
    step_y = 1
    step_z = 1
    num_iterations = (W - 2*margin +1) * (H- 2*margin) * (D - 2*margin) / (step_x * step_y * step_z)
    pbar = ProgressBar(widgets=['Making positive and negative examples for %d points: ' % num_iterations, Percentage()],
                       maxval=num_iterations).start()

    X_positive = np.zeros((num_iterations, patchlen*2), dtype=np.float32)
    y_positive = np.zeros((num_iterations, patchlen), dtype=np.float32)

    X_negative = []
    y_negative = []
    if find_negative == True:
        X_negative = np.zeros((num_iterations, patchlen*2), dtype=np.float32)
        y_negative = np.zeros((num_iterations, patchlen), dtype=np.float32)

    pbi = 0
    for x0 in range(margin, W - margin, step_x):
        for y0 in range(margin, H - margin, step_y):
            for z0 in range(margin, D - margin, step_z):

                if inside_patch(c_array, x0, y0, z0, size, offset=2.0):
                    patch_left = tensor_first_view[z0 - size: z0 + size + 1, y0 - size: y0 + size + 1, x0 - size: x0 + size + 1]
                    patch_right = tensor_second_view[z0 - size: z0 + size + 1, y0 - size: y0 + size + 1, x0 - size: x0 + size + 1]

                    X_positive[num_positive_targets, 0:patchlen] = np.reshape(patch_left, (patchlen,))
                    X_positive[num_positive_targets, patchlen:2*patchlen] = np.reshape(patch_right, (patchlen,))
                    ypatch = target_tensor_3d[z0 - size:z0 + size + 1, y0 - size:y0 + size + 1,
                                x0 - size:x0 + size + 1]

                    y_positive[num_positive_targets, :] = np.reshape(ypatch, (patchlen,))
                    num_positive_targets += 1

                elif find_negative == True:
                    patch_left = tensor_first_view[z0 - size: z0 + size + 1, y0 - size: y0 + size + 1, x0 - size: x0 + size + 1]
                    patch_right = tensor_second_view[z0 - size: z0 + size + 1, y0 - size: y0 + size + 1, x0 - size: x0 + size + 1]
                    if np.mean(patch_left * 255) > 5 or np.mean(patch_right * 255) > 5:
                        X_negative[num_negative_targets, 0:patchlen] = np.reshape(patch_left, (patchlen,))
                        X_negative[num_negative_targets, patchlen:2*patchlen] = np.reshape(patch_right, (patchlen,))
                        ypatch = target_tensor_3d[z0 - size: z0 + size + 1, y0 - size: y0 + size + 1, x0 - size: x0 + size + 1]


                        y_negative[num_negative_targets, :] = np.reshape(ypatch, (patchlen,))
                        num_negative_targets += 1
                    else:
                        nrej_intensity += 1
                pbar.update(pbi+ 1)
                pbi += 1
    pbar.finish()


    X_positive = X_positive[0: num_positive_targets]
    y_positive = y_positive[0: num_positive_targets]
    print('Total positive examples for substack (', ss.substack_id, '):', num_positive_targets)

    if find_negative == True:

        print('Total negative examples for substack (', ss.substack_id, '):', num_negative_targets)
        print('Rejected by intensity ', nrej_intensity)
        X_negative = X_negative[0: num_negative_targets]
        y_negative = y_negative[0: num_negative_targets]

        if (num_negative_targets > num_positive_targets):
            print('Shuffling negative examples...')
            perm = np.random.permutation(len(X_negative)).astype(int)
            X_negative = X_negative[perm]
            y_negative = y_negative[perm]
            print('Shuffling done')
            X_negative = X_negative[0: num_positive_targets]
            y_negative = y_negative[0: num_positive_targets]


    return X_positive, y_positive, X_negative, y_negative


def main(args):

    patchlen = (1+2*args.size_patch) ** 3

    X_sup = np.array([]).reshape(0, 2*patchlen).astype(np.float32)
    y_sup = np.array([]).reshape(0, patchlen).astype(np.float32)
    X_neg = np.array([]).reshape(0, 2*patchlen).astype(np.float32)
    y_neg = np.array([]).reshape(0, patchlen).astype(np.float32)

    data_frame_markers = pd.read_csv(args.list_trainset, dtype={'view1': str, 'view2': str, 'ss_id': str })

    for row in data_frame_markers.index:
        row_data=data_frame_markers.iloc[row]

        first_view_dir=args.substacks_base_path+'/'+row_data['view1']

        substack = SubStack(first_view_dir, row_data['ss_id'])
        markers = args.mergedmarkers_folder + '/' +  row_data['view1']+'_'+row_data['view2']+ '/'+row_data['ss_id']+'-GT.marker'
        print('Loading ground truth markers from', markers)
        C = substack.load_markers(markers, from_vaa3d=True)
        for c in C:
            c.x -= 1
            c.y -= 1
            c.z -= 1

        substack.load_volume(h5filename=args.tensors_base_path+'/'+row_data['view1'] + '.h5')
        tensor_first_view,_ = imtensor.load_nearby(args.tensors_base_path+'/'+row_data['view1'] + '.h5', substack, 0)


        second_view_dir=args.substacks_base_path+'/'+row_data['view2']
        args_transf=argparse.Namespace()
        args_transf.indir=second_view_dir
        args_transf.tensorimage=args.tensors_base_path+'/'+row_data['view2']+'.h5'
        args_transf.log_file=args.transforms_folder+'/'+row_data['ss_id']+'/'+row_data['view1']+'_'+row_data['view2']
        args_transf.outdir='/tmp'
        args_transf.substack_id=row_data['ss_id']
        args_transf.extramargin=0
        args_transf.invert=True
        args_transf.save_tiff=False
        args_transf.get_tensor=True
        R, t = transform_views.parse_transformation_file(args_transf)
        tensor_second_view = transform_views.transform_substack(args_transf,R,t)


        temp_X_sup, temp_y_sup, temp_X_neg, temp_y_neg = make_pos_neg_dataset(tensor_first_view, tensor_second_view, substack, C,row_data['view1'],row_data['view2'], default_sigma=args.sigma,size=args.size_patch, save_tiff_files=False ,find_negative=args.negatives)


        if args.local_standardization:
            print('Do local standardization')
            Xmean = np.vstack((temp_X_sup,temp_X_neg)).mean(axis=0)
            Xstd = np.vstack((temp_X_sup,temp_X_neg)).std(axis=0)
            temp_X_sup = (temp_X_sup - Xmean) / Xstd
            temp_X_neg = (temp_X_neg - Xmean) / Xstd


        X_sup = np.vstack((X_sup, temp_X_sup))
        y_sup = np.vstack((y_sup, temp_y_sup))
        X_neg = np.vstack((X_neg, temp_X_neg))
        y_neg = np.vstack((y_neg, temp_y_neg))


    print('Negative Data set shape:', X_neg.shape, 'size:', X_neg.nbytes / (1024 * 1024), 'MBytes')
    print('Negative target shape:', y_neg.shape, 'size:', y_neg.nbytes / (1024 * 1024), 'MBytes')
    print('Positive Data set shape:', X_sup.shape, 'size:', X_sup.nbytes / (1024 * 1024), 'MBytes')
    print('Positive target shape:', y_sup.shape, 'size:', y_sup.nbytes / (1024 * 1024), 'MBytes')

    ratio_positive_negative = float(y_sup.sum())/float(y_neg.shape[0]*y_neg.shape[1] + y_sup.shape[0]*y_sup.shape[1])
    print('ratio positive-negative:', ratio_positive_negative)

    X = np.vstack((X_sup,X_neg))
    y = np.vstack((y_sup,y_neg))

    print('Total Data set shape:', X.shape, 'size:', X.nbytes / (1024 * 1024), 'MBytes')
    print('Total target shape:', y.shape, 'size:', y.nbytes / (1024 * 1024), 'MBytes')

    if not args.local_standardization:
        print('Do global standardization')
        Xmean = X.mean(axis=0)
        Xstd = X.std(axis=0)
        X = (X - Xmean) / Xstd

    print('Saving training data to', args.outfile)
    h5file = tables.openFile(args.outfile, mode='w', title="Training set")
    root = h5file.root
    h5file.createArray(root, "X", X)
    h5file.createArray(root, "y", y)
    if not args.local_standardization:
        h5file.createArray(root, "Xmean", Xmean)
        h5file.createArray(root, "Xstd", Xstd)
    h5file.close()


def get_parser():
    parser = argparse.ArgumentParser(description="""
    Creates a data set for supervised training on two views of 3D patches
    """, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('list_trainset', metavar='list_trainset', type=str,
                        help='csv file of merged markers of the trainset')
    parser.add_argument('substacks_base_path', metavar='substacks_base_path', type=str,
                        help='Name of the folder in which substacks of the aligned views are stored')
    parser.add_argument('tensors_base_path', metavar='tensors_base_path', type=str,
                        help='Name of the folder in which hdf5 tensors of the aligned views are stored')
    parser.add_argument('mergedmarkers_folder', metavar='mergedmarkers_folder', type=str,
                        help='Name of the folder in which merged markers are stored')
    parser.add_argument('transforms_folder', metavar='transform_folder', type=str,
                        help='Name of the folder in which estimated rigid transformations are stored')
    parser.add_argument('outfile', metavar='outfile', type=str,
                        help='Name of file where the dataset will be saved')
    parser.add_argument('--sigma', metavar='sigma', dest='sigma',
                        action='store', type=float, default=0.8,
                        help='sigma of gaussian filter')
    parser.add_argument('--size_patch', metavar='size_patch', dest='size_patch',
                        action='store', type=int, default=5,
                        help='size of trainset patches')
    parser.add_argument('--negatives', dest='negatives', action='store_true',
                        help='include "negative" (non cell) examples.')
    parser.add_argument('-l','--local_standardization', dest='local_standardization', action='store_true',
                        help='do the standardization locally or globally')
    parser.add_argument('--no-negatives', dest='negatives', action='store_false', help='Include only cell examples.')
    parser.set_defaults(negatives=False)
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
