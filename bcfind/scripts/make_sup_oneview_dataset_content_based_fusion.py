from __future__ import print_function
import numpy as np
import tables
import argparse
from progressbar import *
from bcfind.volume import *
from scipy.spatial import cKDTree
import scipy.ndimage.filters as gfilter
from bcfind.semadec import imtensor
from bcfind.utils import Struct,mkdir_p,which
import os


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

#@profile
def make_pos_neg_dataset(tensor_view, ss, C, size=5, save_tiff_files=False, default_sigma=0.8, find_negative=True):
    tensor_view = tensor_view.astype(np.float32)

    H, W, D = ss.info['Height'], ss.info['Width'], ss.info['Depth']
    patchlen = (1 + 2 * size) ** 3
    c_array = np.array([[c.x, c.y, c.z] for c in C])

    target_tensor_3d = np.zeros(tensor_view.shape)
    margin = ss.plist['Margin'] / 2
  

    trunc=1.5
    fixed_radiiassigned={c:default_sigma for c in C  if inside_margin(c,ss) > 0 }
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


    for c in C:
        if inside_margin(c, ss) > 0:
            target_tensor_3d_tmp = np.zeros(tensor_view.shape)
            target_tensor_3d_tmp[c.z, c.y, c.x] = 1
            sigma=fixed_radiiassigned[c]
            target_tensor_3d_tmp = gfilter.gaussian_filter(target_tensor_3d_tmp, sigma,
                                               mode='constant', cval=0.0,
                                               truncate=trunc)
            target_tensor_3d_tmp = (target_tensor_3d_tmp.astype(np.float32) / np.max(target_tensor_3d_tmp))
            target_tensor_3d =  np.maximum(np.array(target_tensor_3d_tmp * 255.0, dtype=np.uint8), target_tensor_3d)



    if save_tiff_files:
        debug_path='/mnt/data/marco/experiments/markers/debug_single_view_on_fused_image'
        mkdir_p(debug_path)
        minz = 0
        target_tensor_3d = np.array(target_tensor_3d, dtype=np.uint8)
        imtensor.save_tensor_as_tif(target_tensor_3d, debug_path+'/'+ss.substack_id, minz)

    target_tensor_3d = (target_tensor_3d.astype(np.float32) / np.max(target_tensor_3d))
    tensor_view = tensor_view.astype(np.float32) / 255.0

    nrej_intensity = 0
    num_negative_targets = 0
    num_positive_targets = 0

    step_x = 1
    step_y = 1
    step_z = 1
    num_iterations = (W - 2*margin) * (H- 2*margin) * (D - 2*margin) / (step_x * step_y * step_z)
    pbar = ProgressBar(widgets=['Making positive and negative examples for %d points: ' % num_iterations, Percentage()],
                       maxval=num_iterations).start()

    X_positive = np.zeros((num_iterations, patchlen), dtype=np.float32)
    y_positive = np.zeros((num_iterations, patchlen), dtype=np.float32)

    X_negative = []
    y_negative = []
    if find_negative == True:
        X_negative = np.zeros((num_iterations, patchlen), dtype=np.float32)
        y_negative = np.zeros((num_iterations, patchlen), dtype=np.float32)

    pbi = 0
    for x0 in range(margin, W - margin, step_x):
        for y0 in range(margin, H - margin, step_y):
            for z0 in range(margin, D - margin, step_z):

                if inside_patch(c_array, x0, y0, z0, size,offset=2.0):  #found a marker inside margin
                    patch = tensor_view[z0 - size: z0 + size + 1, y0 - size: y0 + size + 1, x0 - size: x0 + size + 1]
                    X_positive[num_positive_targets, :] = np.reshape(patch, (patchlen,))
                    ypatch = target_tensor_3d[z0 - size:z0 + size + 1, y0 - size:y0 + size + 1,
                                x0 - size:x0 + size + 1]
                    y_positive[num_positive_targets, :] = np.reshape(ypatch, (patchlen,))
                    num_positive_targets += 1  # Sample as many negatives as positives

                elif find_negative == True:  #min_distance > (size * (3**(1/2.0))) + 2.0
                    patch = tensor_view[z0 - size: z0 + size + 1, y0 - size: y0 + size + 1, x0 - size: x0 + size + 1]
                    if np.mean(patch * 255) > 5:
                        X_negative[num_negative_targets, :] = np.reshape(patch, (patchlen,))
                        ypatch = target_tensor_3d[z0 - size: z0 + size + 1, y0 - size: y0 + size + 1, x0 - size: x0 + size + 1]
                        y_negative[num_negative_targets, :] = np.reshape(ypatch, (patchlen,))
                        num_negative_targets += 1
                    else:
                        nrej_intensity += 1
                pbar.update(pbi + 1)
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

    patchlen = (1 + 2 * args.size_patch) ** 3

    X_sup = np.array([]).reshape(0, patchlen).astype(np.float32)
    y_sup = np.array([]).reshape(0, patchlen).astype(np.float32)
    X_neg = np.array([]).reshape(0, patchlen).astype(np.float32)
    y_neg = np.array([]).reshape(0, patchlen).astype(np.float32)
    X_original = np.array([]).reshape(0, patchlen).astype(np.float32)

    data_frame_markers = pd.read_csv(args.list_trainset, dtype={'view1': str, 'view2': str, 'ss_id': str })

    for row in data_frame_markers.index:
        row_data=data_frame_markers.iloc[row]

        markers = args.mergedmarkers_folder + '/' +  row_data['view1']+'_'+ row_data['view2']+'/'+row_data['ss_id']+'-GT.marker'

        ss = SubStack(args.indir, row_data['ss_id'])
        hf5_view = tables.openFile(args.indir.rstrip('//') + '/' + row_data['ss_id'] + '.h5', 'r')
        view_shape = hf5_view.root.full_image.shape
        np_tensor_3d_view = hf5_view.root.full_image[args.extramargin:view_shape[0] - args.extramargin, args.extramargin:view_shape[1] - args.extramargin, args.extramargin:view_shape[2] - args.extramargin]
        hf5_view.close()

        print('Loading ground truth markers from', markers)
        C = ss.load_markers(markers, from_vaa3d=True)
        for c in C:
            c.x -= 1
            c.y -= 1
            c.z -= 1


        temp_X_sup, temp_y_sup, temp_X_neg, temp_y_neg = make_pos_neg_dataset(np_tensor_3d_view,  ss, C, save_tiff_files=True , size=args.size_patch, find_negative=args.negatives)

        X_original = np.vstack((X_original, temp_X_sup))
        X_original = np.vstack((X_original, temp_X_neg))
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

  
    print('Compute global mean and std')
    Xmean = X_original.mean(axis=0)
    Xstd = X_original.std(axis=0)

    print('Saving training data to', args.outfile)
    h5file = tables.openFile(args.outfile, mode='w', title="Training set")
    root = h5file.root
    h5file.createArray(root, "X", X)
    h5file.createArray(root, "y", y)
    h5file.createArray(root, "Xmean", Xmean)
    h5file.createArray(root, "Xstd", Xstd)
    h5file.close()


def get_parser():
    parser = argparse.ArgumentParser(description="""
    Creates a data set for supervised training on 3D patches
    """, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('list_trainset', metavar='list_trainset', type=str,
                        help='csv file of merged markers')
    parser.add_argument('indir', metavar='indir', type=str,
                        help='dir of the substacks')
    parser.add_argument('mergedmarkers_folder', metavar='mergedmarkers_folder', type=str,
                        help='Name of the folder in which merged markers are stored')
    parser.add_argument('outfile', metavar='outfile', type=str,
                        help='Name of file where the dataset will be saved')
    parser.add_argument('--extramargin', metavar='extramargin', dest='extramargin',
                        action='store', type=int, default=5,
                        help='Extra margin for convolution. Should be equal to (filter_size - 1)/2')
    parser.add_argument('--negatives', dest='negatives', action='store_true',
                        help='include "negative" (non cell) examples.')
    parser.add_argument('-s', '--size_patch', dest='size_patch',
                        action='store', type=int, default=4,
                        help='Input and output patches are cubes of side (2*size+1)**3')
    parser.add_argument('-l','--local_standardization', dest='local_standardization', action='store_true',
                        help='do the standardization locally or globally')
    parser.add_argument('--no-negatives', dest='negatives', action='store_false', help='Include only cell examples.')
    parser.set_defaults(negatives=False)
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
