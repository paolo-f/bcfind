#!/usr/bin/env python
"""
Creates a training set for semantic deconvolution.
"""

from __future__ import print_function
import numpy as np
import tables
import argparse
import progressbar as pb

from bcfind import volume
from scipy.spatial import cKDTree
from bcfind.semadec import imtensor
import scipy.ndimage.filters as gfilter



def inside_margin(c, substack):
    """ Are we inside the safe region?"""
    m = substack.plist['Margin']/2
    return min(c.x-m,c.y-m,c.z-m,substack.info['Width']-m-c.x,substack.info['Height']-m-c.y,substack.info['Depth']-m-c.z)



def make_dataset(tensorimage, ss, C, L=12, size=None, save_tiff_files=False, negatives=False, margin=None):
    hf5 = tables.openFile(tensorimage, 'r')
    X0,Y0,Z0 = ss.info['X0'], ss.info['Y0'], ss.info['Z0']
    origin = (Z0, Y0, X0)
    H,W,D = ss.info['Height'], ss.info['Width'], ss.info['Depth']
    ss_shape = (D,H,W)
    print('Loading data for substack', ss.substack_id)
    np_tensor_3d = hf5.root.full_image[origin[0]:origin[0]+ss_shape[0],
                                       origin[1]:origin[1]+ss_shape[1],
                                       origin[2]:origin[2]+ss_shape[2]]
    X = []
    y = []
    patchlen = (1+2*size)**3
    print('Preparing..')
    kdtree = cKDTree(np.array([[c.x,c.y,c.z] for c in C]))
    nrej_intensity = 0
    nrej_near = 0
    target_tensor_3d = np.zeros(np_tensor_3d.shape)
    for c in C:
        if inside_margin(c,ss) > 0:
            target_tensor_3d[c.z, c.y, c.x] = 1
    target_tensor_3d = gfilter.gaussian_filter(target_tensor_3d, sigma=1.0,
                                               mode='constant', cval=0.0,
                                               truncate=1.5)
    # undo scipy normalization of the gaussian filter
    target_tensor_3d = (target_tensor_3d / np.max(target_tensor_3d))
    save_tiff_files = False
    if save_tiff_files:
        minz = int(ss.info['Files'][0].split('full_')[1].split('.tif')[0])
        target_tensor_3d = np.array(target_tensor_3d*255.0, dtype=np.uint8)
        imtensor.save_tensor_as_tif(target_tensor_3d,'/tmp/show_target/'+ss.substack_id, minz)
    print('Patches of size', 2*size+1, 'patchlen', patchlen)
    print('Negatives will','' if negatives else 'not', 'be included')
    pbar = pb.ProgressBar(widgets=['Making patches for %d points: ' % len(C), pb.Percentage()],
                          maxval=len(C)).start()
    nrej_intensity = 0
    nrej_near = 0
    for pbi,c in enumerate(C):
        if inside_margin(c,ss) > 0:
            n_neg = 0
            cx = int(c.x)
            cy = int(c.y)
            cz = int(c.z)
            print(cx,cy,cz,margin)
            for x0 in range(cx-L,cx+L+1,3):
                for y0 in range(cy-L,cy+L+1,3):
                    for z0 in range(cz-L,cz+L+1,3):
                        if np.random.rand(1)[0] > 0:  # 0.5: #0.8:
                            print(z0,y0,x0)
                            patch = np_tensor_3d[z0-size:z0+size+1,y0-size:y0+size+1,x0-size:x0+size+1]
                            print(patch.shape)
                            X.append(np.reshape(patch, (patchlen,)))
                            ypatch = target_tensor_3d[z0-size:z0+size+1,y0-size:y0+size+1,x0-size:x0+size+1]
                            y.append(np.reshape(ypatch, (patchlen,)))
                            n_neg += 1  # Sample as many negatives as positives
            if negatives:
                while n_neg > 0:
                    g = [margin/2,margin/2,margin/2] + np.random.rand(3) * [D-margin,H-margin,W-margin]
                    g = g.astype(int)
                    nbrs = kdtree.query_ball_point(g, r=30)
                    if len(nbrs) == 0:
                        patch = np_tensor_3d[g[0]-size:g[0]+size+1,g[1]-size:g[1]+size+1,g[2]-size:g[2]+size+1]
                        if np.mean(patch) > 10:  # or np.random.rand(1)[0] > 0.95:
                            X.append(np.reshape(patch, (patchlen,)))
                            ypatch = target_tensor_3d[z0-size:z0+size+1,y0-size:y0+size+1,x0-size:x0+size+1]
                            y.append(np.reshape(ypatch, (patchlen,)))
                            # y.append(np.zeros(patchlen))
                            n_neg -= 1
                        else:
                            nrej_intensity += 1
                    else:
                        nrej_near += 1
            else:
                pass
        pbar.update(pbi+1)
    pbar.finish()
    print('Rejected:', nrej_intensity, 'by intensity and', nrej_near, 'by distance')
    hf5.close()
    return X, y

def main(args):
    data = []
    target = []
    for substack_id in args.substack_ids:
        substack = volume.SubStack(args.indir,substack_id)
        gt_markers = args.indir+'/'+substack_id+'-GT.marker'
        print('Loading ground truth markers from',gt_markers)
        C = substack.load_markers(gt_markers,from_vaa3d=True)
        for c in C:
            c.x -= 1
            c.y -= 1
            c.z -= 1
        sdata,starget = make_dataset(args.tensorimage, substack, C, size=args.size, negatives=args.negatives, margin=args.margin)
        data.extend(sdata)
        target.extend(starget)
    X = np.zeros((len(data),data[0].shape[0]), dtype=np.float32)
    y = np.zeros((len(data),data[0].shape[0]), dtype=np.float32)
    pbar = pb.ProgressBar(widgets=['Converting to 32-bit numpy array %d examples: ' % X.shape[0], pb.Percentage()],
                          maxval=X.shape[0]).start()
    for i in range(X.shape[0]):
        X[i] = (data[i]/255.0).astype(np.float32)
        y[i] = target[i].astype(np.float32)
        pbar.update(i+1)
    pbar.finish()
    print('Data set shape:', X.shape, 'size:', X.nbytes/(1024*1024), 'MBytes')
    print('target shape:', y.shape, 'size:', y.nbytes/(1024*1024), 'MBytes')

    print('Standardizing')
    Xmean = X.mean(axis=0)
    Xstd = X.std(axis=0)
    X = (X - Xmean) / Xstd

    print('Saving training data to',args.outfile)
    h5file = tables.openFile(args.outfile, mode='w', title="Training set")
    root = h5file.root
    h5file.createArray(root, "X", X)
    h5file.createArray(root, "y", y)
    h5file.createArray(root, "Xmean", Xmean)
    h5file.createArray(root, "Xstd", Xstd)
    h5file.close()


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('indir', metavar='indir', type=str,
                        help='needs indir/info.json, substacks, e.g. indir/000, and ground truth, e.g. indir/000-GT.marker')
    parser.add_argument('tensorimage', metavar='tensorimage', type=str,
                        help='HDF5 file containing the whole stack')
    parser.add_argument('outfile', metavar='outfile', type=str,
                        help='Name of the HDF5 file where results will be saved')
    parser.add_argument('substack_ids', metavar='substack_ids', type=str, nargs='+',
                        help='substack identifier, e.g. 010608')
    parser.add_argument('-s', '--size', dest='size',
                        action='store', type=int, default=6,
                        help='Input and output patches are cubes of side (2*size+1)**3')
    parser.add_argument('-m', '--margin', dest='margin', type=int, default=40, help='Overlap between adjacent substacks')
    parser.add_argument('--negatives', dest='negatives', action='store_true', help='include "negative" (non cell) examples.')
    parser.add_argument('--no-negatives', dest='negatives', action='store_false', help='Include only cell examples.')
    parser.set_defaults(negatives=False)
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
