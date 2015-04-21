#!/usr/bin/env python

"""
Scripts that computs the rigid transformation between two 3D point clouds and then rotates a 3D input tensor to a reference volume
"""
__author__ = 'paciscopi'

import numpy as np
import os
import Image
import glob
from pyparsing import Word, nums, restOfLine, Suppress, Group, Combine, Optional
import timeit
import argparse
import tables
import math
from scipy import special


#Arun algorithm
def get_rigid(src, dst): # Assumes both or Nx3 matrices
    src_mean = src.mean(0)
    dst_mean = dst.mean(0)
    # Compute covariance
    H = reduce(lambda s, (a,b) : s + np.outer(a, b), zip(src - src_mean, dst - dst_mean), np.zeros((3,3)))
    u, s, v = np.linalg.svd(H)
    R = v.T.dot(u.T) # Rotation
    t = - R.dot(src_mean) + dst_mean # Translation
    return R,t

def findBestRigidBodyEstimation(markers_input, markers_output):


    grammar = ('#' + restOfLine).suppress() | Group(Combine(Word(nums) + Optional("." + Word(nums))) + Suppress(",") + Combine(Word(nums) + Optional("." + Word(nums))) + Suppress(",") + Combine(Word(nums) + Optional("." + Word(nums)))) + restOfLine.suppress()

    f = open(markers_input, 'r')
    X = []
    for line in f:
        #print(line)
        parsed_line = grammar.parseString(line)
        if len(parsed_line) == 1:
            X.append(np.array(parsed_line[0][:]).astype(np.float))
    X = np.array(X)

    f = open(markers_output, 'r')
    T = []
    for line in f:
        parsed_line = grammar.parseString(line)
        if len(parsed_line) == 1:
            T.append(np.array(parsed_line[0][:]).astype(np.float))
    T = np.array(T)


    #RANSAC algorithm
    mss = 3
    max_iterations = 10000
    consensus_set = 0
    tolerance = 5.0
    #consensus_threshold = int(4 * X.shape[0]/5)
    consensus_threshold = X.shape[0]
    best_w = np.ones((1, X.shape[0]), dtype=np.double)
    for i in xrange(max_iterations):
        w = np.zeros((1, X.shape[0]), dtype=np.double).ravel()
        idx = np.random.choice(X.shape[0], mss, replace=False)
        w[idx] = 1
        _ ,_ , error_vector, _, _ = rigidMotionEstimation(X, T, w)
        if len(error_vector[error_vector < tolerance])> consensus_set:
            consensus_set =  len(error_vector[error_vector < tolerance])
            id_inliers = np.where(error_vector < tolerance)[0]
            print("consensus set ", consensus_set)
            if consensus_set >= consensus_threshold:
                break


    best_w = np.zeros((1, X.shape[0]), dtype=np.double).ravel()
    best_w[id_inliers] = 1
    R, t, error_vector, transfX, diff = rigidMotionEstimation(X, T, best_w)


    print error_vector
    print len(error_vector[error_vector < tolerance])

    return R, t


#Horn algorithm
def rigidMotionEstimation(X, T, weights, verbose=False):
    normalized_weights = weights / np.sum(weights)

    normalized_centroid_X = np.dot(np.transpose(X), normalized_weights)
    normalized_centroid_T = np.dot(np.transpose(T), normalized_weights)

    if verbose:
        print "normalized input centroid: " + str(normalized_centroid_X)
        print "normalized target centroid: " + str(normalized_centroid_T)
        print "normalized weights: " + str(normalized_weights)

    Xnew = np.array([p - normalized_centroid_X for p in X])
    Tnew = np.array([p - normalized_centroid_T for p in T])
    Xnew_weighted = Xnew * np.sqrt(normalized_weights)[:, np.newaxis]
    Tnew_weighted = Tnew * np.sqrt(normalized_weights)[:, np.newaxis]

    M = np.dot(np.transpose(Xnew_weighted), Tnew_weighted)

    if verbose:
        print "Xnew: \n" + str(Xnew)
        print "Tnew: \n" + str(Tnew)
        print "Matrix M: \n" + str(M)

    N = np.array([[M[0, 0] + M[1, 1] + M[2, 2], M[1, 2] - M[2, 1], M[2, 0] - M[0, 2], M[0, 1] - M[1, 0]],
                  [M[1, 2] - M[2, 1], M[0, 0] - M[1, 1] - M[2, 2], M[0, 1] + M[1, 0], M[2, 0] + M[0, 2]],
                  [M[2, 0] - M[0, 2], M[0, 1] + M[1, 0], -M[0, 0] + M[1, 1] - M[2, 2], M[1, 2] + M[2, 1]],
                  [M[0, 1] - M[1, 0], M[2, 0] + M[0, 2], M[1, 2] + M[2, 1], -M[0, 0] - M[1, 1] + M[2, 2]]])

    w, v = np.linalg.eig(N)

    if verbose:
        print "N: \n" + str(N)
        print "w: \n" + str(w)
        print "v: \n" + str(v)

    argMaxEigenValue = np.argmax(np.real(w))

    maxEigenValueVector = np.real(v[:, argMaxEigenValue])

    sgn = np.sign(maxEigenValueVector[np.argmax(np.abs(maxEigenValueVector))])

    maxEigenValueVector = maxEigenValueVector * sgn

    if verbose:
        print "maxEigenValueVector: \n" + str(maxEigenValueVector)

    q0 = maxEigenValueVector[0]
    qx = maxEigenValueVector[1]
    qy = maxEigenValueVector[2]
    qz = maxEigenValueVector[3]
    v = maxEigenValueVector[1:4]

    Z = np.array([[q0, -qz, qy],
                  [qz, q0, -qx],
                  [-qy, qx, q0]])

    R = np.outer(v, v) + np.dot(Z, Z)

    t = normalized_centroid_T - np.dot(R, normalized_centroid_X)

    if verbose:
        print "R: " + str(R)
        print "t: " + str(t)

    Xfit = np.dot(R, np.transpose(X))
    for i in xrange(Xfit.shape[1]):
        Xfit[:, i] = Xfit[:, i] + t

    diff = np.transpose(T) - Xfit

    err = []
    for i in xrange(diff.shape[1]):
        err.append(np.sqrt(np.dot(diff[:, i], diff[:, i])))
    weighted_err = [err[i] * np.sqrt(normalized_weights)[i] for i in xrange(diff.shape[1])]
    error = np.linalg.norm(weighted_err) * np.sum(weights)

    if verbose:
        print "err: \n" + str(err)
        print "weighted_err: \n" + str(weighted_err)

    transfX = np.array([np.dot(R, x) + t for x in X])


    residual = T - transfX

    error_vector=[]
    for i in xrange(transfX.shape[0]):
        error_vector.append(np.sqrt(np.dot(residual[i, :], residual[i, :])))
    error_vector=np.array(error_vector)

    return R, t, error_vector, transfX, np.transpose(diff)

#@profile
def blending(input_view, target_view, transformed_view, R, t):

    t_start = timeit.default_timer()

    suffix = ".tif"
    convert_to_gray = True


    if (os.path.isdir(target_view)):
        filesTarget = sorted([target_view + '/' + f for f in os.listdir(target_view) if f[0] != '.' and f.endswith(suffix)])
        img_z = np.asarray(Image.open(filesTarget[0]))
        height_target, width_target = img_z.shape
        depth_target = len(filesTarget)
    elif (tables.isHDF5File(target_view)):
        hf5_target_view = tables.openFile(target_view, 'r')
        depth_target, height_target, width_target = hf5_target_view.root.full_image.shape
        hf5_target_view.close()
    else:
        print target_view + " is neither a hdf5 file nor a valid directory"
        sys.exit(1)

    if (os.path.isdir(input_view)):
        filesInput = sorted([input_view + '/' + f for f in os.listdir(input_view) if f[0] != '.' and f.endswith(suffix)])
        img_z = np.asarray(Image.open(filesInput[0]))
        height_input, width_input = img_z.shape
        depth_input = len(filesInput)
        pixels_input = np.empty(shape=(depth_input, height_input, width_input), dtype=np.uint8)
        for z, image_file in enumerate(filesInput):
            img_z = Image.open(image_file)
            if convert_to_gray:
                img_z = img_z.convert('L')
            pixels_input[z, :, :] = np.asarray(img_z)
            #print str(z)
    elif (tables.isHDF5File(input_view)):
        hf5_input_view = tables.openFile(input_view, 'r')
        depth_input, height_input, width_input = hf5_input_view.root.full_image.shape
        pixels_input = hf5_input_view.root.full_image[0:depth_input,0:height_input,0:width_input]
        hf5_input_view.close()
    else:
        print input_view + " is neither a hdf5 file nor a valid directory"
        sys.exit(1)



    t_stop = timeit.default_timer()
    print('Read input stack completed in %s secs.' %(t_stop - t_start) )

    pixels_transformed_input = np.empty((depth_target, height_target, width_target), dtype=np.uint8)

    if not os.path.exists(transformed_view):
        os.makedirs(transformed_view)
    else:
        files = glob.glob(transformed_view + '/*')
        for f in files:
            os.remove(f)

    total_start = timeit.default_timer()


    coords_2d_target = np.vstack(np.indices((width_target,height_target)).swapaxes(0,2).swapaxes(0,1))
    invR = np.linalg.inv(R)
    invR_2d_transpose = np.transpose(np.dot(invR[:, 0:2], np.transpose(coords_2d_target - t[0:2])))
    for z in xrange(0, depth_target, 1): #depth_target
        print "...transforming slice " + str(z)
        start = timeit.default_timer()
        R_t_3d = np.transpose(invR_2d_transpose + invR[:, 2] * (z - t[2]))



        good_indices = np.arange(R_t_3d.shape[1])
        good_indices = good_indices[(R_t_3d[0, :] > 0) * (R_t_3d[1, :] > 0) * (R_t_3d[2, :] > 0) * (R_t_3d[0, :] < (width_input - 1)) * (R_t_3d[1, :] < (height_input - 1)) * (R_t_3d[2, :] < (depth_input - 1))]

        R_t_3d = R_t_3d.take(good_indices,axis=1)
        R_t_3d = np.round(R_t_3d).astype(int)
        coords_2d_target_tmp = coords_2d_target.take(good_indices, axis=0)


        coords_3d_target_tmp = np.hstack((coords_2d_target_tmp, np.ones((coords_2d_target_tmp.shape[0], 1)).astype(int) * z))


        pixels_transformed_input[coords_3d_target_tmp[:, 2], coords_3d_target_tmp[:, 1], coords_3d_target_tmp[:, 0]] = pixels_input[R_t_3d[2, :], R_t_3d[1, :], R_t_3d[0, :]]


        im = Image.fromarray(np.uint8(pixels_transformed_input[z]))
        im.save(transformed_view + '/slice_' + str(z).zfill(4) + ".tif", 'TIFF')
        stop = timeit.default_timer()
        print "time: " + str(stop - start)



    total_stop = timeit.default_timer()
    print "total time transformation: " + str(total_stop - total_start)



def main(args):
    total_start= timeit.default_timer()
    R, t  = findBestRigidBodyEstimation(args.markers_input,args.markers_target)
    print('R: ', R)
    print('t: ', t)
    total_stop = timeit.default_timer()
    print "total time transformation: " + str(total_stop - total_start)
    if args.transform:
        blending(args.input_view, args.target_view, args.transformed_view, R, t)



def get_parser():
    parser = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_view', metavar='input_view', type=str,
                        help='dir of the input tensor')
    parser.add_argument('target_view', metavar='target_view', type=str,
                        help='dir of the reference tensor')
    parser.add_argument('transformed_view', metavar='transformed_view', type=str,
                        help='output tensor of the 3D rigid transformation')
    parser.add_argument('markers_input', metavar='markers_input', type=str,
                        help='markers of the input tensor')
    parser.add_argument('markers_target', metavar='markers_target', type=str,
                        help='markers of the reference tensor')
    parser.add_argument('--transform', dest='transform', action='store_true',
                        help='transform the input tensor according to the estimated rigid transformation.')
    parser.set_defaults(transform=False)

    return parser



if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
