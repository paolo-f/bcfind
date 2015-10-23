# -*- coding: utf-8 -*-
import numpy as np
import os
from PIL import Image
import glob
import argparse
import tables
import math
from scipy import special


#Arun algorithm
def arun_method(X, T): # Assumes both or Nx3 matrices
    """
    Method that estimates a rigid transformation between two point sets, 
    solving  a constrained least squares problem,
    based on the computation of the Singular Value Decomposition (SVD)

    Parameters
    ----------

    X : numpy array of shape (n_points, 3)
	source point cloud
    T : numpy array of shape (n_points, 3)
	target point cloud

    Returns
    -------

    R : numpy array of shape (3, 3)
	rotational component of the estimated rigid transformation
    t : numpy array of shape (3)
	translational component of the estimated rigid transformation

    References
    ----------
    Arun, K.S., Huang, T.S., Blostein, S.D.: Least-squares fitting of two 3-D point sets. Pattern
    Analysis and Machine Intelligence, IEEE Transactions on (5), 698–700 (1987).
    """
    X_mean = X.mean(0)
    T_mean = T.mean(0)
    # Compute covariance
    H = reduce(lambda s, (a,b) : s + np.outer(a, b), zip(X - X_mean, T - T_mean), np.zeros((3,3)))
    u, s, v = np.linalg.svd(H)
    R = v.T.dot(u.T) # Rotation
    t = - R.dot(X_mean) + T_mean # Translation
    return R,t

def do_ransac(X, T, mss=3, max_iterations=1000, tolerance=3.0):
    """
    Method that estimates the rigid transformation (Horn Method) between 3D Points using RANSAC.

    Parameters
    ----------

    X : numpy array of shape (n_points, 3)
	source point cloud
    T : numpy array of shape (n_points, 3)
	target point cloud
    mss : int
	minimal sample set to build the model (Default: 3)
    max_iteration: int
	maximum number of iterations of the RANSAC procedure (Default: 1000)
    tolerance: float
	Error tolerance (Default: 3.0)

    Returns
    -------

    id_inliers: indices of the pairs of points belonging to the consensus set


    References
    ----------
    Fischler, M.A., Bolles, R.C.: Random sample consensus: a paradigm for model fitting with
    applications to image analysis and automated cartography. Communications of the ACM
    24(6), 381–395 (1981)
    """
    consensus_set = 0
    consensus_threshold = X.shape[0]
    best_w = np.ones((1, X.shape[0]), dtype=np.double)
    for i in xrange(max_iterations):
        w = np.zeros((1, X.shape[0]), dtype=np.double).ravel()
        idx = np.random.choice(X.shape[0], mss, replace=False)
        w[idx] = 1
        _ ,_ , error_vector, _, _ = horn_method(X, T, w)
        if len(error_vector[error_vector < tolerance])> consensus_set:
            consensus_set =  len(error_vector[error_vector < tolerance])
            id_inliers = np.where(error_vector < tolerance)[0]
            if consensus_set >= consensus_threshold:
                break
    return id_inliers

def findBestRigidBodyEstimation(markers_input, markers_target):
    """
    Method that estimates the rigid transformation between two marker files.

    Parameters
    ----------
    
    markers_input: str
	filename of the list of markers, extracted from the input view
    markers_output: str
	filename of the list of markers, extracted from the target view

    Returns
    -------

    R : numpy array of shape (3, 3)
	rotational component of the estimated rigid transformation, after computing RANSAC
    t : numpy array of shape (3)
	translational component of the estimated rigid transformation, after computing RANSAC
    """

    grammar = ('#' + restOfLine).suppress() | Group(Combine(Word(nums) + Optional("." + Word(nums))) + Suppress(",") + Combine(Word(nums) + Optional("." + Word(nums))) + Suppress(",") + Combine(Word(nums) + Optional("." + Word(nums)))) + restOfLine.suppress()

    f = open(markers_input, 'r')
    X = []
    for line in f:
        parsed_line = grammar.parseString(line)
        if len(parsed_line) == 1:
            X.append(np.array(parsed_line[0][:]).astype(np.float))
    X = np.array(X)

    f = open(markers_target, 'r')
    T = []
    for line in f:
        parsed_line = grammar.parseString(line)
        if len(parsed_line) == 1:
            T.append(np.array(parsed_line[0][:]).astype(np.float))
    T = np.array(T)

    id_inliers = do_ransac(X, T)

    best_w = np.zeros((1, X.shape[0]), dtype=np.double).ravel()
    best_w[id_inliers] = 1
    R, t, error_vector, transfX, diff = horn_method(X, T, best_w)

    return R, t


def horn_method(X, T, weights, verbose=False):
    """
    Method, based on quaternions, that estimates a rigid transformation between two point sets, 
    solving  a weighted constrained least squares problem

    Parameters
    ----------

    X : numpy array of shape (n_points, 3)
	source point cloud
    T : numpy array of shape (n_points, 3)
	target point cloud

    Returns
    -------

    R : numpy array of shape (3, 3)
	rotational component of the estimated rigid transformation
    t : numpy array of shape (3)
	translational component of the estimated rigid transformation

    References
    ----------
    Horn, B.K.P., Hilden, H., Negahdaripour, S.: Closed-form solution of absolute orientation
    using orthonormal matrices. JOURNAL OF THE OPTICAL SOCIETY AMERICA 5(7), 1127–
    1135 (1988)
    """
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
