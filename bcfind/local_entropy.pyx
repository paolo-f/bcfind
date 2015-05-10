
"""
Cython module to compute local entropy
"""

import numpy as np
cimport numpy as np
cimport cython
from libc cimport math

DTYPE = np.uint8
ctypedef np.uint8_t  DTYPE_t

cdef extern from "math.h":
    double log(double)


#@cython.profile(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def loop_compute_local_entropy(np.ndarray[DTYPE_t, ndim=3] first_view_tensor, np.ndarray[DTYPE_t, ndim=3] second_view_tensor, int extramargin, int size_patch, int nbins, int speedup):

    cdef Py_ssize_t x0,y0,z0
    cdef int start_x=extramargin
    cdef int start_y=extramargin
    cdef int start_z=extramargin
    cdef int stop_x=first_view_tensor.shape[2] - extramargin
    cdef int stop_y=first_view_tensor.shape[1] - extramargin
    cdef int stop_z=first_view_tensor.shape[0] - extramargin
    cdef int n_points=(stop_x-start_x)*(stop_y-start_y)
    cdef np.ndarray[np.float_t, ndim=3] entropy_mask_first_view
    cdef np.ndarray[np.float_t, ndim=3] entropy_mask_second_view
    cdef Py_ssize_t patchlen=size_patch*2+1
    cdef Py_ssize_t _size_patch=size_patch
    cdef np.ndarray[DTYPE_t, ndim=1] patch_first_view_ravel
    cdef np.ndarray[DTYPE_t, ndim=1] patch_second_view_ravel
    cdef Py_ssize_t i,j,k
    cdef int num_samples_patch=patchlen**3

    entropy_mask_first_view = np.zeros(shape=(first_view_tensor.shape[0],first_view_tensor.shape[1],first_view_tensor.shape[2]), dtype=np.float)
    entropy_mask_second_view = np.zeros(shape=(second_view_tensor.shape[0],second_view_tensor.shape[1],second_view_tensor.shape[2]), dtype=np.float)
    patch_first_view_ravel=np.zeros(num_samples_patch, dtype=np.uint8)
    patch_second_view_ravel=np.zeros(num_samples_patch, dtype=np.uint8)

    for z0 from start_z <= z0 < stop_z: #by speedup:
        for y0 from start_y <= y0 < stop_y:# by speedup:
            for x0 from start_x <= x0 < stop_x:# by speedup:


                for k from 0 <= k < patchlen:
                    for j from 0 <= j < patchlen:
                        for i from 0 <= i < patchlen:
                            patch_first_view_ravel[patchlen*patchlen*k+patchlen*j+i]=first_view_tensor[z0-size_patch+k,y0-size_patch+j,x0-size_patch+i]
                            patch_second_view_ravel[patchlen*patchlen*k+patchlen*j+i]=second_view_tensor[z0-size_patch+k,y0-size_patch+j,x0-size_patch+i]



                hist_first_view = _compute_1d_histogram(patch_first_view_ravel, nbins, 0, 256)
                hist_second_view = _compute_1d_histogram(patch_second_view_ravel, nbins, 0, 256)
                hist_first_view = _normalize_histogram(hist_first_view, num_samples_patch)
                hist_second_view = _normalize_histogram(hist_second_view, num_samples_patch)
                entropy_mask_first_view[z0,y0,x0]=_compute_entropy(hist_first_view,nbins)
                entropy_mask_second_view[z0,y0,x0]=_compute_entropy(hist_second_view,nbins)

    return entropy_mask_first_view, entropy_mask_second_view

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double  _compute_entropy(np.ndarray[np.double_t, ndim=1] normalized_hist, int nbins):
    '''
    Compute entropy of a patch
    '''
    cdef int start=0
    cdef Py_ssize_t i
    cdef float h
    cdef float entropy=0.
    for i from start <= i <nbins:
        h=normalized_hist[i]
        if h >0:
            entropy -= h* math.log(h)

    return entropy


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _normalize_histogram(np.ndarray[np.float_t,ndim=1] hist, int num_samples):
    '''
    Normalize the histogram 'hist'
    '''
    cdef int nbins=hist.shape[0]
    cdef Py_ssize_t i
    cdef float sum_hist=float(num_samples)
    for i from 0 <= i < nbins:
        hist[i]/=sum_hist

    return hist

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _compute_1d_histogram(np.ndarray[DTYPE_t, ndim=1] samples, int nbins, int hmin, int hmax):
    '''Return a uniformly binned histogram of samples.

    The histogram will span ``hmin`` to ``hmax``, and have a number of
    bins equal to ``nbins``.
    '''
    cdef np.ndarray[np.float_t, ndim=1] hist = np.zeros(shape=nbins, dtype=np.float)

    #cdef float bin_width = 1.0#(hmax - hmin) / float(nbins)
    cdef int num_samples=samples.shape[0]
    cdef Py_ssize_t i,bin_index
    for i from 0 <= i < num_samples:
        bin_index = samples[i]#int((samples[i] - hmin) / bin_width)
        hist[bin_index] += 1.
        #if 0 <= bin_index < nbins:
            #hist[bin_index] += 1
        #elif int(samples[i]) == hmax:
            #hist[-1] += 1  # Reproduce behavior of numpy
    return hist



