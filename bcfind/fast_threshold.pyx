# cython: profile=True
"""
Functions for image thresholding
"""
import numpy as np
cimport numpy as np
DTYPE = np.int
ctypedef np.int_t DTYPE_t

def two_kapur(np.ndarray[DTYPE_t, ndim=1] histogram):
    """two_kapur(histogram)

    Multi-thresholding using maximum entropy. Specialized verion for 2 divisions

    Parameters
    ----------
    histogram : array-like
        Image histogram

    Returns
    -------
    thresholds : list
        The two max entropy thresholds
    """
    cdef int N = 257
    cdef int max_t
    cdef float best_entropy = -1e50
    cdef float entropy
    cdef np.ndarray[np.float64_t, ndim=2] interval_entropy
    cdef np.ndarray[np.float64_t, ndim=2] hsums
    cdef int best_t0, best_t1, t0, t1, i, j

    interval_entropy = np.zeros((N,N), dtype=np.float64) - 1.0
    
    hsums = np.zeros((N,N), dtype=np.float64)
    for i in range(0,N):
        for j in range(i+1,N):
            hsums[i,j] = hsums[i,j-1] + histogram[j-1]

    max_t = len(histogram)

    # For each division, compute the entropies and sum them;
    # Determine the interval with maximum sum of entropies
    for t0 in range(1, max_t-1):
        for t1 in range(t0+1,max_t):
            # print('debug', [t0, t1])
            entropy = 0.0
            # print('  -- calling', 0, t0)
            entropy += _compute_interval_entropy(histogram,interval_entropy, hsums, 0, t0)
            # print('  -- calling', t0, t1)
            entropy += _compute_interval_entropy(histogram,interval_entropy, hsums, t0, t1)
            # print('  -- calling', t1, max_t)
            entropy += _compute_interval_entropy(histogram,interval_entropy, hsums, t1, max_t)
            # print('Multi_Kapur: i-th entropy i=',i,'e=',entropy,'Interval (thresholds)=',interval)
            if best_entropy < entropy:
                best_entropy = entropy
                best_t0 = t0
                best_t1 = t1
    return [best_t0, best_t1]

cdef extern from "math.h":
    double log(double)

cpdef double _compute_interval_entropy(np.ndarray[np.int64_t, ndim=1] histogram,
                                          np.ndarray[np.float64_t, ndim=2] interval_entropy,
                                          np.ndarray[np.float64_t, ndim=2] hsums,
                                          int begin, int end):
    """
    Compute interval entropy from begin (included) to end (excluded)
    """
    cdef np.float_t ie, entropy, hSum, a
    cdef Py_ssize_t i, h

    ie = interval_entropy[begin][end]
    if ie < 0:
        hSum = hsums[begin,end]
        entropy = 0
        for i in range(begin, end):
            h = histogram[i]
            if h > 0:
                a = histogram[i] / hSum
                entropy -= a * log(a)
        ie = entropy
        interval_entropy[begin][end] = ie
    return ie

