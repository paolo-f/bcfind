import numpy as np
cimport numpy as np
cimport cython
from libc cimport math

DTYPE = np.float32
ctypedef np.float32_t  DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
#def preprocess_2d(np.ndarray[DTYPE_t, ndim=3] tensor, Py_ssize_t z0, int extramargin, int size_patch, int speedup):
def preprocess_2d(np.ndarray[DTYPE_t, ndim=3] tensor, Py_ssize_t z0, int extramargin, int size_patch, int speedup, np.ndarray[DTYPE_t, ndim=1] mean, np.ndarray[DTYPE_t, ndim=1] std):

    cdef Py_ssize_t x0,y0
    cdef int start_x=extramargin
    cdef int start_y=extramargin
    cdef int start_z=extramargin
    cdef int stop_x=tensor.shape[2] - extramargin
    cdef int stop_y=tensor.shape[1] - extramargin
    cdef int stop_z=tensor.shape[0] - extramargin
    cdef np.ndarray[DTYPE_t, ndim=2] flat_data
    cdef np.ndarray[DTYPE_t, ndim=1] patch_ravel
    cdef Py_ssize_t patchlen=size_patch*2+1
    cdef Py_ssize_t _size_patch=size_patch
    cdef Py_ssize_t i,j,k
    cdef int num_samples_patch=patchlen**3
    cdef int patchlen_square=patchlen**2
    cdef int len_x = (stop_x -start_x + speedup -1)/speedup
    cdef int len_y = (stop_y -start_y + speedup -1)/speedup
    cdef int n_points=len_x*len_y

    flat_data = np.empty(( n_points, num_samples_patch), dtype=np.float32)
    patch_ravel = np.empty(num_samples_patch,dtype=np.float32)

    cdef Py_ssize_t iter_xy=0
    cdef Py_ssize_t x0__size_patch
    cdef Py_ssize_t y0__size_patch
    cdef Py_ssize_t z0__size_patch=z0 - _size_patch
    for y0 from start_y <= y0 < stop_y by speedup:
       y0__size_patch=y0-_size_patch
       for x0 from start_x <= x0 < stop_x by speedup:
           x0__size_patch=x0-_size_patch

           for k from 0 <= k < patchlen:
               for j from 0 <= j < patchlen:
                   for i from 0 <= i < patchlen:
                     patch_ravel[patchlen_square*k+patchlen*j+i] = (tensor[z0__size_patch+k,y0__size_patch+j,x0__size_patch+i] - mean[patchlen_square*k+patchlen*j+i])/std[patchlen_square*k+patchlen*j+i]


           flat_data[iter_xy] = patch_ravel
           iter_xy+=1

    return flat_data


@cython.boundscheck(False)
@cython.wraparound(False)
def postprocess_2d(np.ndarray[DTYPE_t, ndim=3] tensor, Py_ssize_t z0, np.ndarray[DTYPE_t, ndim=2] flat_data, np.ndarray[DTYPE_t, ndim=3] reconstruction, np.ndarray[DTYPE_t, ndim=3] normalizer,  int extramargin, int size_patch, int speedup):

    cdef Py_ssize_t x0,y0
    cdef int start_x=extramargin
    cdef int start_y=extramargin
    cdef int start_z=extramargin
    cdef int stop_x=tensor.shape[2] - extramargin
    cdef int stop_y=tensor.shape[1] - extramargin
    cdef int stop_z=tensor.shape[0] - extramargin
    cdef np.ndarray[DTYPE_t, ndim=1] patch_ravel
    cdef Py_ssize_t patchlen=size_patch*2+1
    cdef Py_ssize_t _size_patch=size_patch
    cdef Py_ssize_t i,j,k
    cdef int num_samples_patch=patchlen**3
    cdef int patchlen_square=patchlen**2

    patch_ravel = np.empty(num_samples_patch,dtype=np.float32)

    cdef Py_ssize_t iter_xy=0
    cdef Py_ssize_t x0__size_patch
    cdef Py_ssize_t y0__size_patch
    cdef Py_ssize_t z0__size_patch= z0 - _size_patch
    for y0 from start_y <= y0 < stop_y by speedup:
       y0__size_patch = y0 - _size_patch
       for x0 from start_x <= x0 < stop_x by speedup:
            x0__size_patch = x0 - _size_patch

            patch_ravel = flat_data[iter_xy]
            for k from 0 <= k < patchlen:
                for j from 0 <= j < patchlen:
                    for i from 0 <= i < patchlen:
                        reconstruction[z0__size_patch+k,y0__size_patch+j,x0__size_patch+i] += patch_ravel[patchlen_square*k+patchlen*j+i]
                        normalizer[z0__size_patch+k,y0__size_patch+j,x0__size_patch+i] += 1

            iter_xy+=1


    return reconstruction, normalizer

