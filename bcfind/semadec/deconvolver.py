from __future__ import print_function
import numpy as np
from progressbar import *
import scipy.ndimage.filters as filters
import theano
from pylearn2.space import CompositeSpace,VectorSpace
from pylearn2.models.autoencoder import *
from pylearn2.models.mlp import MLP
import tables
import timeit
from bcfind import extract_patch

from multiprocessing import Pool,Process,Lock,Array,sharedctypes
import os
import warnings
import cProfile

class VPrint(object):
    def __init__(self,verbose=True):
        self.verbose = verbose

    def __call__(self, *args, **kwargs):
        if self.verbose:
            argstring = " ".join([str(arg) for arg in args])
            if 'end' in kwargs:
                print(argstring,end=kwargs['end'])
            else:
                print(argstring)
            sys.stdout.flush()

vprint = VPrint()


def print_percentiles(x,name):
    vprint('Percentiles of', name)
    for v in np.percentile(x,[10,20,30,40,50,60,70,80,90]):
        vprint(' %.2f' % v, end='')
    vprint('')


def normalize(x, normalizer):
    vprint('Normalizing. Normalizer ranges in %.1f--%.1f' % (np.min(normalizer),np.max(normalizer)))
    n_x = x/normalizer
    vprint('rescaling..')
    mm = np.min(n_x)
    vprint('Min value', mm)
    if mm < 0:
        vprint('translating by', -mm)
        n_x = n_x - mm
    MM = np.max(n_x)
    vprint('Max value', MM)
    if MM > 1:
        vprint('dividing by', MM)
        n_x = n_x / MM
    return n_x


def logistic(x):
    return 1.0/(1.0+np.exp(-x))


class Predictor(object):
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        raise NotImplementedError('')


class PredictorFromPylearn2MLP(Predictor):
    def __init__(self, model):
        self.model = model
        self.X_sv = model.get_input_space().make_batch_theano()
        self.Xhat_sv = model.fprop(self.X_sv)
        self.reconstruction_function = theano.function([self.X_sv],[self.Xhat_sv], allow_input_downcast=True)


    def predict(self,X,batch_size=None, n_views=1):
        if batch_size is None:
            batch_size = X.shape[0]
        n_batches = X.shape[0]/batch_size
        Xhat = np.zeros((X.shape[0],X.shape[1]/n_views), dtype=np.float32)
        for i in xrange(n_batches):
            xx = self.reconstruction_function(X[i*batch_size:(i+1)*batch_size,:])
            Xhat[i*batch_size:(i+1)*batch_size] = xx[0]
        return Xhat


def make_predictor(model):
    if isinstance(model, DeepComposedAutoencoder) or isinstance(model, Autoencoder):
        return PredictorFromPylearn2Autoencoder(model)
    elif isinstance(model, MLP):
        return PredictorFromPylearn2MLP(model)
    else:
        raise TypeError('Model has type', type(model))




def process_loop(id_slice, extramargin, size_path, speedup, Xmean, Xstd, predictor):
    #ignore the PEP 3118 buffer warning
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)

        _np_tensor_3d = np.ctypeslib.as_array(np_tensor_3d)
        _reconstruction = np.ctypeslib.as_array(reconstruction)
        _normalizer = np.ctypeslib.as_array(normalizer)
        data = extract_patch.preprocess_2d(_np_tensor_3d, id_slice, extramargin, extramargin, speedup, Xmean, Xstd)
        pred_data = predictor.predict(data)
        sc= _np_tensor_3d.shape
        local_reconstruction = np.zeros(sc, dtype=np.float32)
        local_normalizer = np.zeros(sc, dtype=np.float32)
        local_reconstruction,local_normalizer = extract_patch.postprocess_2d(_np_tensor_3d, id_slice, pred_data, local_reconstruction, local_normalizer, extramargin, extramargin, speedup)
        with lock:
            #print('Lock acquired by process', id_slice)
            _normalizer+=local_normalizer
            _reconstruction+=local_reconstruction


def _init(t,r,n,l):
    global np_tensor_3d
    global lock
    global reconstruction
    global normalizer
    np_tensor_3d = t
    lock =l
    reconstruction=r
    normalizer=n


def filter_volume(np_tensor_3d, Xmean, Xstd,
                  extramargin=None, model=None, speedup=None, do_cython=False, do_multiprocessing=False):


    predictor = make_predictor(model)
    sc = np_tensor_3d.shape
    reconstruction = np.zeros(sc, dtype=np.float32)
    normalizer = np.zeros(sc, dtype=np.float32)
    n_points = (sc[1]-2*extramargin)*(sc[2]-2*extramargin)
    patchlen = (1+2*extramargin)**3
    rangez = range(extramargin, sc[0]-extramargin, speedup)
    rangey = range(extramargin, sc[1]-extramargin, speedup)
    rangex = range(extramargin, sc[2]-extramargin, speedup)
    n_points = len(rangex)*len(rangey)

    np_tensor_3d=(np_tensor_3d/255.).astype(np.float32)

    print('Patch size: %dx%dx%d (%d)' % (1+2*extramargin, 1+2*extramargin, 1+2*extramargin, (1+2*extramargin)**3))
    print('Will subsample jumping by',speedup,'voxels')


    if do_multiprocessing:
        num_processes=len(rangez)
        tmp = np.ctypeslib.as_ctypes(np_tensor_3d)
        np_tensor_3d = sharedctypes.Array(tmp._type_, tmp, lock=False)
        tmp = np.ctypeslib.as_ctypes(normalizer)
        normalizer = sharedctypes.Array(tmp._type_, tmp, lock=False)
        tmp = np.ctypeslib.as_ctypes(reconstruction)
        reconstruction = sharedctypes.Array(tmp._type_, tmp, lock=False)
        lock = Lock()
        pool= Pool(processes=num_processes, initializer=_init, initargs=(np_tensor_3d,reconstruction,normalizer,lock))
        start=timeit.default_timer()
        for id_slice in rangez:
            pool.apply_async(process_loop, args=(id_slice,extramargin,extramargin,speedup,Xmean,Xstd,predictor))
        pool.close()
        pool.join()
        normalizer=np.ctypeslib.as_array(normalizer)
        reconstruction=np.ctypeslib.as_array(reconstruction)
        np_tensor_3d=np.ctypeslib.as_array(np_tensor_3d)
        end=timeit.default_timer()
        print('time of processing: %f seconds' % (end-start,))

    elif do_cython:

        pbar = ProgressBar(widgets=['Processing %d slices (%d patches): ' % (len(rangez),len(rangex)*len(rangey)*len(rangez)),
                                    Percentage(), ' ', ETA()])
        for z0 in pbar(rangez):

            data= extract_patch.preprocess_2d(np_tensor_3d, z0, extramargin, extramargin, speedup, Xmean, Xstd)
            pred_data = predictor.predict(data)
            reconstruction,normalizer = extract_patch.postprocess_2d(np_tensor_3d, z0, pred_data, reconstruction, normalizer, extramargin, extramargin, speedup)
    else:

        pbar = ProgressBar(widgets=['Processing %d slices (%d patches): ' % (len(rangez),len(rangex)*len(rangey)*len(rangez)),
                                    Percentage(), ' ', ETA()])
        for z0 in pbar(rangez):
            data = np.zeros((n_points, patchlen),  dtype=np.float32)
            i = 0
            for y0 in rangey:
                for x0 in rangex:
                    patch = np_tensor_3d[z0-extramargin:z0+extramargin+1,
                                         y0-extramargin:y0+extramargin+1,
                                         x0-extramargin:x0+extramargin+1]
                    data[i] = np.reshape(patch, (patchlen,))
                    data[i] = (data[i] - Xmean)/Xstd
                    i += 1

            pred_data = predictor.predict(data)

            i = 0
            for y0 in rangey:
                for x0 in rangex:
                    reconstructed_patch = np.reshape(pred_data[i], (1+2*extramargin,1+2*extramargin,1+2*extramargin))
                    reconstruction[z0-extramargin:z0+extramargin+1,
                                   y0-extramargin:y0+extramargin+1,
                                   x0-extramargin:x0+extramargin+1] += reconstructed_patch
                    normalizer[z0-extramargin:z0+extramargin+1,
                               y0-extramargin:y0+extramargin+1,
                               x0-extramargin:x0+extramargin+1] += 1
                    i += 1



    vprint('Reconstructed and assembled',n_points*len(rangez),'patches, now clipping..')
    reconstruction = reconstruction[
        extramargin:sc[0]-extramargin,
        extramargin:sc[1]-extramargin,
        extramargin:sc[2]-extramargin]
    normalizer = normalizer[
        extramargin:sc[0]-extramargin,
        extramargin:sc[1]-extramargin,
        extramargin:sc[2]-extramargin]
    #print_percentiles(reconstruction, 'Reconstruction')
    reconstruction = normalize(reconstruction, normalizer)
    #print_percentiles(reconstruction, 'Normalized reconstruction')
    # # smooth aliasing due to speedup
    reconstruction = filters.gaussian_filter(reconstruction,sigma=speedup/3.0, mode='constant', cval=0.0)
    #print_percentiles(reconstruction, 'Normalized reconstruction')
    # reconstruction = filters.median_filter(reconstruction,size=3, mode='constant', cval=0.0)
    # print_percentiles(reconstruction, 'Medianized reconstruction')
    reconstruction = np.array(reconstruction*255.0, dtype=np.uint8)
    #print_percentiles(reconstruction, 'reconstruction 8 bit')
    return reconstruction
