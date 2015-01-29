from __future__ import print_function
import numpy as np
from progressbar import *
import scipy.ndimage.filters as filters
import theano

from pylearn2.models.autoencoder import *
from pylearn2.models.mlp import MLP
import timeit


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
        self.X_sv.name = 'X_sv'
        self.Xhat_sv = model.fprop(self.X_sv)
        self.reconstruction_function = theano.function([self.X_sv],[self.Xhat_sv])

    def predict(self,X,batch_size=None, n_views=1):
        if batch_size is None:
            batch_size = X.shape[0]
        n_batches = X.shape[0]/batch_size
        Xhat = np.zeros((X.shape[0],X.shape[1]/n_views))
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


def filter_volume(np_tensor_3d, Xmean, Xstd,
                  extramargin=None, model=None, speedup=None):
    predictor = make_predictor(model)
    sc = np_tensor_3d.shape
    reconstruction = np.zeros(sc)
    n_points = (sc[1]-2*extramargin)*(sc[2]-2*extramargin)
    patchlen = (1+2*extramargin)**3
    rangez = range(extramargin, sc[0]-extramargin, speedup)
    rangey = range(extramargin, sc[1]-extramargin, speedup)
    rangex = range(extramargin, sc[2]-extramargin, speedup)
    n_points = len(rangex)*len(rangey)

    normalizer = np.zeros(sc)
    print('Patch size: %dx%dx%d (%d)' % (1+2*extramargin, 1+2*extramargin, 1+2*extramargin, (1+2*extramargin)**3))
    print('Will subsample jumping by',speedup,'voxels')
    pbar = ProgressBar(widgets=['Processing %d slices (%d patches): ' % (len(rangez),len(rangex)*len(rangey)*len(rangez)),
                                Percentage(), ' ', AdaptiveETA()])
    ndone = 0
    for z0 in pbar(rangez):
        data = np.zeros((n_points, patchlen))
        # print('Collecting data for z=',z0, ': ', data.nbytes/(2**20), 'MBytes')
        i = 0
        for y0 in rangey:
            for x0 in rangex:
                patch = np_tensor_3d[z0-extramargin:z0+extramargin+1,
                                     y0-extramargin:y0+extramargin+1,
                                     x0-extramargin:x0+extramargin+1]
                data[i] = np.reshape(patch, (patchlen,))/255.0
                data[i] = (data[i] - Xmean) / Xstd
                i += 1
        pred_data = predictor.predict(data)
        i = 0
        for y0 in rangey:
            for x0 in rangex:
                reconstructed_patch = np.reshape(pred_data[i], (1+2*extramargin,1+2*extramargin,1+2*extramargin))
                # print_percentiles(reconstructed_patch, 'Reconstructed patch at (%d,%d,%d)'%(z0,y0,x0))
                # reconstructed_patch = np.reshape(data[i], (1+2*extramargin,1+2*extramargin,1+2*extramargin))
                reconstruction[z0-extramargin:z0+extramargin+1,
                               y0-extramargin:y0+extramargin+1,
                               x0-extramargin:x0+extramargin+1] += reconstructed_patch
                normalizer[z0-extramargin:z0+extramargin+1,
                           y0-extramargin:y0+extramargin+1,
                           x0-extramargin:x0+extramargin+1] += 1
                i += 1
                ndone += 1
    vprint('Reconstructed and assembled',ndone,'patches, now clipping..')
    reconstruction = reconstruction[
        extramargin:sc[0]-extramargin,
        extramargin:sc[1]-extramargin,
        extramargin:sc[2]-extramargin]
    normalizer = normalizer[
        extramargin:sc[0]-extramargin,
        extramargin:sc[1]-extramargin,
        extramargin:sc[2]-extramargin]
    print_percentiles(reconstruction, 'Reconstruction')
    reconstruction = normalize(reconstruction, normalizer)
    print_percentiles(reconstruction, 'Normalized reconstruction')
    # # smooth aliasing due to speedup
    reconstruction = filters.gaussian_filter(reconstruction,sigma=speedup/3.0, mode='constant', cval=0.0)
    print_percentiles(reconstruction, 'Normalized reconstruction')
    # reconstruction = filters.median_filter(reconstruction,size=3, mode='constant', cval=0.0)
    # print_percentiles(reconstruction, 'Medianized reconstruction')
    reconstruction = np.array(reconstruction*255.0, dtype=np.uint8)
    print_percentiles(reconstruction, 'reconstruction 8 bit')
    return reconstruction

#@profile
def filter_multiview_volume(Xmean, Xstd, np_tensor_3d_first_view, np_tensor_3d_second_view,
                            extramargin=None, model=None, speedup=None, sparsity_measure_type=None):

    np_tensor_3d_first_view = np_tensor_3d_first_view / 255.0
    np_tensor_3d_second_view = np_tensor_3d_second_view / 255.0

    print('np_tensor_3d_first_view.shape=', np_tensor_3d_first_view.shape)
    print('np_tensor_3d_second_view.shape=', np_tensor_3d_second_view.shape)

    predictor = make_predictor(model)
    sc = np_tensor_3d_first_view.shape
    reconstruction = np.zeros(sc)
    patchlen = (1+2*extramargin)**3
    rangez = range(extramargin, sc[0]-extramargin, speedup)
    rangey = range(extramargin, sc[1]-extramargin, speedup)
    rangex = range(extramargin, sc[2]-extramargin, speedup)

    n_points = len(rangex)*len(rangey)

    normalizer = np.zeros(sc)
    print('Patch size: %dx%dx%d (%d)' % (1+2*extramargin, 1+2*extramargin, 1+2*extramargin, (1+2*extramargin)**3))
    print('Will subsample jumping by',speedup,'voxels')
    pbar_prediction = ProgressBar(widgets=['Pre-processing %d slices (%d patches): ' % (len(rangez),len(rangex)*len(rangey)*len(rangez)),
                                  Percentage(), ' ', AdaptiveETA()])
    bar_reconstruction = ProgressBar(widgets=['Post-processing %d slices (%d patches): ' % (len(rangez),len(rangex)*len(rangey)*len(rangez)),
                                     Percentage(), ' ', AdaptiveETA()])

    ndone = 0
    data = np.zeros((len(rangez) * n_points, patchlen*2))
    iter_z = 0
    for z0 in pbar_prediction(rangez):
        i = 0
        for y0 in rangey:
            for x0 in rangex:
                patch_first_view = np_tensor_3d_first_view[z0-extramargin:z0+extramargin+1,
                                                           y0-extramargin:y0+extramargin+1,
                                                           x0-extramargin:x0+extramargin+1]
                patch_second_view = np_tensor_3d_second_view[z0-extramargin:z0+extramargin+1,
                                                             y0-extramargin:y0+extramargin+1,
                                                             x0-extramargin:x0+extramargin+1]

                data[iter_z * n_points + i][0:patchlen] = np.reshape(patch_first_view, (patchlen,))
                data[iter_z * n_points + i][patchlen:2*patchlen] = np.reshape(patch_second_view, (patchlen,))
                data[iter_z * n_points + i] = (data[iter_z * n_points + i] - Xmean) / Xstd
                i += 1
        iter_z += 1
    print('Predicting ',data.shape[0],' patches ...')
    pred_start = timeit.default_timer()
    data = predictor.predict(data,n_views=2)
    pred_stop = timeit.default_timer()
    print('Prediction done in %s seconds' % (pred_stop - pred_start))
    iter_z = 0
    for z0 in bar_reconstruction(rangez):
        i = 0
        for y0 in rangey:
            for x0 in rangex:
                reconstructed_patch = np.reshape(data[iter_z * n_points + i], (1+2*extramargin,1+2*extramargin,1+2*extramargin))
                # print_percentiles(reconstructed_patch, 'Reconstructed patch at (%d,%d,%d)'%(z0,y0,x0))
                # reconstructed_patch = np.reshape(data[i], (1+2*extramargin,1+2*extramargin,1+2*extramargin))
                reconstruction[z0-extramargin:z0+extramargin+1,
                               y0-extramargin:y0+extramargin+1,
                               x0-extramargin:x0+extramargin+1] += reconstructed_patch
                normalizer[z0-extramargin:z0+extramargin+1,
                           y0-extramargin:y0+extramargin+1,
                           x0-extramargin:x0+extramargin+1] += 1
                i += 1
                ndone += 1
        iter_z += 1
    vprint('Reconstructed and assembled',ndone,'patches, now clipping..')

    reconstruction = reconstruction[
        extramargin:sc[0]-extramargin,
        extramargin:sc[1]-extramargin,
        extramargin:sc[2]-extramargin]
    normalizer = normalizer[
        extramargin:sc[0]-extramargin,
        extramargin:sc[1]-extramargin,
        extramargin:sc[2]-extramargin]

    print_percentiles(reconstruction, 'Reconstruction')
    reconstruction = normalize(reconstruction, normalizer)
    print_percentiles(reconstruction, 'Normalized reconstruction')
    # # smooth aliasing due to speedup
    reconstruction = filters.gaussian_filter(reconstruction,sigma=speedup/3.0, mode='constant', cval=0.0)
    print_percentiles(reconstruction, 'Normalized reconstruction')
    # reconstruction = filters.median_filter(reconstruction,size=3, mode='constant', cval=0.0)
    # print_percentiles(reconstruction, 'Medianized reconstruction')
    reconstruction = np.array(reconstruction*255.0, dtype=np.uint8)
    print_percentiles(reconstruction, 'reconstruction 8 bit')
    return reconstruction
