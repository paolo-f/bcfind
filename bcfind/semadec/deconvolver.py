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


    def predict(self,X,model,batch_size=None, n_views=1):
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


def filter_volume(np_tensor_3d, Xmean, Xstd,
                  extramargin=None, model=None, speedup=None, do_cython=False):


    predictor = make_predictor(model)
    sc = np_tensor_3d.shape
    reconstruction = np.zeros(sc, dtype=np.float32)
    reconstruction_2 = np.zeros(sc, dtype=np.float32)
    n_points = (sc[1]-2*extramargin)*(sc[2]-2*extramargin)
    patchlen = (1+2*extramargin)**3
    rangez = range(extramargin, sc[0]-extramargin, speedup)
    rangey = range(extramargin, sc[1]-extramargin, speedup)
    rangex = range(extramargin, sc[2]-extramargin, speedup)
    n_points = len(rangex)*len(rangey)

    normalizer = np.zeros(sc, dtype=np.float32)
    normalizer_2 = np.zeros(sc, dtype=np.float32)
    print('Patch size: %dx%dx%d (%d)' % (1+2*extramargin, 1+2*extramargin, 1+2*extramargin, (1+2*extramargin)**3))
    print('Will subsample jumping by',speedup,'voxels')
    pbar = ProgressBar(widgets=['Processing %d slices (%d patches): ' % (len(rangez),len(rangex)*len(rangey)*len(rangez)),
                                Percentage(), ' ', ETA()])

    np_tensor_3d=(np_tensor_3d/255.).astype(np.float32)

    if do_cython:

        for z0 in pbar(rangez):

            data= extract_patch.preprocess_2d(np_tensor_3d, z0, extramargin, extramargin, speedup, Xmean, Xstd)
            pred_data = predictor.predict(data, model)
            reconstruction,normalizer = extract_patch.postprocess_2d(np_tensor_3d, z0, pred_data, reconstruction, normalizer, extramargin, extramargin, speedup)
    else:

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

            pred_data = predictor.predict(data, model)

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



def filter_multiview_volume(np_tensor_3d_first_view, np_tensor_3d_second_view, size_patch, trainfile, local_mean_std=False,
                            extramargin=None, model=None, speedup=None):

    np_tensor_3d_first_view = np_tensor_3d_first_view / 255.0
    np_tensor_3d_second_view = np_tensor_3d_second_view / 255.0

    print('np_tensor_3d_first_view.shape=', np_tensor_3d_first_view.shape)
    print('np_tensor_3d_second_view.shape=', np_tensor_3d_second_view.shape)


    predictor = make_predictor(model)
    sc = np_tensor_3d_first_view.shape
    reconstruction = np.zeros(sc)
    patchlen = (1+2*size_patch)**3
    rangez = range(extramargin, sc[0]-extramargin, speedup)
    rangey = range(extramargin, sc[1]-extramargin, speedup)
    rangex = range(extramargin, sc[2]-extramargin, speedup)

    n_points = len(rangex)*len(rangey)

    normalizer = np.zeros(sc)
    print('Patch size: %dx%dx%d (%d)' % (1+2*size_patch, 1+2*size_patch, 1+2*size_patch, (1+2*size_patch)**3))
    print('Will subsample jumping by',speedup,'voxels')
    bar_prediction = ProgressBar(widgets=['Pre-processing %d slices (%d patches): ' % (len(rangez),len(rangex)*len(rangey)*len(rangez)),
                                  Percentage(), ' ', AdaptiveETA()])
    bar_reconstruction = ProgressBar(widgets=['Post-processing %d slices (%d patches): ' % (len(rangez),len(rangex)*len(rangey)*len(rangez)),
                                     Percentage(), ' ', AdaptiveETA()])

    ndone = 0
    data = np.zeros((len(rangez) * n_points, patchlen*2))
    iter_z = 0
    for z0 in bar_prediction(rangez):
        i = 0
        for y0 in rangey:
            for x0 in rangex:
                patch_first_view = np_tensor_3d_first_view[z0-size_patch:z0+size_patch+1,
                                                           y0-size_patch:y0+size_patch+1,
                                                           x0-size_patch:x0+size_patch+1]
                patch_second_view = np_tensor_3d_second_view[z0-size_patch:z0+size_patch+1,
                                                             y0-size_patch:y0+size_patch+1,
                                                             x0-size_patch:x0+size_patch+1]

                data[iter_z * n_points + i][0:patchlen] = np.reshape(patch_first_view, (patchlen,))
                data[iter_z * n_points + i][patchlen:2*patchlen] = np.reshape(patch_second_view, (patchlen,))
                #data[iter_z * n_points + i] = (data[iter_z * n_points + i] - Xmean) / Xstd
                i += 1
        iter_z += 1


    if local_mean_std:
        print('Reading standardization data from substack')
        #Xmean = data.mean(axis=0)
        #Xstd = data.std(axis=0)
        Xmean=data[np.mean(data[:,]*255,axis=1)>5].mean(axis=0)#5
        Xstd= data[np.mean(data[:,]*255,axis=1)>5].std(axis=0)#5

    else:
        print('Reading standardization data from trainfile', trainfile)
        h5 = tables.openFile(trainfile)
        Xmean = h5.root.Xmean[:]
        Xstd = h5.root.Xstd[:]
        h5.close()

    std_start = timeit.default_timer()
    data = (data - Xmean) / Xstd
    std_stop = timeit.default_timer()
    print('Standardization done in %s seconds' % (std_stop - std_start))
    pred_start = timeit.default_timer()
    data = predictor.predict(data, model, n_views=2)
    pred_stop = timeit.default_timer()
    print('Prediction done in %s seconds' % (pred_stop - pred_start))
    iter_z = 0
    for z0 in bar_reconstruction(rangez):
        i = 0
        for y0 in rangey:
            for x0 in rangex:
                reconstructed_patch = np.reshape(data[iter_z * n_points + i], (1+2*size_patch,1+2*size_patch,1+2*size_patch))
                reconstruction[z0-size_patch:z0+size_patch+1,
                               y0-size_patch:y0+size_patch+1,
                               x0-size_patch:x0+size_patch+1] += reconstructed_patch
                normalizer[z0-size_patch:z0+size_patch+1,
                           y0-size_patch:y0+size_patch+1,
                           x0-size_patch:x0+size_patch+1] += 1
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
    if speedup>1:
        reconstruction = filters.gaussian_filter(reconstruction,sigma=speedup/3.0, mode='constant', cval=0.0)
    print_percentiles(reconstruction, 'Normalized reconstruction')
    # reconstruction = filters.median_filter(reconstruction,size=3, mode='constant', cval=0.0)
    # print_percentiles(reconstruction, 'Medianized reconstruction')


    reconstruction = np.array(reconstruction*255.0, dtype=np.uint8)
    print_percentiles(reconstruction, 'reconstruction 8 bit')
    return reconstruction
