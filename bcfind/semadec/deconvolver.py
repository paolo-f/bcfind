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
import os
import warnings
from bcfind import timer


deconvolver_timer = timer.Timer('Semantic Deconvolution analysis')


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

        if isinstance(model.get_input_space(), VectorSpace):
            self.is_vector_space=True
        elif isinstance(model.get_input_space(), CompositeSpace):
            self.is_vector_space=False
        else:
            raise ValueError("Can't support input space " + str(model.get_input_space().__class__))

        self.model = model
        self.X_sv = model.get_input_space().make_batch_theano()
        self.Xhat_sv = model.fprop(self.X_sv)

        if isinstance(self.X_sv, tuple):
            self.reconstruction_function = theano.function(self.X_sv,[self.Xhat_sv], allow_input_downcast=True)
        else:
            self.reconstruction_function = theano.function([self.X_sv],[self.Xhat_sv], allow_input_downcast=True)

    def predict(self,X,batch_size=None, n_views=1):
        if batch_size is None:
            batch_size = X.shape[0]
        n_batches = X.shape[0]/batch_size
        Xhat = np.zeros((X.shape[0],X.shape[1]/n_views), dtype=np.float32)
        for i in xrange(n_batches):

            if self.is_vector_space:
                xx = self.reconstruction_function(X[i*batch_size:(i+1)*batch_size,:])
            else:
                xx = self.reconstruction_function(*tuple(X[:,j*(X.shape[1]/n_views):(j+1)*(X.shape[1]/n_views)] for j in xrange(n_views)))

            Xhat[i*batch_size:(i+1)*batch_size] = xx[0]
        return Xhat

def make_predictor(model):
    if isinstance(model, DeepComposedAutoencoder) or isinstance(model, Autoencoder):
        return PredictorFromPylearn2Autoencoder(model)
    elif isinstance(model, MLP):
        return PredictorFromPylearn2MLP(model)
    else:
        raise TypeError('Model has type', type(model))



def compute_data_mean_std(list_tensors,extramargin,speedup,do_cython,rangex,rangey,rangez,patchlen,shape_tensor,trainfile):


    print('Patch size: %dx%dx%d (%d)' % (1+2*extramargin, 1+2*extramargin, 1+2*extramargin, (1+2*extramargin)**3))
    print('Will subsample jumping by',speedup,'voxels')

    data=np.empty(shape=(len(rangex)*len(rangey)*len(rangez),patchlen*len(list_tensors)),dtype=np.float32)
    num_examples_slice=len(rangex)*len(rangey)
    if do_cython:

        pbar = ProgressBar(widgets=['PreProcessing %d slices (%d patches): ' % (len(rangez),len(rangex)*len(rangey)*len(rangez)),
                                    Percentage(), ' ', ETA()])
        j=0
        for z0 in pbar(rangez):

            for i in xrange(len(list_tensors)):
                data_tensor= extract_patch.preprocess_2d(list_tensors[i], z0, extramargin, extramargin, speedup)
                if i==0:
                    data_slice=data_tensor
                else:
                    data_slice=np.hstack((data_slice,data_tensor))

            data[j*num_examples_slice:(j+1)*num_examples_slice]=data_slice
            j+=1
    else:

        pbar = ProgressBar(widgets=['PreProcessing %d slices (%d patches): ' % (len(rangez),len(rangex)*len(rangey)*len(rangez)),
                                    Percentage(), ' ', ETA()])

        i = 0
        for z0 in pbar(rangez):
            for y0 in rangey:
                for x0 in rangex:
                    for j in xrange(len(list_tensors)):
                        patch = list_tensors[j][z0-extramargin:z0+extramargin+1,
                                            y0-extramargin:y0+extramargin+1,
                                            x0-extramargin:x0+extramargin+1]

                        data_tensor = np.reshape(patch, (patchlen,))
                        if j==0:
                            data_slice=data_tensor
                        else:
                            data_slice=np.hstack((data_slice,data_tensor))
                    data[i]=data_slice
                    i += 1

    good_indices=np.mean(data[:,]*255,axis=1)>5
    min_examples=data.shape[0]/patchlen
    if len(np.where(good_indices==True)[0])>=min_examples:
	Xmean=data[good_indices].mean(axis=0).astype(np.float32)
	Xstd=data[good_indices].std(axis=0).astype(np.float32)
    else:
	h5 = tables.openFile(trainfile)
	Xmean = h5.root.Xmean[:].astype(np.float32)
	Xstd = h5.root.Xstd[:].astype(np.float32)
	h5.close()

    return data,Xmean,Xstd


def deconvolve(list_tensors,extramargin,speedup,do_cython,rangex,rangey,rangez,patchlen,shape_tensor, data, Xmean,Xstd,model):


    reconstruction = np.zeros(shape_tensor, dtype=np.float32)
    normalizer = np.zeros(shape_tensor, dtype=np.float32)
    predictor = make_predictor(model)
    num_tensors=len(list_tensors)
    n_points=len(rangex)*len(rangey)
    if do_cython:

        pbar = ProgressBar(widgets=['Processing %d slices (%d patches): ' % (len(rangez),len(rangex)*len(rangey)*len(rangez)),
                                    Percentage(), ' ', ETA()])
        j=0
        for z0 in pbar(rangez):

            if data is None:
                for i in xrange(len(list_tensors)):
                    data_tensor= extract_patch.preprocess_2d_mean_std(list_tensors[i], z0, extramargin, extramargin, speedup, Xmean, Xstd)
                    if i==0:
                        data_slice=data_tensor
                    else:
                        data_slice=np.hstack((data_slice,data_tensor))
            else:
                data_slice=(data[j*n_points:(j+1)*n_points]-Xmean)/Xstd

            pred_data = predictor.predict(data_slice,n_views=num_tensors)
            reconstruction,normalizer = extract_patch.postprocess_2d(list_tensors[0], z0, pred_data, reconstruction, normalizer, extramargin, extramargin, speedup)
            j+=1
    else:

        pbar = ProgressBar(widgets=['Processing %d slices (%d patches): ' % (len(rangez),len(rangex)*len(rangey)*len(rangez)),
                                    Percentage(), ' ', ETA()])
        j=0
        for z0 in pbar(rangez):
            data_slice = np.zeros((n_points, patchlen),  dtype=np.float32)
            if data is None:
                i = 0
                for y0 in rangey:
                    for x0 in rangex:
                        for j in xrange(len(list_tensors)):
                            patch = list_tensors[j][z0-extramargin:z0+extramargin+1,
                                                y0-extramargin:y0+extramargin+1,
                                                x0-extramargin:x0+extramargin+1]

                            data_patch_flat_view = np.reshape(patch, (patchlen,))
                            if j==0:
                                data_patch_flat=data_patch_flat_view
                            else:
                                data_patch_flat=np.hstack((data_patch_flat,data_patch_flat_view))
                        data_slice[i] = (data_patch_flat- Xmean)/Xstd
                        i += 1
            else:
                data_slice=(data[j*n_points:(j+1)*n_points]-Xmean)/Xstd

            pred_data = predictor.predict(data_slice, n_views=num_tensors)

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

            j+=1

    vprint('Reconstructed and assembled',n_points*len(rangez),'patches, now clipping..')
    reconstruction = reconstruction[
        extramargin:shape_tensor[0]-extramargin,
        extramargin:shape_tensor[1]-extramargin,
        extramargin:shape_tensor[2]-extramargin]
    normalizer = normalizer[
        extramargin:shape_tensor[0]-extramargin,
        extramargin:shape_tensor[1]-extramargin,
        extramargin:shape_tensor[2]-extramargin]
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


def compute_ranges(list_tensors,extramargin,speedup):

    if len(list_tensors)==0:
        raise Exception('No tensores have been specified')

    shape_list=[]
    for tensor in list_tensors:
        shape_list.append(tensor.shape)

    if not all(x == shape_list[0] for x in shape_list):
            raise AssertionError()

    sc = shape_list[0]

    patchlen = (1+2*extramargin)**3
    rangez = range(extramargin, sc[0]-extramargin, speedup)
    rangey = range(extramargin, sc[1]-extramargin, speedup)
    rangex = range(extramargin, sc[2]-extramargin, speedup)

    return rangex,rangey,rangez,patchlen,sc


@deconvolver_timer.timed
def filter_volume(list_tensors, Xmean, Xstd, extramargin, model,
                  speedup, do_cython=False,trainfile=None):
    """
    Method that applies the semantic deconvolution to a list of tensors of the same shape in the 
    multiview scenario or to a single tensor in a single view scenario. Firstly, the function 
    preprocess the tensors obtaining a standardized data array of shape (num_examples,patchlen*len(list_tensors)),
    then deconvolves it with a trained neural network and finally reconstructs the deconvolved output tensor

    Parameters
    ----------
    
    list_tensors : [list| numpy tensor]
        list of tensors of the same shape (nz,ny,nx) in the multiview scenario or 
        a single numpy tensor in the single view scenario
    Xmean : str
	Mean vector of length patchlen*len(list_tensors) used for normalizing data
    Xstd : str
	Standard Deviation vector of length patchlen*len(list_tensors) used for normalizing data
    extramargin : int
        Extra margin for convolution. Should be equal to (filter_size - 1)/2	
    model : str 
	pickle file containing a trained network
    speedup : boolean 
	convolution stride (isotropic along X,Y,Z)
    do_cython : boolean 
	use the compiled cython module to speedup preprocessing and postprocessing (Default: False)
    trainfile : boolean 
	HDF5 file on which the network was trained (should contain mean/std arrays) (Default: None)

    Returns
    -------

    renconstruction: numpy tensor
	deconvolved tensor of the same shape of the input tensors

    """
    if isinstance(list_tensors, (np.ndarray, np.generic)):
        list_tensors=[list_tensors]

    for i in xrange(len(list_tensors)):
        list_tensors[i]=(list_tensors[i]/255.).astype(np.float32)

    rangex,rangey,rangez,patchlen,shape_tensor=compute_ranges(list_tensors,extramargin,speedup)

    data=None
    if Xstd is None or Xmean is None:
        data,Xmean,Xstd=compute_data_mean_std(list_tensors,extramargin,speedup,do_cython,rangex,rangey,rangez,patchlen,shape_tensor,trainfile)

    reconstruction=deconvolve(list_tensors,extramargin,speedup,do_cython,rangex,rangey,rangez,patchlen,shape_tensor,data, Xmean,Xstd,model)

    return reconstruction



