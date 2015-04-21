from pylearn2.datasets import vector_spaces_dataset, dataset
import numpy as np
import pandas as pd
from theano import config
from theano.compat.python2x import OrderedDict
import theano.tensor as T

from pylearn2.space import CompositeSpace, VectorSpace
from pylearn2.datasets.dataset import Dataset
from pylearn2.utils import wraps
from pylearn2.models.mlp import Layer,CompositeLayer
import tables
import functools

from pylearn2.utils.iteration import (
    FiniteDatasetIterator,
    resolve_iterator_class,
    SubsetIterator,
)





class MultiViewDataset(vector_spaces_dataset.VectorSpacesDataset):
    """This class is used to make a dataset suitable for pylearn2. The data is
    represented as VectorSpacesDataset, i.e  a tuple that contains the input
    examples of both left and right views (X_left, X_right) and their labels (y)
    """
    def __init__(self, filename, fraction=1.0, do_dropout=False,p_dropout=0.5):

        self.args = locals()

        print("Reading hdf5 file...")
        h5file = tables.openFile(filename, "r")
        n = int(h5file.root.X.shape[0])
        X = h5file.root.X[0:n]
        y = h5file.root.y[0:n]
        print('Shuffling')
        perm = np.random.permutation(range(n))
        X = X[perm]
        y = y[perm]
        print('Shuffling done')
        X = X[0:n * fraction]


        print('Loaded data matrix of shape', X.shape, 'size:', X.nbytes / (1024 * 1024), 'MBytes')
        y = y[0:n * fraction]
        print('Loaded target matrix of shape', y.shape, 'size:', y.nbytes / (1024 * 1024), 'MBytes')



        X_left = X[:,0:X.shape[1]/2]
        X_right = X[:,X.shape[1]/2:X.shape[1]]

        print(X_left.shape,X_right.shape)

        Xl_space = VectorSpace(X.shape[1]/2, dtype=config.floatX)
        Xr_space = VectorSpace(X.shape[1]/2, dtype=config.floatX)
        y_space = VectorSpace(y.shape[1], dtype=config.floatX)
        space = CompositeSpace((Xl_space, Xr_space, y_space,))
        source = ("left_features", "right_features", "targets",)
        self.data_specs = (space, source)
        self.data = (X_left, X_right, y)

        super(MultiViewDataset,self).__init__(data=self.data,
                                              data_specs=self.data_specs)

    def get_data_specs(self):
        return self.data_specs

    def get_num_examples(self):
        return self.data[0].shape[0]

    @functools.wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, return_tuple=False, targets=None, rng=None, data_specs=None):


        mode = resolve_iterator_class(mode)
        self.data_specs = data_specs
        if self.do_dropoout:
            return MultiViewDatasetIteratorDropout(
                    self,
                    mode(self.data[0].shape[0],
                        batch_size, num_batches, rng),self.p_dropout,0.5,0.5,
                    data_specs=data_specs
                    )
        else:
            return FiniteDatasetIterator(
                    self,
                    mode(self.data[0].shape[0],
                        batch_size, num_batches, rng),
                    data_specs=data_specs
                    )



class MultiViewDatasetIteratorDropout(FiniteDatasetIterator):

    """This class iterates over the dataset applying a dropout  mask on the left
    or right part of each example with probability dropout_p
    """

    def __init__(self, dataset, subset_iterator, dropout_p, dropout_p_left, dropout_p_right, data_specs=None,
		 return_tuple=False, convert=None):

        FiniteDatasetIterator.__init__(self, dataset, subset_iterator,data_specs,return_tuple,convert)
        self.dropout_p=dropout_p
        self.dropout_p_left=dropout_p_left
        self.dropout_p_right=dropout_p_right



    @wraps(SubsetIterator.next)
    def next(self):
        next_index = self._subset_iterator.next()
        # If the dataset is incompatible with the new interface, fall back to
        # the old one
        if hasattr(self._dataset, 'get'):
            rval = self._next(next_index)
        else:
            rval = self._fallback_next(next_index)

        dropout_mask=np.random.binomial(1,1-self.dropout_p,rval[0].shape[0])
        ind=np.where(dropout_mask==0)[0]
        left_mask=dropout_mask.copy()
        left_mask[ind]=np.random.binomial(1,1-self.dropout_p_left,len(ind))
        right_mask=np.logical_not(np.logical_xor(dropout_mask,left_mask)).astype(int)
        rval=((rval[0]*left_mask[:,np.newaxis]).astype(np.float32),(rval[1]*right_mask[:,np.newaxis]).astype(np.float32),rval[2])

        if not self._return_tuple and len(rval) == 1:
            rval, = rval
        return rval


