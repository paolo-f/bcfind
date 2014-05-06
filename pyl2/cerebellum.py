from __future__ import print_function
import numpy as N
np = N
from pylearn2.datasets import dense_design_matrix
import tables


class CerebellumFlat(dense_design_matrix.DenseDesignMatrix):
    """This class is used to make a dataset suitable for pylearn2. The
    design matrix contains a numpy array for each example which
    consists of a 3D patch (Z,Y,X) reshaped to 1D. Gray levels are
    rescaled as floats in [0,1]. Since creating the data takes a little while,
    this class reads from an HDF5 file.
    """
    def __init__(self, filename=None, fraction=1.0):
        self.args = locals()
        h5file = tables.openFile(filename, "r")
        n = int(h5file.root.X.shape[0]*fraction)
        X = np.array(h5file.root.X[0:n], dtype=np.float32)
        print('Loaded data matrix of shape', X.shape, 'size:', X.nbytes/(1024*1024), 'MBytes')
        print('Shuffling')
        np.random.shuffle(X)
        print('Shuffling done')
        h5file.close()
        super(CerebellumFlat,self).__init__(X=X)


class CerebellumFlatSupervised(dense_design_matrix.DenseDesignMatrix):
    """Data set with y corresponding to 3D Gaussians centered around GT markers
    """
    def __init__(self, filename=None, fraction=1.0):
        self.args = locals()
        h5file = tables.openFile(filename, "r")
        n = int(h5file.root.X.shape[0]*fraction)
        X = h5file.root.X[0:n]
        print('Loaded data matrix of shape', X.shape, 'size:', X.nbytes/(1024*1024), 'MBytes')
        y = h5file.root.y[0:n]
        print('Loaded target matrix of shape', y.shape, 'size:', y.nbytes/(1024*1024), 'MBytes')
        print('Shuffling')
        perm = np.random.permutation(range(n))
        X = X[perm]
        y = y[perm]
        # good=np.where(y.sum(axis=1)>100)
        # X = X[good]
        # y = y[good]
        # y = np.ceil(y)
        print('Shuffling done')
        print('#Targets=1:', y.sum(), 'out of', y.shape[0]*y.shape[1], 'ratio:', y.mean())
        h5file.close()
        super(CerebellumFlatSupervised,self).__init__(X=X,y=y)


# The rest is probably to be removed
class CerebellumFlatAuto(dense_design_matrix.DenseDesignMatrix):
    """Useful to train an MLP as a deep autoencoder.
    """
    def __init__(self, filename=None, fraction=0.5):
        self.args = locals()
        h5file = tables.openFile(filename, "r")
        n = int(h5file.root.X.shape[0]*fraction)
        X = h5file.root.X[0:n]/255.0
        h5file.close()
        print('Loaded data matrix of shape', X.shape, 'size:', X.nbytes/(1024*1024), 'MBytes')
        super(CerebellumFlatAuto,self).__init__(X=X,y=X)


class CerebellumFlatWithCatForAuto(dense_design_matrix.DenseDesignMatrix):
    """Provides X where category!=0
    """
    def __init__(self, filename=None, fraction=0.5):
        self.args = locals()
        h5file = tables.openFile(filename, "r")
        n = int(h5file.root.X.shape[0]*fraction)
        X = h5file.root.X[0:n]/255.0
        category = h5file.root.category[0:n]
        h5file.close()
        print('Loaded data matrix of shape', X.shape, 'size:', X.nbytes/(1024*1024), 'MBytes')
        print('Loaded category, sum=', np.sum(category), 'mean=', np.mean(category))
        super(CerebellumFlatWithCatForAuto,self).__init__(X=X[category==1])


class CerebellumFlatWithCat(dense_design_matrix.DenseDesignMatrix):
    """Provides X and y==X*category
    """
    def __init__(self, filename=None, fraction=0.5):
        self.args = locals()
        h5file = tables.openFile(filename, "r")
        n = int(h5file.root.X.shape[0]*fraction)
        X = h5file.root.X[0:n]/255.0
        category = h5file.root.category[0:n]
        h5file.close()
        print('Loaded data matrix of shape', X.shape, 'size:', X.nbytes/(1024*1024), 'MBytes')
        print('Loaded category, sum=', np.sum(category), 'mean=', np.mean(category))
        super(CerebellumFlatWithCat,self).__init__(X=X,y=X*np.tile(category,(X.shape[1],1)).T)
