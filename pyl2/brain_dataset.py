from __future__ import print_function
import numpy as N
np = N
from pylearn2.datasets import dense_design_matrix,hdf5
from pylearn2.space import CompositeSpace,VectorSpace
from theano import config
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

        n = int(h5file.root.X.shape[0])
        X = np.array(h5file.root.X[0:n], dtype=np.float32)
        ##for shuffling uncomment
        print('Shuffling')
        np.random.shuffle(X)
        print('Shuffling done')
        X = X[0:n * fraction]
        print('Loaded data matrix of shape', X.shape, 'size:', X.nbytes / (1024 * 1024), 'MBytes')
        h5file.close()

        X_space = VectorSpace(X.shape[1], dtype=config.floatX)
        space = CompositeSpace((X_space,))
        source = ("features",)
        self.data_specs = (space, source)

        super(CerebellumFlat, self).__init__(X=X)

    def get_data_specs(self):
        return self.data_specs

class CerebellumFlatH5(hdf5.HDF5Dataset):
    """This class is used to make a dataset suitable for pylearn2. The
    design matrix contains a numpy array for each example which
    consists of a 3D patch (Z,Y,X) reshaped to 1D. Gray levels are
    rescaled as floats in [0,1]. Since creating the data takes a little while,
    this class reads from an HDF5 file.
    """

    def __init__(self, filename=None, fraction=1.0):
        self.args = locals()
        h5file = tables.openFile(filename, "r")

        n = int(h5file.root.X.shape[0])
        X = np.array(h5file.root.X[0:n], dtype=np.float32)
        ##for shuffling uncomment
        print('Shuffling')
        np.random.shuffle(X)
        print('Shuffling done')
        X = X[0:n * fraction]
        print('Loaded data matrix of shape', X.shape, 'size:', X.nbytes / (1024 * 1024), 'MBytes')
        h5file.close()

        X_space = VectorSpace(X.shape[1], dtype=config.floatX)
        space = CompositeSpace((X_space,))
        source = ("features",)
        self.data_specs = (space, source)

        super(CerebellumFlatH5, self).__init__(X='X',filename=filename,load_all=True,cache_size=5832000)

    def get_data_specs(self):
        return self.data_specs

class CerebellumFlatSupervised(dense_design_matrix.DenseDesignMatrix):
    """Data set with y corresponding to 3D Gaussians centered around GT markers
    """

    def __init__(self, filename=None, fraction=1.0):
        self.args = locals()
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


        print('#Targets=1:', y.sum(), 'out of', y.shape[0] * y.shape[1], 'ratio:', y.mean())
        h5file.close()
        super(CerebellumFlatSupervised, self).__init__(X=X, y=y)

class CerebellumFlatSupervisedH5(hdf5.HDF5Dataset):
    """Data set with y corresponding to 3D Gaussians centered around GT markers
    """

    def __init__(self, filename=None, fraction=1.0):
        self.args = locals()
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


        print('#Targets=1:', y.sum(), 'out of', y.shape[0] * y.shape[1], 'ratio:', y.mean())
        h5file.close()
        super(CerebellumFlatSupervisedH5, self).__init__(X='X',y='y',filename=filename,load_all=False)

