__authors__ = "Marco Paciscopi"

from pylearn2.datasets.dataset import Dataset
from pylearn2.utils.data_specs import is_flat_specs
from theano import config
from pylearn2.utils import wraps
from pylearn2.space import CompositeSpace, VectorSpace
import numpy as np

class TransformerMultipleSourceDataset(Dataset):
    """
        A dataset that applies a transformation of a raw dataset for each view
        by its corresponding transformer (transformer_left and transformer_right)
    """
    def __init__(self, raw, transformer_left,transformer_right, cpu_only = False,
            space_preserving=False):
        """
            raw: a pylearn2 Dataset that provides raw data
            transformer_left: a pylearn2 Block to transform the data
            transformer_right: a pylearn2 Block to transform the data
        """
        self.__dict__.update(locals())
        del self.self

    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, targets=None, rng=None, data_specs=None,
                 return_tuple=False):



        # Build the right data_specs to query self.raw
        if data_specs is not None:
            assert is_flat_specs(data_specs)
            space, source = data_specs
            if not isinstance(source, tuple):
                source = (source,)
            if isinstance(space, CompositeSpace):
                space = tuple(space.components)
            else:
                space = (space,)


        raw_data_specs = self.raw.get_data_specs()

        print 'data_specs'
        print data_specs
        print 'raw_data_specs'
        print raw_data_specs

        raw_iterator = self.raw.iterator(mode=mode, batch_size=batch_size,
                num_batches=num_batches, topo=None, targets=targets, rng=rng,
                data_specs=raw_data_specs, return_tuple=return_tuple)

        final_iterator = TransformerIterator(raw_iterator, self,
                                             data_specs=data_specs)

        return final_iterator

    def has_targets(self):
        return self.raw.has_targets()

    def adjust_for_viewer(self, X):
        if self.space_preserving:
            return self.raw.adjust_for_viewer(X)
        return X

    def get_weights_view(self, *args, **kwargs):
        if self.space_preserving:
            return self.raw.get_weights_view(*args, **kwargs)
        raise NotImplementedError()

    def get_topological_view(self, *args, **kwargs):
        if self.space_preserving:
            return self.raw.get_weights_view(*args, **kwargs)
        raise NotImplementedError()

    def adjust_to_be_viewed_with(self, *args, **kwargs):
        return self.raw.adjust_to_be_viewed_with(*args, **kwargs)

    @wraps(Dataset.get_num_examples)
    def get_num_examples(self):
        return self.raw.get_num_examples()


class TransformerIterator(object):

    def __init__(self, raw_iterator, transformer_dataset, data_specs):
        self.raw_iterator = raw_iterator
        self.transformer_dataset = transformer_dataset
        self.stochastic = raw_iterator.stochastic
        self.uneven = raw_iterator.uneven
        self.data_specs = data_specs

    def __iter__(self):
        return self

    def next(self):
        raw_batch = self.raw_iterator.next()

        #print raw_batch[0].shape
        # Apply transformation on raw_batch, and format it
        # in the requested Space
        transformer_left = self.transformer_dataset.transformer_left
        transformer_right = self.transformer_dataset.transformer_right


        def transform(raw_batch):

            rval_left = transformer_left.perform(raw_batch[0][:,0:raw_batch[0].shape[1]])
            rval_right = transformer_right.perform(raw_batch[1][:,0:raw_batch[1].shape[1]])
            rval = np.hstack((rval_left, rval_right))
            return rval


        rval = transform(raw_batch)

        #rval = (rval,) + raw_batch[2:]
        rval = (rval,)

        #if not isinstance(raw_batch, tuple):
            ## Only one source, return_tuple is False
            #rval = transform(raw_batch)
        #else:
            ## Apply the transformer only on the first element
            #rval = (transform(raw_batch[0]),) + raw_batch[1:]

        return rval

    @property
    def num_examples(self):
        return self.raw_iterator.num_examples
