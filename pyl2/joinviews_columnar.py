from pylearn2.models.mlp import MLP, CompositeLayer, FlattenerLayer, Layer
from pylearn2.space import CompositeSpace,VectorSpace
from theano.compat.python2x import OrderedDict
from pylearn2.utils import wraps
import theano
import theano.tensor as T

class ColumnarMLP(MLP):
    def __init__(self, **kwargs):

        self._input_source = kwargs.pop('input_source', 'features')
        self._target_source = kwargs.pop('target_source','targets')

        print('input_source: ',self._input_source)
        print('target_source: ',self._target_source)

        #super(ColumnarMLP, self).__init__(*args, **kwargs)
        super(ColumnarMLP, self).__init__(input_source=self._input_source, target_source=self._target_source)

    def get_input_source(self):
        return self._input_source

    def get_target_source(self):
        return self._target_source



class CustomCompositeLayer(CompositeLayer):

    def set_input_space(self,space):

        self.input_space = space
        for layer, component_space in  zip(self.layers,space.components):
            layer.set_input_space(component_space)
        self.output_space = CompositeSpace(tuple(layer.get_output_space() for  layer in self.layers))


    def get_input_source(self):
        return ([layer.get_input_source() for layer in self.layers])

    def get_target_source(self):
        return ([layer.get_target_source() for layer in self.layers])

    def fprop(self, state_below):
        return tuple(layer.fprop(component_state) for layer, component_state in zip(self.layers,state_below))

    @wraps(Layer.get_monitoring_channels)
    def get_monitoring_channels(self):
        return OrderedDict()

class CustomFlattenerLayer(FlattenerLayer):

    @wraps(Layer.get_monitoring_channels)
    def get_monitoring_channels(self):
        return self.raw_layer.get_monitoring_channels()



