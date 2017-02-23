import theano.tensor as T
from pylearn2.models.mlp import Layer
from pylearn2.utils import wraps
from pylearn2.compat import OrderedDict


class PretrainedLayer(Layer):

    def __init__(self, layer_name, layer_content, layer_params=None):
        super(PretrainedLayer, self).__init__()
        self.__dict__.update(locals())
        del self.self
        if layer_params is not None:
            self.layer_content.set_param_values(layer_params)
        del self.layer_params

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        assert self.get_input_space() == space

    @wraps(Layer.get_params)
    def get_params(self):
        return self.layer_content.get_params()

    @wraps(Layer.get_input_space)
    def get_input_space(self):
        return self.layer_content.get_input_space()

    @wraps(Layer.get_output_space)
    def get_output_space(self):
        return self.layer_content.get_output_space()

    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                      state=None, targets=None):
        return OrderedDict([])

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        return self.layer_content.upward_pass(state_below)

    @wraps(Layer.get_weight_decay)
    def get_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        if hasattr(self.layer_content, 'W'):
            return coeff * T.sqr(self.layer_content.W).sum()
        if hasattr(self.layer_content, 'weights'):
            return coeff * T.sqr(self.layer_content.weights).sum()

    @wraps(Layer.get_l1_weight_decay)
    def get_l1_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        if hasattr(self.layer_content, 'W'):
            return coeff * abs(self.layer_content.W).sum()
        if hasattr(self.layer_content, 'weights'):
            return coeff * abs(self.layer_content.weights).sum()