import theano.tensor as T
from lasagne.layers import MergeLayer


class MaskedSoftmaxLayer(MergeLayer):
    
    def __init__(self, incomings, **kwargs):
        
        assert len(incomings) == 2
        assert len(incomings[0].output_shape) == 2
        assert len(incomings[1].output_shape) == 2
        
        super(MaskedSoftmaxLayer, self).__init__(incomings, **kwargs)
        
    def get_output_for(self, inputs, **kwargs):
        
        assert len(inputs) == 2        
        input_, mask = inputs
        
        input_ = T.exp(input_) * mask
        sums = input_.sum(axis=1).dimshuffle(0, 'x')        
        return input_ / sums
        
    def get_output_shape_for(self, input_shapes, **kwargs):
        assert len(input_shapes) == 2
        return input_shapes[0]
