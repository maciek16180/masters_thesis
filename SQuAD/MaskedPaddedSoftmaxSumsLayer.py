import theano.tensor as T
import numpy as np
from lasagne import init
from lasagne.layers import Layer, MergeLayer, InputLayer

from lasagne.random import get_rng
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


class MaskedPaddedSoftmaxSumsLayer(MergeLayer):
    
    def __init__(self, incomings, **kwargs):
        
        assert len(incomings) == 2
        assert len(incomings[0].output_shape) == 3
        assert len(incomings[1].output_shape) == 3
        
        super(MaskedPaddedSoftmaxSumsLayer, self).__init__(incomings, **kwargs)
        
    def get_output_for(self, inputs, **kwargs):
        
        assert len(inputs) == 2
        
        input_ = inputs[0]
        mask = inputs[1]
        
        input_ = T.exp(input_) * mask
        sums = input_.sum(axis=2)
        input_ /= sums.dimshuffle(0, 1, 'x')
        
        return T.concatenate([T.ones(input_.shape[:2]), input_.sum(axis=1)], axis=1)

        
    def get_output_shape_for(self, input_shapes, **kwargs):
        assert len(input_shapes) == 2
        return input_shapes[0][0], input_shapes[0][2]
    
    
    