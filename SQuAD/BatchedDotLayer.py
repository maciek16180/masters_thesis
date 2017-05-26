import theano.tensor as T
from lasagne.layers import MergeLayer


class BatchedDotLayer(MergeLayer):
    
    def __init__(self, incomings, **kwargs):
        
        assert len(incomings) == 2        
        super(BatchedDotLayer, self).__init__(incomings, **kwargs)
        
    def get_output_for(self, inputs, **kwargs):
        
        assert len(inputs) == 2
        input_, weights = inputs        
        return (input_ * weights.dimshuffle(0, 1, 'x')).sum(axis=1)
        
    def get_output_shape_for(self, input_shapes, **kwargs):
        assert len(input_shapes) == 2
        return input_shapes[0][0], input_shapes[0][2]
    
    
    