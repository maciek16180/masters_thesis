import theano.tensor as T
from lasagne.layers import MergeLayer


class MultNormKLDivLayer(MergeLayer):
    
    def __init__(self, incomings, **kwargs):

        assert len(incomings) == 4
        for i in xrange(4):
            assert len(incomings[i].output_shape) == 2
        
        super(MultNormKLDivLayer, self).__init__(incomings, **kwargs)
        
    def get_output_for(self, inputs, **kwargs):
        
        assert len(inputs) == 4
        
        mu0, sig0, mu1, sig1 = inputs
        
        res = (sig0 / sig1).sum(axis=1) + ((mu1 - mu0) / sig1 * (mu1 - mu0)).sum(axis=1) - 
              mu0.shape[0] + (sig1.prod(axis=1) / sig0.prod(axis=1)).log()
        return .5 * res
        
    def get_output_shape_for(self, input_shapes, **kwargs):
        return (input_shapes[0][0],)
