import theano.tensor as T
from lasagne import init
from lasagne.layers import Layer,MergeLayer, InputLayer


class SampledSoftmaxDenseLayer(MergeLayer):
    # voc_mask is a mask vector of length voc_size
    
    def __init__(self, incoming, voc_mask, voc_size,
                 W_init = init.GlorotUniform(),
                 b_init = init.Constant(0),
                 **kwargs):

        incomings = [incoming]
        
        if not isinstance(voc_mask, Layer):
            assert voc_mask.ndim == 1
            voc_mask_shape = (voc_size,)
            voc_mask = InputLayer(voc_mask_shape, input_var=voc_mask, name="voc_mask inputlayer")
            
        incomings.append(voc_mask)
        
        super(SampledSoftmaxDenseLayer,self).__init__(incomings,**kwargs)
        
        self.voc_size = voc_size
        
        n_inputs = incoming.output_shape[1]
        self.W = self.add_param(W_init, (n_inputs, self.voc_size),
                                name="W")
        self.b = self.add_param(b_init, (self.voc_size,),
                                name="b",regularizable=False)
        
    def get_output_for(self, inputs, **kwargs):
        # returns softmax output calculated only on voc_mask, 
        # surrounded by zeros to match the shape of a full softmax output
        
        assert len(inputs) == 2
        input, voc_mask = inputs
        
        idx = voc_mask.nonzero()[0]
        
        input = input.dot(self.W[:,idx]) + self.b[idx]
            
        out = T.zeros((input.shape[0], self.voc_size))
        ssoft = T.nnet.softmax(input)
        out = T.set_subtensor(out[:,idx], ssoft)
        
        return out        
        
    def get_output_shape_for(self, input_shapes, **kwargs):
        assert len(input_shapes) == 2
        return (input_shapes[0][0], self.voc_size)