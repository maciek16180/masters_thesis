import theano.tensor as T
from lasagne import init
from lasagne.layers import Layer,MergeLayer, InputLayer


class SampledSoftmaxDenseLayer(MergeLayer):
    # voc_mask is a mask vector of length voc_size
    
    def __init__(self, incoming, voc_mask, voc_size, targets=None,
                 W_init = init.GlorotUniform(),
                 b_init = init.Constant(0),
                 **kwargs):

        assert len(incoming.output_shape) == 2
        
        incomings = [incoming]
        
        if not isinstance(voc_mask, Layer):
            assert voc_mask.ndim == 1
            voc_mask_shape = (voc_size,)
            voc_mask = InputLayer(voc_mask_shape, input_var=voc_mask, name="voc_mask inputlayer")
            
        incomings.append(voc_mask)
        
        if targets is not None:
            if not isinstance(targets, Layer):
                assert targets.ndim == 1
                targets_shape = (incoming.output_shape[0],)
                targets = InputLayer(targets_shape, input_var=targets, name="targets inputlayer")

            incomings.append(targets)
        
        super(SampledSoftmaxDenseLayer,self).__init__(incomings,**kwargs)
        
        self.voc_size = voc_size
        
        n_inputs = incoming.output_shape[1]
        self.W = self.add_param(W_init, (n_inputs, self.voc_size), name="W")
        self.b = self.add_param(b_init, (self.voc_size,), name="b", regularizable=False)
        
    def get_output_for(self, inputs, **kwargs):
        #
        # if targets are NOT provided: 
        #     returns softmax output calculated only on where voc_mask == 1,
        #     output is surrounded by zeros to match the shape of a full softmax output
        # otherwise:
        #     returns sampled softmax output only for specified class for each sample
        
        assert len(inputs) in [2,3]
        input, voc_mask = inputs[:2]
        
        idx = voc_mask.nonzero()[0]        
        input = input.dot(self.W[:,idx]) + self.b[idx]
        ssoft = T.nnet.softmax(input)
        
        if len(inputs) == 3:
            targets = inputs[2]
            target_idx_in_masked_voc = T.extra_ops.cumsum(voc_mask)[targets] - 1
            return ssoft[T.arange(ssoft.shape[0]), target_idx_in_masked_voc]
        
        else:
            out = T.zeros((input.shape[0], self.voc_size))
            return T.set_subtensor(out[:,idx], ssoft)
        
    def get_output_shape_for(self, input_shapes, **kwargs):
        assert len(input_shapes) in [2,3]
        if len(input_shapes) == 3:
            return (input_shapes[0][0],)
        else:
            return (input_shapes[0][0], self.voc_size)    
    
    
    