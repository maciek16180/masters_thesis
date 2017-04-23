import theano.tensor as T
from lasagne.layers import Layer,MergeLayer, InputLayer


class MaskedReshapeLayer(MergeLayer):
    
    def __init__(self, incoming, input_mask, shape,
                 **kwargs):

        incomings = [incoming]
        
        if not isinstance(input_mask, Layer):
            assert input_mask.ndim <= len(incoming.output_shape)
            input_mask_shape = incoming.output_shape[:input_mask.ndim]
            input_mask = InputLayer(input_mask_shape, input_var=input_mask, name="input_mask inputlayer")
            
        incomings.append(input_mask)
        
        super(MaskedReshapeLayer,self).__init__(incomings,**kwargs)
        
        self.shape = shape
        
    def get_output_for(self, inputs, **kwargs):
        
        assert len(inputs) == 2
        input_, input_mask = inputs
        
        return input_[input_mask.nonzero()].reshape(shape=self.shape)
        
    def get_output_shape_for(self, input_shapes, **kwargs):
        assert len(input_shapes) == 2
        return self.shape
    
    
    
class UnmaskedReshapeLayer(MergeLayer):
    
    def __init__(self, incoming, input_unmask, shape,
                 **kwargs):

        incomings = [incoming]
        
        if not isinstance(input_unmask, Layer):
            assert input_unmask.ndim <= len(shape)
            input_unmask_shape = shape[:input_unmask.ndim]
            input_unmask = InputLayer(input_unmask_shape, input_var=input_unmask, name="input_unmask inputlayer")
            
        incomings.append(input_unmask)
        
        super(UnmaskedReshapeLayer,self).__init__(incomings,**kwargs)
        
        self.shape = shape
        
    def get_output_for(self, inputs, **kwargs):
        
        assert len(inputs) == 2
        input_, input_unmask = inputs
        
        data_shape = (-1,) + self.shape[input_unmask.ndim:]
        
        out = T.zeros(self.shape)
        out = T.set_subtensor(out[input_unmask.nonzero()], input_.reshape(shape=data_shape))
        
        return out
        
    def get_output_shape_for(self, input_shapes, **kwargs):
        assert len(input_shapes) == 2
        return self.shape