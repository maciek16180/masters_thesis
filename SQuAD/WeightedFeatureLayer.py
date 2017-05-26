# as in https://arxiv.org/abs/1703.04816

import theano.tensor as T
from lasagne.layers import MergeLayer
from lasagne import init


class WeightedFeatureLayer(MergeLayer):
    
    def __init__(self, incomings, V=init.Uniform(), **kwargs):
        
        assert len(incomings) == 3
        assert len(incomings[0].output_shape) == 3
        assert len(incomings[1].output_shape) == 3
        assert len(incomings[2].output_shape) == 2
        
        super(WeightedFeatureLayer, self).__init__(incomings, **kwargs)
        
        emb_size = incomings[0].output_shape[2]
        self.V = self.add_param(V, (emb_size,), name="V")
        
    def get_output_for(self, inputs, **kwargs):
        
        assert len(inputs) == 3
        
        context, question, mask = inputs
        qs = question.shape
        
        question = question.reshape((-1, qs[-1])) * self.V
        question = question.reshape(qs) # batch_size x question_len x emb_size
        
        context = context.dimshuffle(0, 2, 1) # batch_size x emb_size x (max_context_len-1)*con_seq_len
        
        esim = T.exp(T.batched_dot(question, context)) # batch_size x question_len x (max_context_len-1)*con_seq_len
        esim *= mask.reshape((qs[0], 1, -1))
        
        sums = esim.sum(axis=2)
        esim /= sums.dimshuffle(0, 1, 'x')
        
        return esim.sum(axis=1) # batch_size x (max_context_len-1)*con_seq_len
        
    def get_output_shape_for(self, input_shapes, **kwargs):
        assert len(input_shapes) == 3
        return input_shapes[0][:2]
    
    
    