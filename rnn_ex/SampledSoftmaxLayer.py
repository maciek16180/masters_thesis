import theano.tensor as T
import numpy as np
from lasagne import init
from lasagne.layers import Layer,MergeLayer, InputLayer


class SampledSoftmaxDenseLayer(MergeLayer):
    # samples is a mask vector of length voc_size
    
    def __init__(self, incoming, num_sampled, voc_size, 
                 targets=None, 
                 use_all_words=False,
                 sample_unique=True,
                 W_init = init.GlorotUniform(),
                 b_init = init.Constant(0),
                 **kwargs):

        assert len(incoming.output_shape) == 2
        
        incomings = [incoming]
        
        if targets is not None:
            if not isinstance(targets, Layer):
                assert targets.ndim == 1
                targets_shape = (incoming.output_shape[0],)
                targets = InputLayer(targets_shape, input_var=targets, name="targets inputlayer")

            incomings.append(targets)
        
        super(SampledSoftmaxDenseLayer,self).__init__(incomings,**kwargs)
        
        self.voc_size = voc_size
        self.num_sampled = num_sampled
        self.use_all_words = use_all_words
        self.sample_unique = sample_unique
        
        n_inputs = incoming.output_shape[1]
        self.W = self.add_param(W_init, (n_inputs, self.voc_size), name="W")
        self.b = self.add_param(b_init, (self.voc_size,), name="b", regularizable=False)
        
    def get_output_for(self, inputs, **kwargs):
        #
        # if targets are NOT provided: 
        #     returns softmax output calculated only on where samples == 1,
        #     output is surrounded by zeros to match the shape of a full softmax output
        # otherwise:
        #     returns sampled softmax output only for specified class for each sample
        
        assert len(inputs) in [1,2]
        input = inputs[0]
        
        # if targets are provided, return only probs for targets
        if len(inputs) == 2:
            targets = inputs[1]
            
            if not self.use_all_words:
                if self.sample_unique:
                    samples = T.as_tensor(np.random.choice(np.arange(self.voc_size),
                                                           size=self.num_sampled,
                                                           replace=False))
                else:
                    samples = T.as_tensor(np.random.randint(self.voc_size, size=self.num_sampled))

                true_logits = (input * self.W[:,targets].T).sum(axis=1) + self.b[targets]
                sampled_logits = input.dot(self.W[:,samples]) + self.b[samples]
                #
                # here we should subtract log(Q(y|x)), but using uniform sampling with replacement
                # makes all of the Q values equal, so it's unnecessary
                #
                logits = T.concatenate([true_logits.dimshuffle((0,'x')), sampled_logits], axis=1)
                ssoft = T.nnet.softmax(logits)
            
                return ssoft[:,0] # only values for targets
            
            else: # this part is for validation, where we use full softmax loss
                logits = input.dot(self.W) + self.b
                soft = T.nnet.softmax(logits)
            
                return soft[T.arange(soft.shape[0]), targets]
        
        # if targets are not provided, return full softmax                
        else:
            logits = input.dot(self.W) + self.b
            soft = T.nnet.softmax(logits)
            
            return soft
        
    def get_output_shape_for(self, input_shapes, **kwargs):
        assert len(input_shapes) in [1,2]
        if len(input_shapes) == 2:
            return (input_shapes[0][0],)
        else:
            return (input_shapes[0][0], self.voc_size)    
    
    
    