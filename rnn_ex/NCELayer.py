import theano.tensor as T
import numpy as np
from lasagne import init
from lasagne.layers import Layer, MergeLayer, InputLayer

from lasagne.random import get_rng
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class NCEDenseLayer(MergeLayer):
    
    def __init__(self, incoming, num_sampled, voc_size, 
                 targets=None,
                 sample_unique=False,
                 probs=None,
                 W=init.GlorotUniform(),
                 b=init.Constant(0),
                 **kwargs):

        assert len(incoming.output_shape) == 2
        
        incomings = [incoming]
        
        if targets is not None:
            if not isinstance(targets, Layer):
                assert targets.ndim == 1
                targets_shape = (incoming.output_shape[0],)
                targets = InputLayer(targets_shape, input_var=targets, name="targets inputlayer")

            incomings.append(targets)
        
        super(NCEDenseLayer, self).__init__(incomings, **kwargs)
        
        self.voc_size = voc_size
        self.num_sampled = num_sampled
        self.sample_unique = sample_unique
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        
        if probs is None:
            probs = np.ones(voc_size) / float(voc_size)
        
        n_inputs = incoming.output_shape[1]
        self.W = self.add_param(W, (n_inputs, self.voc_size), name="W")
        self.b = self.add_param(b, (self.voc_size,), name="b", regularizable=False)
        self.p = self.add_param(probs, (self.voc_size,), name="p", trainable=False)
        
    def get_output_for(self, inputs, deternimistic=False, **kwargs):
        
        assert len(inputs) in [1,2]
        input_ = inputs[0]

        # if targets are provided, return only probs for targets
        if len(inputs) == 2:
            targets = inputs[1]
            
            # here we sample num_sampled noise classes for NCE
            if not deternimistic:
                
                if self.sample_unique:
                    samples = self._srng.multinomial_wo_replacement(n=self.num_sampled, pvals=[self.p]).ravel()
                else:
                    bins = self._srng.multinomial(n=self.num_sampled, pvals=[self.p]).ravel()
                    samples = T.extra_ops.repeat(bins.nonzero()[0], bins.nonzero_values())

                true_logits = (input_ * self.W[:,targets].T).sum(axis=1) + self.b[targets]
                sampled_logits = input_.dot(self.W[:,samples]) + self.b[samples]
                
                if self.sample_unique:
                    raise NotImplementedError('Not implemented: computation of Q(y|x)')
                else:
                    true_logits -= T.log(self.p[targets] * self.num_sampled)
                    sampled_logits -= T.log(self.p[samples] * self.num_sampled)

                true_part = T.max([true_logits, T.zeros_like(true_logits)], axis=0) - \
                            true_logits + T.log(1 + T.exp(-T.abs_(true_logits)))
                sampled_part = (T.max([sampled_logits, T.zeros_like(sampled_logits)], axis=0) + \
                               T.log(1 + T.exp(-T.abs_(sampled_logits)))).sum(axis=1)

                return true_part + sampled_part
            
            # this part is for validation, where we use full softmax loss
            else:
                logits = input_.dot(self.W) + self.b
                soft = T.nnet.softmax(logits)
            
                return soft[T.arange(soft.shape[0]), targets]
        
        # if targets are not provided, return full softmax                
        else:
            logits = input_.dot(self.W) + self.b
            soft = T.nnet.softmax(logits)
            
            return soft
        
    def get_output_shape_for(self, input_shapes, **kwargs):
        assert len(input_shapes) in [1,2]
        if len(input_shapes) == 2:
            return input_shapes[0][0],
        else:
            return input_shapes[0][0], self.voc_size
    
    
    