from lasagne.layers import Layer
from lasagne import init
import theano.tensor as TT
import numpy as np


class TrainPartOfEmbsLayer(Layer):   
    
    def __init__(self, incoming, train_inds, E, input_size, output_size, 
                 W=init.Normal(), **kwargs):
        
        super(TrainPartOfEmbsLayer, self).__init__(incoming, **kwargs)
        
        self.output_size = output_size
        self.input_size = input_size
        self.train_inds = train_inds
        
        self.E = self.add_param(E, (self.input_size, self.output_size), name="E", trainable=False)
        
        self.W = None
        if len(train_inds):
            self.W = self.add_param(W, (len(self.train_inds), output_size), name="W")
        
        
    def get_output_for(self, input_, **kwargs):
        W = self.E
        if self.W is not None:
            W = TT.set_subtensor(self.E[self.train_inds], self.W)
        return W[input_]
    
    def get_output_shape_for(self, input_shape):
        return input_shape + (self.output_size, )