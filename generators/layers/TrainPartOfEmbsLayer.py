from lasagne.layers import Layer
from lasagne import init
import theano
import theano.tensor as TT
import numpy as np

from lasagne.random import get_rng
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


class TrainPartOfEmbsLayer(Layer):

    def __init__(self, incoming, train_inds, input_size, output_size,
                 E=init.Normal(), W=init.Normal(), keep_rate=1, **kwargs):

        self.rng = RandomStreams(get_rng().randint(1, 2147462579))
        self.keep_rate = keep_rate
        super(TrainPartOfEmbsLayer, self).__init__(incoming, **kwargs)

        self.output_size = output_size
        self.input_size = input_size
        self.train_inds = train_inds

        self.E = self.add_param(E, (self.input_size, self.output_size),
                                name="E", trainable=False)

        self.W = None
        if len(train_inds):
            self.W = self.add_param(W, (len(self.train_inds), output_size),
                                    name="W")

    def get_output_for(self, input_, deterministic=False, **kwargs):
        W = self.E
        if self.W is not None:
            W = TT.set_subtensor(self.E[self.train_inds], self.W)

        res = W[input_]

        if self.keep_rate < 1 and not deterministic:
            mask = self.rng.binomial((self.output_size,), p=self.keep_rate,
                                     dtype=theano.config.floatX)
            res_shape = res.shape
            res = res.reshape((-1, self.output_size))
            res = (res * mask).reshape(res_shape) / self.keep_rate

        return res

    def get_output_shape_for(self, input_shape):
        return input_shape + (self.output_size, )
