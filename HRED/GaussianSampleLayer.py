import theano.tensor as T
import numpy as np
from lasagne import init
from lasagne.layers import MergeLayer

from lasagne.random import get_rng
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


class GaussianSampleLayer(MergeLayer):
    def __init__(self, mus, sigmas, **kwargs):
        self.rng = RandomStreams(get_rng().randint(1, 2147462579))
        super(GaussianSampleLayer, self).__init__([mus, sigmas], **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        mus, sigmas = inputs # sigma[i] is a 1D representation of a diagonal covariance matrix 
        if deterministic:
            return mus
        return mus + T.sqrt(sigmas) * self.rng.normal(inputs[0].shape)