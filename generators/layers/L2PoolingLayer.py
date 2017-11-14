import theano.tensor as T
from lasagne.layers import Layer


class L2PoolingLayer(Layer):

    def get_output_for(self, input_, **kwargs):
        return T.sqrt((input_**2).mean(axis=1))

    def get_output_shape_for(self, input_shapes, **kwargs):
        assert len(input_shapes) == 3
        return input_shapes[0], input_shapes[2]
