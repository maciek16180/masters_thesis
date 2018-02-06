import theano.tensor as T
from lasagne.layers import MergeLayer


class L2PoolingLayer(MergeLayer):

    def __init__(self, incoming, mask=None, **kwargs):
        assert len(incoming.output_shape) == 3
        incomings = [incoming]
        if mask is not None:
            incomings.append(mask)
        super(L2PoolingLayer, self).__init__(incomings, **kwargs)

    def get_output_for(self, inputs, **kwargs):
        assert len(inputs) in [1, 2]
        input_ = inputs[0]
        if len(inputs) == 2:
            mask = inputs[1]
            input_ *= mask.dimshuffle((0, 1, 'x'))
            return T.sqrt((input_**2).sum(axis=1) /
                          mask.sum(axis=1).dimshuffle((0, 'x')))
        else:
            return T.sqrt((input_**2).mean(axis=1))

    def get_output_shape_for(self, input_shapes, **kwargs):
        assert len(input_shapes) in [1, 2]
        assert len(input_shapes[0]) == 3
        return input_shapes[0][0], input_shapes[0][2]
