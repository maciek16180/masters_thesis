import theano.tensor as T
from lasagne.layers import MergeLayer


class ShiftLayer(MergeLayer):

    '''
    Shifts the 3D input right along the 2nd axis, padding
    the left side with pad. If pad=None, zero-padding is used.
    '''

    def __init__(self, incoming, pad=None, **kwargs):

        assert len(incoming.output_shape) == 3

        incomings = [incoming]

        if pad is not None:
            incomings.append(pad)

        super(ShiftLayer, self).__init__(incomings, **kwargs)

    def get_output_for(self, inputs, **kwargs):
        input_ = inputs[0]
        if len(inputs) == 2:
            pad = inputs[1]
            if pad.ndim == 2:
                pad = pad.dimshuffle(0, 'x', 1)
        else:
            pad = T.zeros_like(input_[:, :1, :])
        return T.concatenate([pad, input_[:, :-1, :]], axis=1)
