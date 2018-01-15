import theano.tensor as T
from lasagne.layers import Layer


class ShiftLayer(Layer):

    def get_output_for(self, input_, **kwargs):
        return T.concatenate(
            [T.zeros_like(input_[:, :1, :]), input_[:, :-1, :]], axis=1)
