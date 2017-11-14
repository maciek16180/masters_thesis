import theano.tensor as TT
from lasagne.layers import Layer
from lasagne import init


class TrainPartOfEmbsLayer(Layer):

    '''
    This layer allows for training only a specified subset of embeddings.

    Parameters:
        incoming:    incoming layer of integers
        train_inds:  a list of indices of words to train
        E:           an array with all embeddings
        input_size:  dictionary size
        output_size: embedding size
        W:           initializer for vectors specified in train_inds,
                     defaults to Normal()

    Return:
        embedded incoming layer
    '''

    def __init__(self, incoming, train_inds, E, input_size, output_size,
                 W=init.Normal(), **kwargs):

        super(TrainPartOfEmbsLayer, self).__init__(incoming, **kwargs)

        self.output_size = output_size
        self.input_size = input_size
        self.train_inds = train_inds

        self.E = self.add_param(
            E, (self.input_size, self.output_size), name="E", trainable=False)

        self.W = None
        if self.train_inds:
            self.W = self.add_param(
                W, (len(self.train_inds), output_size), name="W")

    def get_output_for(self, input_, **kwargs):
        E = self.E
        if self.train_inds:
            E = TT.set_subtensor(self.E[self.train_inds], self.W)
        return E[input_]

    def get_output_shape_for(self, input_shape):
        return input_shape + (self.output_size, )
