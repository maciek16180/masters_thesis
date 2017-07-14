import theano.tensor as T
from lasagne.layers import MergeLayer

from lasagne.random import get_rng
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


class WordDropoutLayer(MergeLayer):
    def __init__(self, incoming, v, drop_rate=.25, **kwargs):
        self.rng = RandomStreams(get_rng().randint(1, 2147462579))
        self.drop_rate = drop_rate
        super(WordDropoutLayer, self).__init__([incoming, v], **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        input_, v = inputs
        bs, seq_len, emb_size = input_.shape
        if deterministic:
            return input_
        input_ = input_.reshape((bs * seq_len, emb_size))
        idx = self.rng.binomial((bs * seq_len,), p=self.drop_rate).nonzero()
        
        return T.set_subtensor(input_[idx], v).reshape((bs, seq_len, emb_size))