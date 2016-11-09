# work in progress, clearly

import numpy as np
import theano
import theano.tensor as T

import lasagne as L

import sys
sys.path.insert(0, '../HSoftmaxLayerLasagne/')

from HSoftmaxLayer import HierarchicalSoftmaxDenseLayer
from SampledSoftmaxLayer import SampledSoftmaxDenseLayer


class SimpleRNNLM(object):
    
    def __init__(self, mode='ssoft'):
        pass

    @staticmethod
    def __build_net(input_var, voc_mask_var, voc_size, emb_size, rec_size,
                    mask_input_var= None, target_var=None, mode='ssoft', gclip=100):

        l_in = L.layers.InputLayer(shape=(None, None), input_var=input_var)
        batch_size, seq_len = l_in.input_var.shape

        l_mask = None
        if mask_input_var is not None:
            print 'setting up input mask...'
            l_mask = L.layers.InputLayer(shape=(batch_size, seq_len), input_var=mask_input_var)

        l_emb = L.layers.EmbeddingLayer(l_in,
                                        input_size=voc_size+1,
                                        output_size=emb_size)

        l_lstm1 = L.layers.LSTMLayer(l_emb,
                                     num_units=rec_size,
                                     nonlinearity=L.nonlinearities.tanh,
                                     grad_clipping=gclip,
                                     mask_input=l_mask)

        l_lstm2 = L.layers.LSTMLayer(l_lstm1,
                                     num_units=rec_size,
                                     nonlinearity=L.nonlinearities.tanh,
                                     grad_clipping=gclip,
                                     mask_input=l_mask)

        l_resh = L.layers.ReshapeLayer(l_lstm2, shape=(-1, rec_size))

        l_tar = None
        if target_var is not None:
            print 'setting up targets for sampled softmax...'
            l_tar = L.layers.InputLayer(shape=(-1,), input_var=target_var.reshape(shape=(batch_size * seq_len,)))

        if mode == 'ssoft':
            l_soft = SampledSoftmaxDenseLayer(l_resh, voc_mask_var, voc_size, targets=l_tar)
        elif mode == 'hsoft':
            l_soft = HierarchicalSoftmaxDenseLayer(l_resh, voc_size, target=l_tar)
        else:
            raise NameError('Mode not recognised.')

        if target_var is not None:
            l_out = L.layers.ReshapeLayer(l_soft, shape=(batch_size, seq_len))
        else:
            l_out = L.layers.ReshapeLayer(l_soft, shape=(batch_size, seq_len, voc_size))

        return l_out


    def save_params(self, fname='model.npz'):
        np.savez(fname, *self.params)

    @staticmethod
    def load_params(fname='model.npz'):
        with np.load(fname) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            L.layers.set_all_param_values(net, param_values)

    