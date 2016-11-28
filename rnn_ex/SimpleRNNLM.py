# work in progress

import numpy as np
import theano
import theano.tensor as T

import lasagne as L

import sys
sys.path.insert(0, '../HSoftmaxLayerLasagne/')

from HSoftmaxLayer import HierarchicalSoftmaxDenseLayer
from SampledSoftmaxLayer import SampledSoftmaxDenseLayer


class SimpleRNNLM(object):
    
    def __init__(self, voc_size, emb_size, rec_size, mode='ssoft', **kwargs):

        input_var = T.imatrix('inputs')
        target_var = T.imatrix('targets')  # these will be inputs shifted by 1
        mask_input_var = T.matrix('input_mask')

        if mode == 'full':
            self.train_net = _build_full_softmax_net(input_var, mask_input_var, voc_size, emb_size, rec_size, **kwargs)
        elif mode == 'ssoft':
            num_sampled = kwargs['num_sampled']
            self.train_net = _build_sampled_softmax_net(input_var, mask_input_var, voc_size, emb_size, rec_size,
                                                        num_sampled, target_var=target_var, **kwargs)
        elif mode == 'hsoft':
            self.train_net = _build_hierarchical_softmax_net(input_var, mask_input_var, voc_size, emb_size, rec_size,
                                                             target_var=target_var, **kwargs)

    def save_params(self, fname='model.npz'):
        np.savez(fname, *L.layers.get_all_param_values(self.train_net))

    def load_params(self, fname='model.npz'):
        with np.load(fname) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            L.layers.set_all_param_values(self.train_net, param_values)


def _build_architecture(input_var, mask_input_var, voc_size, emb_size, rec_size,
                         emb_init=None, train_emb=True):
    l_in = L.layers.InputLayer(shape=(None, None), input_var=input_var)

    l_mask = None
    if mask_input_var is not None:
        print 'Setting up input mask...'
        l_mask = L.layers.InputLayer(shape=(None, None), input_var=mask_input_var)

    if emb_init is None:
        l_emb = L.layers.EmbeddingLayer(l_in,
                                        input_size=voc_size,  # not voc_size+1, because pad_value = <utt_end>
                                        output_size=emb_size)
    else:
        l_emb = L.layers.EmbeddingLayer(l_in,
                                        input_size=voc_size,
                                        output_size=emb_size,
                                        W=emb_init)
        if not train_emb:
            l_emb.params[l_emb.W].remove('trainable')

    l_lstm1 = L.layers.LSTMLayer(l_emb,
                                 num_units=rec_size,
                                 nonlinearity=L.nonlinearities.tanh,
                                 grad_clipping=100,
                                 mask_input=l_mask)

    l_lstm2 = L.layers.LSTMLayer(l_lstm1,
                                 num_units=rec_size,
                                 nonlinearity=L.nonlinearities.tanh,
                                 grad_clipping=100,
                                 mask_input=l_mask)

    l_resh = L.layers.ReshapeLayer(l_lstm2, shape=(-1, rec_size))

    return l_resh


def _build_full_softmax_net(input_var, mask_input_var, voc_size, emb_size, rec_size,
                             emb_init=None, train_emb=True, **kwargs):
    l_resh = _build_architecture(input_var, mask_input_var, voc_size, emb_size, rec_size,
                                  emb_init=emb_init, train_emb=train_emb)

    l_soft = L.layers.DenseLayer(l_resh,
                                 num_units=voc_size,
                                 nonlinearity=L.nonlinearities.softmax)

    l_out = L.layers.ReshapeLayer(l_soft, shape=(input_var.shape[0], input_var.shape[1], voc_size))

    return l_out


def _build_sampled_softmax_net(input_var, mask_input_var, voc_size, emb_size, rec_size, num_sampled,
                                emb_init=None, train_emb=True, target_var=None,
                                ssoft_probs=None, sample_unique=False, **kwargs):
    l_resh = _build_architecture(input_var, mask_input_var, voc_size, emb_size, rec_size,
                                  emb_init=emb_init, train_emb=train_emb)

    if target_var is not None:
        print 'Setting up targets for sampled softmax...'
        target_var = target_var.ravel()

    l_ssoft = SampledSoftmaxDenseLayer(l_resh, num_sampled, voc_size,
                                       targets=target_var,
                                       probs=ssoft_probs,
                                       sample_unique=sample_unique)

    if target_var is not None:
        l_out = L.layers.ReshapeLayer(l_ssoft, shape=(input_var.shape[0], input_var.shape[1]))
    else:
        l_out = L.layers.ReshapeLayer(l_ssoft, shape=(input_var.shape[0], input_var.shape[1], voc_size))

    return l_out


def _build_hierarchical_softmax_net(input_var, mask_input_var, voc_size, emb_size, rec_size,
                                     emb_init=None, train_emb=True, target_var=None, **kwargs):
    l_resh = _build_architecture(input_var, mask_input_var, voc_size, emb_size, rec_size,
                                  emb_init=emb_init, train_emb=train_emb)

    if target_var is not None:
        print 'Setting up targets for hierarchical softmax...'
        target_var = target_var.ravel()

    l_hsoft = HierarchicalSoftmaxDenseLayer(l_resh,
                                            num_units=voc_size,
                                            target=target_var)

    if target_var is not None:
        l_out = L.layers.ReshapeLayer(l_hsoft, shape=(input_var.shape[0], input_var.shape[1]))
    else:
        l_out = L.layers.ReshapeLayer(l_hsoft, shape=(input_var.shape[0], input_var.shape[1], voc_size))

    return l_out