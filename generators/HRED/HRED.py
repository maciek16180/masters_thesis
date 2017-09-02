import numpy as np
import theano
import theano.tensor as T
import time

import lasagne as L

import sys
sys.path.append('../SimpleRNNLM/')
sys.path.append('../')

from SimpleRNNLM import SimpleRNNLM

from layers import SampledSoftmaxDenseLayer
from layers import ShiftLayer
from layers import L2PoolingLayer


class HRED(SimpleRNNLM):

    def __init__(self, voc_size, emb_size, lv1_rec_size, lv2_rec_size, out_emb_size, mode='ssoft', n=3,
                 pad_value=-1, **kwargs):
        self.pad_value = pad_value
        self.voc_size = voc_size
        self.emb_size = emb_size
        self.lv1_rec_size = lv1_rec_size
        self.lv2_rec_size = lv2_rec_size
        self.out_emb_size = out_emb_size

        input_var = T.imatrix('inputs')
        target_var = T.imatrix('targets')  # these will be inputs shifted by 1
        mask_input_var = T.matrix('input_mask')
        mask_idx = mask_input_var.nonzero()

        context_init = T.matrix('context_init')
        decoder_init = T.matrix('decoder_init')

        # BUILD THE MODEL
        print 'Building the model...'

        assert mode in ['ssoft']

        self.train_net = _build_hred(input_var, mask_input_var, voc_size, emb_size, lv1_rec_size, lv2_rec_size,
                                     out_emb_size, target_var=target_var, n=n, **kwargs)

        # CALCULATE THE LOSS

        train_out = L.layers.get_output(self.train_net)
        test_out = L.layers.get_output(self.train_net, deterministic=True)

        train_loss = -T.log(train_out[mask_idx]).mean()
        test_loss = -T.log(test_out[mask_idx]).mean()

        # MAKE TRAIN AND VALIDATION FUNCTIONS
        print 'Compiling theano functions...'

        params = L.layers.get_all_params(self.train_net, trainable=True)

        if kwargs.has_key('update_fn'):
            update_fn = kwargs['update_fn']
        else:
            update_fn = lambda l, p: L.updates.adagrad(l, p, learning_rate=.01)

        updates = update_fn(train_loss, params)

        self.train_fn = theano.function([input_var, target_var, mask_input_var], train_loss, updates=updates)
        self.val_fn = theano.function([input_var, target_var, mask_input_var], test_loss)

        # BUILD NET FOR GENERATING, WITH SHARED PARAMETERS
        print 'Building a network for generating...'

        all_params = {x.name : x for x in L.layers.get_all_params(self.train_net)}

        self.context_net = _build_context_net_with_params(input_var, voc_size, emb_size, lv1_rec_size, lv2_rec_size,
                                                          context_init, all_params)
        self.decoder_net = _build_decoder_net_with_params(input_var, voc_size, emb_size, lv1_rec_size, lv2_rec_size,
                                                          out_emb_size, decoder_init, all_params)

        dec_net_out = L.layers.get_output(self.decoder_net, deterministic=True)
        new_con_init = L.layers.get_output(self.context_net, deterministic=True)

        self.get_probs_and_new_dec_init_fn = theano.function([input_var, decoder_init], dec_net_out)
        self.get_new_con_init_fn = theano.function([input_var, context_init], new_con_init)

        print 'Done'


'''
MODEL PARAMETERS (as in L.layers.get_params(train_net))

 index - description

     0 - emb
  1:10 - GRU forward lv1
 11:20 - GRU backward lv1
 21:30 - GRU session
 31:32 - dec init
 33:41 - GRU dec (without hid_init)
 41:42 - H0
    43 - E0
 44:46 - sampled softmax (p is unnecessary for generating)

 context_net: emb, GRUs lv1, GRU ses (no hid_init)
 decoder_net: emb, GRU dec (no hid_init), H0, E0, softmax (full, no p from ssoft)
'''

# "n" argument below is the maximum number of utterances in the sequence (so n=3 for Movie Triples).
# It doesn't affect number or shape of net parameters, so we can, for example, pretrain the net on short contexts
# and then feed it with longer ones. We have to rebuild the net with different n for that though.
# Because i use 2D input, batch_size has to be divisible by n.

# TODO: make it so we don't have to rebuild the net to feed in context with different n.
#       the input shape will be (batch_size x n x sequence_len)

def _build_hred(input_var, mask_input_var, voc_size, emb_size, lv1_rec_size, lv2_rec_size, out_emb_size,
                num_sampled, ssoft_probs=None, emb_init=None, train_emb=True, target_var=None, n=3, **kwargs):

    batch_size = input_var.shape[0]
    sequence_len = input_var.shape[1]

    ''' Inputs '''

    l_in = L.layers.InputLayer(shape=(None, None), input_var=input_var)

    l_mask = None
    if mask_input_var is not None:
        l_mask = L.layers.InputLayer(shape=(None, None), input_var=mask_input_var)

    ''' Word embeddings '''

    if emb_init is None:
        l_emb = L.layers.EmbeddingLayer(l_in,
                                        input_size=voc_size,  # not voc_size+1, because pad_value = <utt_end>
                                        output_size=emb_size,
                                        name='emb')
    else:
        l_emb = L.layers.EmbeddingLayer(l_in,
                                        input_size=voc_size,
                                        output_size=emb_size,
                                        W=emb_init,
                                        name='emb')
    if not train_emb:
        l_emb.params[l_emb.W].remove('trainable')

    ''' Level 1 (sentence) BiGRU encoding with L2-pooling '''

    l_lv1_enc_forw = L.layers.GRULayer(l_emb, # we process all utts in parallel, out_shape is batch_size x lv1_rec_size
                                       num_units=lv1_rec_size,
                                       grad_clipping=100,
                                       mask_input=l_mask,
                                       name='GRU1forw')

    l_lv1_enc_back = L.layers.GRULayer(l_emb, # backward pass of encoder rnn, out_shape is batch_size x lv1_rec_size
                                       num_units=lv1_rec_size,
                                       grad_clipping=100,
                                       mask_input=l_mask,
                                       backwards=True,
                                       name='GRU1back')

    l2_pooled_forw = L2PoolingLayer(l_lv1_enc_forw)
    l2_pooled_back = L2PoolingLayer(l_lv1_enc_back)

    l_lv1_enc = L.layers.ConcatLayer([l2_pooled_forw, l2_pooled_back], axis=1) # concatenation of L2-pooled states

    ''' Level 2 (context) encoding '''

    l_resh = L.layers.ReshapeLayer(l_lv1_enc, shape=(batch_size / n, n, 2 * lv1_rec_size))

    l_lv2_enc = L.layers.GRULayer(l_resh, # out_shape is batch_size/n x n x lv2_rec_size
                                  num_units=lv2_rec_size,
                                  grad_clipping=100,
                                  name='GRU2')

    ''' Decoder '''

    l_shift = ShiftLayer(l_lv2_enc) # we want to use i-th utterance summary as an init for decoding (i+1)-th

    l_resh2 = L.layers.ReshapeLayer(l_shift, shape=(batch_size, lv2_rec_size))

    l_dec_inits = L.layers.DenseLayer(l_resh2, # out_shape is batch_size x lv1_rec_size
                                      num_units=lv1_rec_size,
                                      nonlinearity=L.nonlinearities.tanh,
                                      name='dec_init')

    l_dec = L.layers.GRULayer(l_emb, # out_shape is batch_size x sequence_len x lv1_rec_size
                              num_units=lv1_rec_size,
                              grad_clipping=100,
                              mask_input=l_mask,
                              hid_init=l_dec_inits,
                              name='GRUdec')

    l_resh3 = L.layers.ReshapeLayer(l_dec, shape=(batch_size * sequence_len, lv1_rec_size))

    l_H0 = L.layers.DenseLayer(l_resh3,
                               num_units=out_emb_size,
                               nonlinearity=None,
                               name='h0')

    l_resh4 = L.layers.ReshapeLayer(l_emb, shape=(batch_size * sequence_len, emb_size))

    l_E0 = L.layers.DenseLayer(l_resh4,
                               num_units=out_emb_size,
                               b=None,
                               nonlinearity=None,
                               name='e0')

    l_soft_in = L.layers.ElemwiseSumLayer([l_H0, l_E0])

    if target_var is not None:
        target_var = target_var.ravel()

    l_ssoft = SampledSoftmaxDenseLayer(l_soft_in, num_sampled, voc_size,
                                       targets=target_var,
                                       probs=ssoft_probs,
                                       sample_unique=False,
                                       name='soft')

    if target_var is not None:
        l_out = L.layers.ReshapeLayer(l_ssoft, shape=(batch_size, sequence_len))
    else:
        l_out = L.layers.ReshapeLayer(l_ssoft, shape=(batch_size, sequence_len, voc_size))

    return l_out


def _build_context_net_with_params(input_var, voc_size, emb_size, lv1_rec_size, lv2_rec_size, context_init, params):

    l_in = L.layers.InputLayer(shape=(1,None), input_var=input_var)

    l_emb = L.layers.EmbeddingLayer(l_in,
                                    input_size=voc_size,  # not voc_size+1, because pad_value = <utt_end>
                                    output_size=emb_size,
                                    W=params['emb.W'])

    l_lv1_enc_forw = L.layers.GRULayer(l_emb,
                                       num_units=lv1_rec_size,
                                       grad_clipping=100,
                                       resetgate=L.layers.Gate(W_in=params['GRU1forw.W_in_to_resetgate'],
                                                               W_hid=params['GRU1forw.W_hid_to_resetgate'],
                                                               W_cell=None,
                                                               b=params['GRU1forw.b_resetgate']),
                                       updategate=L.layers.Gate(W_in=params['GRU1forw.W_in_to_updategate'],
                                                                W_hid=params['GRU1forw.W_hid_to_updategate'],
                                                                W_cell=None,
                                                                b=params['GRU1forw.b_updategate']),
                                       hidden_update=L.layers.Gate(W_in=params['GRU1forw.W_in_to_hidden_update'],
                                                                   W_hid=params['GRU1forw.W_hid_to_hidden_update'],
                                                                   W_cell=None,
                                                                   b=params['GRU1forw.b_hidden_update'],
                                                                   nonlinearity=L.nonlinearities.tanh),
                                       hid_init=params['GRU1forw.hid_init'])

    l_lv1_enc_back = L.layers.GRULayer(l_emb, # backward pass of encoder rnn
                                       num_units=lv1_rec_size,
                                       grad_clipping=100,
                                       backwards=True,
                                       resetgate=L.layers.Gate(W_in=params['GRU1back.W_in_to_resetgate'],
                                                               W_hid=params['GRU1back.W_hid_to_resetgate'],
                                                               W_cell=None,
                                                               b=params['GRU1back.b_resetgate']),
                                       updategate=L.layers.Gate(W_in=params['GRU1back.W_in_to_updategate'],
                                                                W_hid=params['GRU1back.W_hid_to_updategate'],
                                                                W_cell=None,
                                                                b=params['GRU1back.b_updategate']),
                                       hidden_update=L.layers.Gate(W_in=params['GRU1back.W_in_to_hidden_update'],
                                                                   W_hid=params['GRU1back.W_hid_to_hidden_update'],
                                                                   W_cell=None,
                                                                   b=params['GRU1back.b_hidden_update'],
                                                                   nonlinearity=L.nonlinearities.tanh),
                                       hid_init=params['GRU1back.hid_init'])

    l2_pooled_forw = L2PoolingLayer(l_lv1_enc_forw)
    l2_pooled_back = L2PoolingLayer(l_lv1_enc_back)

    l_lv1_enc = L.layers.ConcatLayer([l2_pooled_forw, l2_pooled_back]) # concatenation of L2-pooled states

    l_resh = L.layers.ReshapeLayer(l_lv1_enc, shape=(1, 1, 2*lv1_rec_size))

    l_lv2_enc = L.layers.GRULayer(l_resh,
                                  num_units=lv2_rec_size,
                                  hid_init=context_init,
                                  grad_clipping=100,
                                  only_return_final=True,
                                  resetgate=L.layers.Gate(W_in=params['GRU2.W_in_to_resetgate'],
                                                          W_hid=params['GRU2.W_hid_to_resetgate'],
                                                          W_cell=None,
                                                          b=params['GRU2.b_resetgate']),
                                  updategate=L.layers.Gate(W_in=params['GRU2.W_in_to_updategate'],
                                                           W_hid=params['GRU2.W_hid_to_updategate'],
                                                           W_cell=None,
                                                           b=params['GRU2.b_updategate']),
                                  hidden_update=L.layers.Gate(W_in=params['GRU2.W_in_to_hidden_update'],
                                                              W_hid=params['GRU2.W_hid_to_hidden_update'],
                                                              W_cell=None,
                                                              b=params['GRU2.b_hidden_update'],
                                                              nonlinearity=L.nonlinearities.tanh))

    return l_lv2_enc


def _build_decoder_net_with_params(input_var, voc_size, emb_size, lv1_rec_size, lv2_rec_size, out_emb_size,
                                   decoder_init, params):

    l_in = L.layers.InputLayer(shape=(None, None), input_var=input_var)

    l_emb = L.layers.EmbeddingLayer(l_in,
                                    input_size=voc_size,  # not voc_size+1, because pad_value = <utt_end>
                                    output_size=emb_size,
                                    W=params['emb.W'])

    l_dec_init = L.layers.InputLayer(shape=(None, lv1_rec_size), input_var=decoder_init)

    l_dec = L.layers.GRULayer(l_emb,
                              num_units=lv1_rec_size,
                              grad_clipping=100,
                              hid_init=l_dec_init,
                              only_return_final=True,
                              resetgate=L.layers.Gate(W_in=params['GRUdec.W_in_to_resetgate'],
                                                      W_hid=params['GRUdec.W_hid_to_resetgate'],
                                                      W_cell=None,
                                                      b=params['GRUdec.b_resetgate']),
                              updategate=L.layers.Gate(W_in=params['GRUdec.W_in_to_updategate'],
                                                       W_hid=params['GRUdec.W_hid_to_updategate'],
                                                       W_cell=None,
                                                       b=params['GRUdec.b_updategate']),
                              hidden_update=L.layers.Gate(W_in=params['GRUdec.W_in_to_hidden_update'],
                                                          W_hid=params['GRUdec.W_hid_to_hidden_update'],
                                                          W_cell=None,
                                                          b=params['GRUdec.b_hidden_update'],
                                                          nonlinearity=L.nonlinearities.tanh))

    l_H0 = L.layers.DenseLayer(l_dec,
                               num_units=out_emb_size,
                               nonlinearity=None,
                               W=params['h0.W'],
                               b=params['h0.b'])

    l_slice = L.layers.SliceLayer(l_emb, indices=-1, axis=1)

    l_E0 = L.layers.DenseLayer(l_slice,
                               num_units=out_emb_size,
                               W=params['e0.W'],
                               b=None,
                               nonlinearity=None)

    l_soft_in = L.layers.ElemwiseSumLayer([l_H0, l_E0])

    l_soft = L.layers.DenseLayer(l_soft_in,
                                 num_units=voc_size,
                                 nonlinearity=L.nonlinearities.softmax,
                                 W=params['soft.W'],
                                 b=params['soft.b'])

    l_out = L.layers.ReshapeLayer(l_soft, shape=(input_var.shape[0], voc_size))

    return l_out, l_dec # l_out - probabilities, l_dec - new decoder init