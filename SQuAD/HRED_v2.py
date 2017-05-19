import numpy as np
import theano
import theano.tensor as T
import time

import lasagne as L
import lasagne.layers as LL

import sys
sys.path.insert(0, '../SimpleRNNLM/')
sys.path.insert(0, '../HRED/')

from SimpleRNNLM import iterate_minibatches, SimpleRNNLM
from SampledSoftmaxLayer import SampledSoftmaxDenseLayer

from ShiftLayer import ShiftLayer
from L2PoolingLayer import L2PoolingLayer
from ForgetSizeLayer import ForgetSizeLayer
from MaskedPaddedSoftmaxSumsLayer import MaskedPaddedSoftmaxSumsLayer

from itertools import chain

max_context_len = 28

class HRED(SimpleRNNLM):
    
    def __init__(self, voc_size, emb_size, lv1_rec_size, lv2_rec_size, out_emb_size, pad_value=-1, **kwargs):
        self.pad_value = pad_value
        self.voc_size = voc_size
        self.emb_size = emb_size
        self.lv1_rec_size = lv1_rec_size
        self.lv2_rec_size = lv2_rec_size
        self.out_emb_size = out_emb_size

        input_var = T.imatrix('inputs')
        bin_feat_var = T.matrix('bin_feat')
        mask_input_var = T.matrix('input_mask')
        mask_idx = mask_input_var.nonzero()
        
        ###
        mask_context_var = T.matrix('context_mask')
        answer_var = T.matrix('answer')
        ###
        
        context_init = T.matrix('context_init')
        decoder_init = T.matrix('decoder_init')

        # BUILD THE MODEL
        print 'Building the model...'

        self.train_net = _build_hred(input_var, bin_feat_var, mask_input_var, mask_context_var, voc_size, 
                                     emb_size, lv1_rec_size, lv2_rec_size, out_emb_size, **kwargs)

        # CALCULATE THE LOSS

        train_out = LL.get_output(self.train_net)
        test_out = LL.get_output(self.train_net, deterministic=True)
        
        def weighted_bin_ce(p, t, w=10):
            return -w * t * np.log(p) - (1 - t) * np.log(1 - p)

        train_loss = L.objectives.binary_crossentropy(train_out[mask_idx].ravel(), answer_var[mask_idx].ravel()).mean()
        test_loss = L.objectives.binary_crossentropy(test_out[mask_idx].ravel(), answer_var[mask_idx].ravel()).mean()
        
        #train_loss = weighted_bin_ce(train_out[mask_idx].ravel(), answer_var[mask_idx].ravel()).mean()
        #test_loss = weighted_bin_ce(test_out[mask_idx].ravel(), answer_var[mask_idx].ravel()).mean()

        # MAKE TRAIN AND VALIDATION FUNCTIONS
        print 'Compiling theano functions...'

        params = LL.get_all_params(self.train_net, trainable=True)

        if kwargs.has_key('update_fn'):
            update_fn = kwargs['update_fn']
        else:
            update_fn = lambda l, p: L.updates.adagrad(l, p, learning_rate=.01)

        updates = update_fn(train_loss, params)
        
        #
        # to train only part of the embeddings I can modify updates by hand here
        # I will need additional __init__ argument: indices of words that are fixed
        #
        
        self.train_fn = theano.function([input_var, bin_feat_var, answer_var, mask_input_var, mask_context_var], 
                                        train_loss, updates=updates)
        self.val_fn = theano.function([input_var, bin_feat_var, answer_var, mask_input_var, mask_context_var], 
                                      test_loss)
        
        self.get_output_fn = theano.function([input_var, bin_feat_var, mask_input_var, mask_context_var], test_out)

        # BUILD NET FOR GENERATING, WITH SHARED PARAMETERS
        
        #print 'Building a network for generating...'

        #all_params = LL.get_all_params(self.train_net)
        
        #self.context_net = _build_context_net_with_params(input_var, voc_size, emb_size, lv1_rec_size, lv2_rec_size,
        #                                                  context_init, all_params[:30])
        #self.decoder_net = _build_decoder_net_with_params(input_var, voc_size, emb_size, lv1_rec_size, lv2_rec_size,
        #                                                  out_emb_size, decoder_init, all_params[:1] + all_params[33:])
        
        #dec_net_out = LL.get_output(self.decoder_net, deterministic=True)
        #new_con_init = LL.get_output(self.context_net, deterministic=True)
        
        #self.get_probs_and_new_dec_init_fn = theano.function([input_var, decoder_init], dec_net_out)
        #self.get_new_con_init_fn = theano.function([input_var, context_init], new_con_init)
        
        print 'Done'
        
    def get_output(self, data, batch_size):
        result = []
        for batch in iterate_minibatches(data, batch_size, self.pad_value, with_answers=False):
            inputs, bin_feat, input_mask, context_mask = batch
            out = self.get_output_fn(inputs, bin_feat, input_mask, context_mask).reshape(batch_size, max_context_len, -1)
            input_mask = input_mask.reshape(batch_size, max_context_len, -1).astype(np.bool)
            context_mask = context_mask.astype(np.bool)
            
            for i in xrange(batch_size):
                out_i_list = []
                out_i = out[i][context_mask[i]]
                for j in xrange(1, out_i.shape[0]): # from 1, because the question is at 0
                    out_i_list.append(out_i[j][input_mask[i][j]])
                result.append(out_i_list)
                
        return result
            
    def train_one_epoch(self, train_data, batch_size, log_interval=10):
        train_err = 0.
        train_batches = 0
        num_training_words = 0
        start_time = time.time()

        for batch in iterate_minibatches(train_data, batch_size, self.pad_value):
            inputs, bin_feat, answers, input_mask, context_mask = batch

            num_batch_words = input_mask.sum()
            train_err += self.train_fn(inputs, bin_feat, answers, input_mask, context_mask) * num_batch_words
            train_batches += 1
            num_training_words += num_batch_words

            if not train_batches % log_interval:
                print "Done {} batches in {:.2f}s\ttraining loss:\t{:.6f}".format(
                    train_batches, time.time() - start_time, train_err / num_training_words)

        return  train_err / num_training_words

    def validate(self, val_data, batch_size):
        val_err = 0.
        val_batches = 0
        num_validate_words = 0
        start_time = time.time()

        for batch in iterate_minibatches(val_data, batch_size, self.pad_value):
            inputs, bin_feat, answers, input_mask, context_mask = batch

            num_batch_words = input_mask.sum()
            val_err += self.val_fn(inputs, bin_feat, answers, input_mask, context_mask) * num_batch_words
            val_batches += 1
            num_validate_words += num_batch_words

            if not val_batches % 100:
                print "Done {} batches in {:.2f}s".format(val_batches, time.time() - start_time)

        return val_err / num_validate_words

            
'''          
MODEL PARAMETERS (as in LL.get_params(train_net))

 index - description

     0 - emb
  1:10 - GRU forward lv1
 11:20 - GRU backward lv1
 21:30 - GRU session
 31:32 - dec init
 33:41 - GRU dec (without hid_init)
 41:42 - H0
    43 - E0
 44:45 - sigmoid

 context_net: emb, GRUs lv1, GRU ses (no hid_init)
 decoder_net: emb, GRU dec (no hid_init), H0, E0, sig
'''
            

def _build_hred(input_var, bin_feat_var, mask_input_var, mask_context_var, voc_size, emb_size, lv1_rec_size, 
                lv2_rec_size, out_emb_size, emb_init=None, train_emb=True, **kwargs):
    l_in = LL.InputLayer(shape=(None, None), input_var=input_var)
    
    l_bin_feat = LL.InputLayer(shape=(None, None), input_var=bin_feat_var)
    
    l_mask = None
    if mask_input_var is not None:
        l_mask = LL.InputLayer(shape=(None, None), input_var=mask_input_var)
        
    l_mask2 = None
    if mask_context_var is not None:
        l_mask2 = LL.InputLayer(shape=(None, None), input_var=mask_context_var)
    
    if emb_init is None:
        l_emb = LL.EmbeddingLayer(l_in,
                                        input_size=voc_size,  # not voc_size+1, because pad_value = <utt_end>
                                        output_size=emb_size)
    else:
        l_emb = LL.EmbeddingLayer(l_in,
                                        input_size=voc_size,
                                        output_size=emb_size,
                                        W=emb_init)
    if not train_emb:
        l_emb.params[l_emb.W].remove('trainable')
        
    # here we calculate wiq from https://arxiv.org/abs/1703.04816
    
    batch_size = input_var.shape[0] / max_context_len
    seq_len = input_var.shape[1]
    
    l_emb_temp = LL.reshape(l_emb, shape=(batch_size, max_context_len, seq_len, emb_size))
    l_qs = LL.reshape(LL.SliceLayer(l_emb_temp, 0, axis=1), shape=(batch_size * seq_len, emb_size))
    l_cs = LL.reshape(LL.SliceLayer(l_emb_temp, slice(1, max_context_len), axis=1), 
                      shape=(batch_size * (max_context_len - 1) * seq_len, emb_size))
    
    l_v1 = LL.DenseLayer(l_qs, 1, b=None)
    l_v2 = LL.DenseLayer(l_cs, 1, W=l_v1.W, b=None)
    
    l_sim1 = LL.reshape(l_v1, shape=(batch_size, seq_len, 1))
    l_sim2 = LL.reshape(l_v2, shape=(batch_size, 1, (max_context_len - 1) * seq_len))
    
    l_sim1 = ForgetSizeLayer(l_sim1, axis=2)
    l_sim2 = ForgetSizeLayer(l_sim2, axis=1)
    
    l_sim = LL.ElemwiseSumLayer([l_sim1, l_sim2])
    
    l_feat_mask = LL.reshape(LL.SliceLayer(LL.reshape(l_mask, shape=(batch_size, max_context_len, seq_len)), 
                                slice(1, max_context_len), axis=1), shape=(batch_size, 1, (max_context_len - 1) * seq_len))
    
    l_feat = MaskedPaddedSoftmaxSumsLayer([l_sim, l_feat_mask])
    
    l_weighted_feat = LL.reshape(l_feat, shape=(batch_size * max_context_len, seq_len, 1))
    
    
    ###
        
    l_bin_feat = LL.dimshuffle(l_bin_feat, (0, 1, 'x'))
        
    l_emb = LL.concat([l_emb, l_bin_feat, l_weighted_feat], axis=2)
            
    l_lv1_enc_forw = LL.GRULayer(l_emb, # we process all utts in parallel, out_shape is batch_size x lv1_rec_size
                                       num_units=lv1_rec_size,
                                       grad_clipping=100,
                                       # only_return_final=True,
                                       mask_input=l_mask)
    
    l_lv1_enc_back = LL.GRULayer(l_emb, # backward pass of encoder rnn, out_shape is batch_size x lv1_rec_size
                                       num_units=lv1_rec_size,
                                       grad_clipping=100,
                                       # only_return_final=True,
                                       mask_input=l_mask,
                                       backwards=True)
    
    l2_pooled_forw = L2PoolingLayer(l_lv1_enc_forw)
    l2_pooled_back = L2PoolingLayer(l_lv1_enc_back)

    l_lv1_enc = LL.concat([l2_pooled_forw, l2_pooled_back], axis=1) # concatenation of L2-pooled states
    
    l_resh = LL.reshape(l_lv1_enc, shape=(-1, max_context_len, 2*lv1_rec_size))
    
    l_lv2_enc = LL.GRULayer(l_resh, # out_shape is batch_size/max_context_len x max_context_len x lv2_rec_size
                                  num_units=lv2_rec_size,
                                  grad_clipping=100,
                                  mask_input=l_mask2)
    
    ############
    
    l_shift = ShiftLayer(l_lv2_enc)

    l_resh2 = LL.reshape(l_shift, shape=(-1, lv2_rec_size))
    
    l_dec_inits = LL.DenseLayer(l_resh2, # out_shape is batch_size x lv1_rec_size
                                      num_units=lv1_rec_size,
                                      nonlinearity=L.nonlinearities.tanh)
    
    l_dec = LL.GRULayer(l_emb, # out_shape is batch_size x seq_len x lv1_rec_size
                              num_units=lv1_rec_size,
                              grad_clipping=100,
                              mask_input=l_mask,
                              hid_init=l_dec_inits)
    
    l_resh3 = LL.reshape(l_dec, shape=(-1, lv1_rec_size))
    
    l_H0 = LL.DenseLayer(l_resh3,
                               num_units=out_emb_size,
                               nonlinearity=None)
    
    l_resh4 = LL.reshape(l_emb, shape=(-1, emb_size + 2))
    
    l_E0 = LL.DenseLayer(l_resh4,
                               num_units=out_emb_size,
                               b=None,
                               nonlinearity=None)
    
    l_sig_in = LL.ElemwiseSumLayer([l_H0, l_E0])

    ###
    
    l_sig = LL.DenseLayer(l_sig_in, 1, nonlinearity=L.nonlinearities.sigmoid)

    l_out = LL.reshape(l_sig, shape=(input_var.shape[0], input_var.shape[1]))

    return l_out


def _build_context_net_with_params(input_var, voc_size, emb_size, lv1_rec_size, lv2_rec_size, params):
    assert len(params) == 30
    params = [params[:1], params[1:11], params[11:21], params[21:]]
    em, lv1f, lv1b, lv2 = map(lambda p: {x.name: x for x in p}, params)
    
    l_in = LL.InputLayer(shape=(None,None), input_var=input_var)
    
    l_emb = LL.EmbeddingLayer(l_in,
                                    input_size=voc_size,  # not voc_size+1, because pad_value = <utt_end>
                                    output_size=emb_size,
                                    W=em['W'])
            
    l_lv1_enc_forw = LL.GRULayer(l_emb,
                                       num_units=lv1_rec_size,
                                       grad_clipping=100,
                                       # only_return_final=True,
                                       resetgate=LL.Gate(W_in=lv1f['W_in_to_resetgate'],
                                                               W_hid=lv1f['W_hid_to_resetgate'],
                                                               W_cell=None,
                                                               b=lv1f['b_resetgate']),
                                       updategate=LL.Gate(W_in=lv1f['W_in_to_updategate'],
                                                                W_hid=lv1f['W_hid_to_updategate'],
                                                                W_cell=None,
                                                                b=lv1f['b_updategate']),
                                       hidden_update=LL.Gate(W_in=lv1f['W_in_to_hidden_update'],
                                                                   W_hid=lv1f['W_hid_to_hidden_update'],
                                                                   W_cell=None,
                                                                   b=lv1f['b_hidden_update'],
                                                                   nonlinearity=L.nonlinearities.tanh),
                                       hid_init=lv1f['hid_init'])
    
    l_lv1_enc_back = LL.GRULayer(l_emb, # backward pass of encoder rnn
                                       num_units=lv1_rec_size,
                                       grad_clipping=100,
                                       # only_return_final=True,
                                       backwards=True,
                                       resetgate=LL.Gate(W_in=lv1b['W_in_to_resetgate'],
                                                               W_hid=lv1b['W_hid_to_resetgate'],
                                                               W_cell=None,
                                                               b=lv1b['b_resetgate']),
                                       updategate=LL.Gate(W_in=lv1b['W_in_to_updategate'],
                                                                W_hid=lv1b['W_hid_to_updategate'],
                                                                W_cell=None,
                                                                b=lv1b['b_updategate']),
                                       hidden_update=LL.Gate(W_in=lv1b['W_in_to_hidden_update'],
                                                                   W_hid=lv1b['W_hid_to_hidden_update'],
                                                                   W_cell=None,
                                                                   b=lv1b['b_hidden_update'],
                                                                   nonlinearity=L.nonlinearities.tanh),
                                       hid_init=lv1b['hid_init'])
    
    l2_pooled_forw = L2PoolingLayer(l_lv1_enc_forw)
    l2_pooled_back = L2PoolingLayer(l_lv1_enc_back)

    l_lv1_enc = LL.concat([l2_pooled_forw, l2_pooled_back]) # concatenation of L2-pooled states
    
    l_resh = LL.reshape(l_lv1_enc, shape=(1, -1, 2*lv1_rec_size))
    
    l_lv2_enc = LL.GRULayer(l_resh,
                                  num_units=lv2_rec_size,
                                  grad_clipping=100,
                                  only_return_final=False,
                                  resetgate=LL.Gate(W_in=lv2['W_in_to_resetgate'],
                                                          W_hid=lv2['W_hid_to_resetgate'],
                                                          W_cell=None,
                                                          b=lv2['b_resetgate']),
                                  updategate=LL.Gate(W_in=lv2['W_in_to_updategate'],
                                                           W_hid=lv2['W_hid_to_updategate'],
                                                           W_cell=None,
                                                           b=lv2['b_updategate']),
                                  hidden_update=LL.Gate(W_in=lv2['W_in_to_hidden_update'],
                                                              W_hid=lv2['W_hid_to_hidden_update'],
                                                              W_cell=None,
                                                              b=lv2['b_hidden_update'],
                                                              nonlinearity=L.nonlinearities.tanh))

    return l_lv2_enc # shape is 1 x num_sentences x lv2_rec_size


def _build_decoder_net_with_params(input_var, voc_size, emb_size, lv1_rec_size, lv2_rec_size, out_emb_size, 
                                   decoder_init, params):
    assert len(params) == 15
    params = [params[:1], params[1:10], params[10:12], params[12:13], params[13:]]
    em, dec, h0, e0, sm = map(lambda p: {x.name: x for x in p}, params)
    
    l_in = LL.InputLayer(shape=(None, None), input_var=input_var)
    
    l_emb = LL.EmbeddingLayer(l_in,
                                    input_size=voc_size,  # not voc_size+1, because pad_value = <utt_end>
                                    output_size=emb_size,
                                    W=em['W'])
    
    l_dec_init = LL.InputLayer(shape=(None, lv1_rec_size), input_var=decoder_init)
    
    l_dec = LL.GRULayer(l_emb,
                              num_units=lv1_rec_size,
                              grad_clipping=100,
                              hid_init=l_dec_init,
                              only_return_final=True,
                              resetgate=LL.Gate(W_in=dec['W_in_to_resetgate'],
                                                      W_hid=dec['W_hid_to_resetgate'],
                                                      W_cell=None,
                                                      b=dec['b_resetgate']),
                              updategate=LL.Gate(W_in=dec['W_in_to_updategate'],
                                                       W_hid=dec['W_hid_to_updategate'],
                                                       W_cell=None,
                                                       b=dec['b_updategate']),
                              hidden_update=LL.Gate(W_in=dec['W_in_to_hidden_update'],
                                                          W_hid=dec['W_hid_to_hidden_update'],
                                                          W_cell=None,
                                                          b=dec['b_hidden_update'],
                                                          nonlinearity=L.nonlinearities.tanh))
    
    l_H0 = LL.DenseLayer(l_dec,
                               num_units=out_emb_size,
                               nonlinearity=None,
                               W=h0['W'],
                               b=h0['b'])
    
    l_slice = LL.SliceLayer(l_emb, indices=-1, axis=1)
    
    l_E0 = LL.DenseLayer(l_slice,
                               num_units=out_emb_size,
                               W=e0['W'],
                               b=None,
                               nonlinearity=None)
    
    l_soft_in = LL.ElemwiseSumLayer([l_H0, l_E0])
    
    l_soft = LL.DenseLayer(l_soft_in,
                                 num_units=voc_size,
                                 nonlinearity=L.nonlinearities.softmax,
                                 W=sm['W'],
                                 b=sm['b'])
    
    l_out = LL.reshape(l_soft, shape=(input_var.shape[0], voc_size))

    return l_out, l_dec # l_out - probabilities, l_dec - new decoder init


def iterate_minibatches(inputs, batch_size, pad=-1, max_context_len=max_context_len, with_answers=True):
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        excerpt = slice(start_idx, start_idx + batch_size)
        examples = inputs[excerpt]

        max_seq_len = len(max([x for e in examples for x in e[1]], key=len))
        
        inp = []
        bin_feats = []
        if with_answers:
            answers = np.zeros((batch_size * max_context_len, max_seq_len), dtype=np.float32)
        
        for e in examples:
            e_padded = [l + [pad] * (max_seq_len - len(l)) for l in e[1]]
            q = set(e[1][0])
            bin_feat_padded = [[k in q for k in l] + [pad] * (max_seq_len - len(l)) for l in e[1]]
            
            e_padded = np.asarray(e_padded, dtype=np.int32)
            bin_feat_padded = np.asarray(bin_feat_padded, dtype=np.float32)

            fill = np.zeros((max_context_len - e_padded.shape[0], max_seq_len), dtype=np.int32) + pad
            e_padded = np.vstack([e_padded, fill])
            bin_feat_padded = np.vstack([bin_feat_padded, fill.astype(np.float32)])            
            
            inp.append(e_padded)
            bin_feats.append(bin_feat_padded)
            
            if with_answers:
                ans_inds = np.array(list(chain(*e[0])))

                idx = 1
                while ans_inds.size:
                    while ans_inds.size and ans_inds[0] < len(e[1][idx]):
                        ans_idx = ans_inds[0]
                        ans_inds = ans_inds[1:]
                        answers[idx, ans_idx] = 1
                    ans_inds -= len(e[1][idx])
                    idx += 1
            
        inp = np.vstack(inp)
        bin_feats = np.vstack(bin_feats)

        v_not_pad = np.vectorize(lambda x: x != pad, otypes=[np.float32])
        input_mask = v_not_pad(inp)  # there is no separate value for the end of an utterance, just pad
        
        context_mask = np.zeros(batch_size * max_context_len, dtype=np.float32)
        context_mask[inp[:, 0] != pad] = 1
        context_mask = context_mask.reshape(batch_size, max_context_len)
        
        if with_answers:
            yield inp, bin_feats, answers, input_mask, context_mask
        else:
            yield inp, bin_feats, input_mask, context_mask
