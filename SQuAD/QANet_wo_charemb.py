# similar to FastQA, https://arxiv.org/abs/1703.04816

import numpy as np
import theano
import theano.tensor as T
import time

import lasagne as L
import lasagne.layers as LL

import sys
sys.path.append('../SimpleRNNLM/')
sys.path.append('../HRED/')
sys.path.append('layers/')

from SimpleRNNLM import SimpleRNNLM

from WeightedFeatureLayer import WeightedFeatureLayer
from BatchedDotLayer import BatchedDotLayer
from MaskedSoftmaxLayer import MaskedSoftmaxLayer
from StartFeaturesLayer import StartFeaturesLayer
from EndFeaturesLayer import EndFeaturesLayer
from TrainPartOfEmbsLayer import TrainPartOfEmbsLayer

from itertools import chain


class QANet(SimpleRNNLM):
    
    def __init__(self, voc_size, emb_size, rec_size, train_inds, emb_init, pad_value=-1, **kwargs):
        
        self.pad_value = pad_value
        self.voc_size = voc_size
        self.emb_size = emb_size
        self.rec_size = rec_size
        
        context_var = T.imatrix('contexts')
        mask_context_var = T.matrix('context_mask')
        
        question_var = T.imatrix('questions')
        mask_question_var = T.matrix('question_mask')
        
        bin_feat_var = T.matrix('bin_feat')
        answer_starts_var = T.ivector('answer_starts')
        answer_ends_var = T.ivector('answer_ends')
        
        # BUILD THE MODEL
        print 'Building the model...'
        
        self.train_net = _build_net(context_var, question_var, bin_feat_var, mask_context_var, mask_question_var, 
                                    answer_starts_var, voc_size, emb_size, rec_size, train_inds, emb_init, **kwargs)

        # CALCULATE THE LOSS

        train_out = LL.get_output(self.train_net)
        test_out = LL.get_output(self.train_net, deterministic=True)
        
        start_probs = train_out[0][T.arange(context_var.shape[0]), answer_starts_var]
        end_conditional_probs = train_out[1][T.arange(context_var.shape[0]), answer_ends_var]
        span_probs = start_probs * end_conditional_probs

        train_loss = -T.log(span_probs).mean()

        params = LL.get_all_params(self.train_net, trainable=True)

        if kwargs.has_key('update_fn'):
            update_fn = kwargs['update_fn']
            print 'Using custom update_fn.'
        else:
            update_fn = lambda l, p: L.updates.adagrad(l, p, learning_rate=.01)

        updates = update_fn(train_loss, params)
        
        # to train only part of the embeddings I can modify updates by hand here?
        # I will need additional __init__ argument: indices of words that are fixed
        
        # MAKE THEANO FUNCTIONS
        print 'Compiling theano functions...'
        
        self.train_fn = theano.function([context_var, question_var, bin_feat_var, mask_context_var, mask_question_var, 
                                         answer_starts_var, answer_ends_var], train_loss, updates=updates)
        
        self.get_start_probs_fn = theano.function([context_var, question_var, bin_feat_var, mask_context_var,
                                                   mask_question_var], test_out[0])
        self.get_end_probs_fn = theano.function([context_var, question_var, bin_feat_var, mask_context_var,
                                                 mask_question_var, answer_starts_var], test_out[1])
        
        print 'Done'
        
       
    def get_start_probs(self, data, batch_size):
        result = []
        for batch in iterate_minibatches(data, batch_size, self.pad_value, with_answer_inds=False):
            questions, contexts, bin_feats, question_mask, context_mask = batch
            out = self.get_start_probs_fn(contexts, questions, bin_feats, context_mask, question_mask)
            result.append(out)
            
        return np.vstack(result)
    
    '''
    tu jest blad! rozne batche maja rozna dlugosc, trzeba to zrobic inaczej
    '''        
    
    
    # current implementation only allows for generating ends for one start index per example
    # to get end probs for k candidates for the start index, get_end_probs needs to be run k times
    def get_end_probs(self, data, answer_start_inds, batch_size):
        result = []
        idx = 0
        for batch in iterate_minibatches(data, batch_size, self.pad_value, with_answer_inds=False):
            questions, contexts, bin_feats, question_mask, context_mask = batch
            start_inds = answer_start_inds[idx:idx + batch_size]
            out = self.get_end_probs_fn(contexts, questions, bin_feats, context_mask, question_mask, start_inds)
            result.append(out)
            idx += batch_size
                
        return np.vstack(result)

            
    def train_one_epoch(self, train_data, batch_size, log_interval=10):
        train_err = 0.
        train_batches = 0
        start_time = time.time()

        for batch in iterate_minibatches(train_data, batch_size, self.pad_value, shuffle=True):
            questions, contexts, bin_feats, question_mask, context_mask, answer_inds = batch
            
            train_err += self.train_fn(contexts, questions, bin_feats, context_mask, question_mask, 
                                       answer_inds[:,0], answer_inds[:,1])
            train_batches += 1

            if not train_batches % log_interval:
                print "Done {} batches in {:.2f}s\ttraining loss:\t{:.6f}".format(
                    train_batches, time.time() - start_time, train_err / train_batches)

        return  train_err / train_batches
    
    
    def save_params(self, fname): # without the fixed word embeddings matrix
        np.savez(fname, *L.layers.get_all_param_values(self.train_net)[1:])
        

    def load_params(self, fname, E): # E is the fixed word embeddings matrix
        with np.load(fname) as f:
            param_values = [E] + [f['arr_%d' % i] for i in range(len(f.files))]
            L.layers.set_all_param_values(self.train_net, param_values)
    
            
'''          
MODEL PARAMETERS (as in LL.get_params(train_net))

    TODO
'''
            

def _build_net(context_var, question_var, bin_feat_var, mask_context_var, mask_question_var, answer_starts_var,
               voc_size, emb_size, rec_size, train_inds, emb_init, emb_dropout=False, **kwargs):
    
    ''' Inputs '''
    
    l_context = LL.InputLayer(shape=(None, None), input_var=context_var)
    l_question = LL.InputLayer(shape=(None, None), input_var=question_var)
    
    l_c_mask = None
    if mask_context_var is not None:
        l_c_mask = LL.InputLayer(shape=(None, None), input_var=mask_context_var)

    l_q_mask = None
    if mask_question_var is not None:
        l_q_mask = LL.InputLayer(shape=(None, None), input_var=mask_question_var) 
        
    ''' Word embeddings '''
    
    l_c_emb = TrainPartOfEmbsLayer(l_context,
                                   output_size=emb_size,
                                   input_size=voc_size,
                                   W=emb_init[train_inds],
                                   E=emb_init,
                                   train_inds=train_inds)
        
    l_q_emb = TrainPartOfEmbsLayer(l_question,
                                   output_size=emb_size,
                                   input_size=voc_size,
                                   W=l_c_emb.W,
                                   E=l_c_emb.E,
                                   train_inds=train_inds) 
        
    ''' Dropout at the embeddings '''
    
    if emb_dropout:
        l_c_emb = LL.dropout(l_c_emb)
        l_q_emb = LL.dropout(l_q_emb)
       
    ''' Here we calculate wiq features from https://arxiv.org/abs/1703.04816 '''
    
    l_feat = WeightedFeatureLayer([l_c_emb, l_q_emb, l_c_mask])    
    l_weighted_feat = LL.dimshuffle(l_feat, (0, 1, 'x'))
    
    l_bin_feat = LL.InputLayer(shape=(None, None), input_var=bin_feat_var)        
    l_bin_feat = LL.dimshuffle(l_bin_feat, (0, 1, 'x'))
        
    l_c_emb = LL.concat([l_c_emb, l_bin_feat, l_weighted_feat], axis=2) # both features are concatenated to the embeddings
    l_q_emb = LL.pad(l_q_emb, width=[(0, 2)], val=1, batch_ndim=2) # for the question we fix the features to 1
    
    ''' Context and question encoding using the same BiLSTM for both '''

    l_c_enc_forw = LL.LSTMLayer(l_c_emb, # output shape is (batch_size x context_len x rec_size)
                                num_units=rec_size,
                                grad_clipping=100,
                                mask_input=l_c_mask)
    
    l_c_enc_back = LL.LSTMLayer(l_c_emb,
                                num_units=rec_size,
                                grad_clipping=100,
                                mask_input=l_c_mask,
                                backwards=True)
    
    l_q_enc_forw = LL.LSTMLayer(l_q_emb, # output shape is (batch_size x question_len x rec_size)
                                num_units=rec_size,
                                grad_clipping=100,
                                mask_input=l_q_mask,
                                ingate=LL.Gate(W_in=l_c_enc_forw.W_in_to_ingate,
                                               W_hid=l_c_enc_forw.W_hid_to_ingate,
                                               W_cell=l_c_enc_forw.W_cell_to_ingate, 
                                               b=l_c_enc_forw.b_ingate),
                                forgetgate=LL.Gate(W_in=l_c_enc_forw.W_in_to_forgetgate,
                                                   W_hid=l_c_enc_forw.W_hid_to_forgetgate,
                                                   W_cell=l_c_enc_forw.W_cell_to_forgetgate, 
                                                   b=l_c_enc_forw.b_forgetgate),
                                outgate=LL.Gate(W_in=l_c_enc_forw.W_in_to_outgate,
                                                W_hid=l_c_enc_forw.W_hid_to_outgate,
                                                W_cell=l_c_enc_forw.W_cell_to_outgate, 
                                                b=l_c_enc_forw.b_outgate),
                                cell=LL.Gate(W_in=l_c_enc_forw.W_in_to_cell,
                                             W_hid=l_c_enc_forw.W_hid_to_cell,
                                             W_cell=None,
                                             b=l_c_enc_forw.b_cell,
                                             nonlinearity=L.nonlinearities.tanh))
    
    l_q_enc_back = LL.LSTMLayer(l_q_emb,
                                num_units=rec_size,
                                grad_clipping=100,
                                mask_input=l_q_mask,
                                backwards=True,                               
                                ingate=LL.Gate(W_in=l_c_enc_back.W_in_to_ingate,
                                               W_hid=l_c_enc_back.W_hid_to_ingate,
                                               W_cell=l_c_enc_back.W_cell_to_ingate, 
                                               b=l_c_enc_back.b_ingate),
                                forgetgate=LL.Gate(W_in=l_c_enc_back.W_in_to_forgetgate,
                                                   W_hid=l_c_enc_back.W_hid_to_forgetgate,
                                                   W_cell=l_c_enc_back.W_cell_to_forgetgate, 
                                                   b=l_c_enc_back.b_forgetgate),
                                outgate=LL.Gate(W_in=l_c_enc_back.W_in_to_outgate,
                                                W_hid=l_c_enc_back.W_hid_to_outgate,
                                                W_cell=l_c_enc_back.W_cell_to_outgate, 
                                                b=l_c_enc_back.b_outgate),
                                cell=LL.Gate(W_in=l_c_enc_back.W_in_to_cell,
                                             W_hid=l_c_enc_back.W_hid_to_cell,
                                             W_cell=None,
                                             b=l_c_enc_back.b_cell,
                                             nonlinearity=L.nonlinearities.tanh))
    

    l_c_enc = LL.concat([l_c_enc_forw, l_c_enc_back], axis=2) # batch_size x context_len x 2*rec_size
    l_q_enc = LL.concat([l_q_enc_forw, l_q_enc_back], axis=2) # batch_size x question_len x 2*rec_size
        
    # this is H from the paper
    l_c_proj = LL.DenseLayer(LL.reshape(l_c_enc, (-1, 2 * rec_size)), # batch_size * context_len x rec_size
                             num_units=rec_size,
                             W=np.vstack([np.eye(rec_size), np.eye(rec_size)]),
                             b=None,
                             nonlinearity=L.nonlinearities.tanh)
    
    # this is Z from the paper
    l_q_proj = LL.DenseLayer(LL.reshape(l_q_enc, (-1, 2 * rec_size)), # batch_size * question_len x rec_size
                             num_units=rec_size,
                             W=np.vstack([np.eye(rec_size), np.eye(rec_size)]),
                             b=None,
                             nonlinearity=L.nonlinearities.tanh)
        
    ''' Additional, weighted question encoding (alphas from https://arxiv.org/abs/1703.04816) '''
    
    batch_size = question_var.shape[0]
    context_len = context_var.shape[1]
    
    l_alpha = LL.DenseLayer(l_q_proj, # batch_size * question_len x 1
                            num_units=1,
                            b=None,
                            nonlinearity=None)
    
    l_alpha = MaskedSoftmaxLayer([LL.reshape(l_alpha, (batch_size, -1)), l_q_mask])
    
    l_z_hat = BatchedDotLayer([LL.reshape(l_q_proj, (batch_size, -1, rec_size)), l_alpha]) # batch_size x rec_size
    
    ''' Answer span prediction '''
    
    # span start
    
    l_start_feat = StartFeaturesLayer([LL.reshape(l_c_proj, (batch_size, context_len, rec_size)), l_z_hat])
    
    l_start = LL.DenseLayer(LL.reshape(l_start_feat, (batch_size * context_len, 3*rec_size)),
                            num_units=rec_size,
                            nonlinearity=L.nonlinearities.rectify) # batch_size * context_len x rec_size
    
    l_Vs = LL.DenseLayer(l_start, # batch_size * context_len x 1
                         num_units=1,
                         b=None,
                         nonlinearity=None)
    
    # this is p_s from the paper
    l_start_soft = MaskedSoftmaxLayer([LL.reshape(l_Vs, (batch_size, context_len)), l_c_mask]) # batch_size x context_len
    
    # span end
    
    l_answer_starts = LL.InputLayer(shape=(None,), input_var=answer_starts_var)
    
    l_end_feat = EndFeaturesLayer([LL.reshape(l_c_proj, (batch_size, context_len, rec_size)), l_z_hat, l_answer_starts])
    
    l_end = LL.DenseLayer(LL.reshape(l_end_feat, (batch_size * context_len, 5*rec_size)),
                          num_units=rec_size,
                          nonlinearity=L.nonlinearities.rectify) # batch_size * context_len x rec_size
    
    l_Ve = LL.DenseLayer(l_end, # batch_size * context_len x 1
                         num_units=1,
                         b=None,
                         nonlinearity=None)
    
    # this is p_e from the paper
    l_end_soft = MaskedSoftmaxLayer([LL.reshape(l_Ve, (batch_size, context_len)), l_c_mask]) # batch_size x context_len
    
    return l_start_soft, l_end_soft


def iterate_minibatches(inputs, batch_size, pad=-1, with_answer_inds=True, shuffle=False):
    
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
        inputs = np.array(inputs)
    
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
            
        examples = inputs[excerpt]

        context_len = max(len(e[2]) for e in examples)
        question_len = max(len(e[1]) for e in examples)
        
        questions = []
        contexts = []
        bin_feats = []
        
        if with_answer_inds:
            answer_inds = []
        
        for ans, q, c in examples:
            q_words = set(q)
            q = q + [pad] * (question_len - len(q))
            
            bin_feat = [w in q_words for w in c] + [pad] * (context_len - len(c))
            c = c + [pad] * (context_len - len(c))
            
            if with_answer_inds:
                answer_inds.append((min(ans[0]), max(ans[0])))
                
            questions.append(q)
            contexts.append(c)
            bin_feats.append(bin_feat)
            
        questions = np.vstack(questions).astype(np.int32)
        contexts = np.vstack(contexts).astype(np.int32)
        bin_feats = np.vstack(bin_feats).astype(np.float32)
        
        if with_answer_inds:
            answer_inds = np.vstack(answer_inds).astype(np.int32)
        
        question_mask = (questions != pad).astype(np.float32)
        context_mask = (contexts != pad).astype(np.float32)
        
        if with_answer_inds:
            yield questions, contexts, bin_feats, question_mask, context_mask, answer_inds
        else:
            yield questions, contexts, bin_feats, question_mask, context_mask
