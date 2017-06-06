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
from HighwayLayer import HighwayLayer
from ForgetSizeLayer import ForgetSizeLayer

from itertools import chain


class QANet(SimpleRNNLM):
    
    def __init__(self, voc_size, alphabet_size, emb_size, emb_char_size, num_emb_char_filters, rec_size, 
                 pad_value=-1, **kwargs):
        
        self.pad_value = pad_value
        self.voc_size = voc_size
        self.emb_size = emb_size
        self.rec_size = rec_size
        
        context_var = T.imatrix('contexts')
        mask_context_var = T.matrix('context_mask')
        
        question_var = T.imatrix('questions')
        mask_question_var = T.matrix('question_mask')
        
        context_char_var = T.itensor3('context_char')
        mask_context_char_var = T.tensor3('context_char_mask')
        
        question_char_var = T.itensor3('question_char')
        mask_question_char_var = T.tensor3('question_char_mask')
        
        bin_feat_var = T.matrix('bin_feat')
        answer_starts_var = T.ivector('answer_starts')
        answer_ends_var = T.ivector('answer_ends')
        
        # BUILD THE MODEL
        print 'Building the model...'
        
        self.train_net = _build_net(context_var, question_var, context_char_var, question_char_var, bin_feat_var,
                                    mask_context_var, mask_question_var, mask_context_char_var, mask_question_char_var,
                                    answer_starts_var, voc_size, alphabet_size, emb_size, emb_char_size, 
                                    num_emb_char_filters, rec_size, **kwargs)

        # CALCULATE THE LOSS

        train_out = LL.get_output(self.train_net)
        test_out = LL.get_output(self.train_net, deterministic=True)
        
        start_probs = train_out[0][T.arange(context_var.shape[0]), answer_starts_var]
        end_conditional_probs = train_out[1][T.arange(context_var.shape[0]), answer_ends_var]
        span_probs = start_probs * end_conditional_probs

        train_loss = -T.log(span_probs).mean()

        params = LL.get_all_params(self.train_net, trainable=True)

        if kwargs.has_key('update_fn'):
            print 'Using custom update_fn.'
            update_fn = kwargs['update_fn']
        else:
            update_fn = lambda l, p: L.updates.adagrad(l, p, learning_rate=.01)

        updates = update_fn(train_loss, params)
        
        # MAKE THEANO FUNCTIONS
        print 'Compiling theano functions:'
        
        # to train only part of the embeddings I can modify updates by hand here?
        # I will need additional __init__ argument: indices of words that are fixed
        
        print '    train_fn...'
        
        self.train_fn = theano.function([context_var, question_var, context_char_var, question_char_var, bin_feat_var,
                                         mask_context_var, mask_question_var, mask_context_char_var, mask_question_char_var,
                                         answer_starts_var, answer_ends_var], train_loss, updates=updates)
        
        compile_pred_fns = not kwargs.get('skip_pred_fns', False)
        
        if compile_pred_fns:
            
            print '    get_start_probs_fn...'

            self.compile_get_start_probs_fn([context_var, question_var, context_char_var, question_char_var,
                                             bin_feat_var, mask_context_var, mask_question_var, mask_context_char_var,
                                             mask_question_char_var], test_out[0])

            print '    get_end_probs_fn...'

            self.compile_get_end_probs_fn([context_var, question_var, context_char_var, question_char_var, 
                                           bin_feat_var, mask_context_var, mask_question_var, mask_context_char_var,
                                           mask_question_char_var, answer_starts_var], test_out[1])
        else:
            print 'Skipping predictions functions.'
        
        print 'Done'
        
    
    def compile_get_start_probs_fn(self, variables, out):
        self.get_start_probs_fn = theano.function(variables, out)
        
        
    def compile_get_end_probs_fn(self, variables, out):
        self.get_end_probs_fn = theano.function(variables, out)
        
       
    def get_start_probs(self, data, batch_size):
        result = []
        for batch in iterate_minibatches(data, batch_size, self.pad_value, with_answer_inds=False):
            questions, contexts, questions_char, contexts_char, bin_feats, \
                question_mask, context_mask, question_char_mask, context_char_mask = batch
            out = self.get_start_probs_fn(contexts, questions, contexts_char, questions_char, bin_feats, 
                                          context_mask, question_mask, context_char_mask, question_char_mask)
            result.append(out)
            
        return np.vstack(result)
    
    '''
    tu jest blad! rozne batche maja rozna dlugosc, trzeba to zrobic inaczej (albo wsadzac jednobatchowe porcje danych)
    (batchowanie na zewnatrz)
    '''    
    
    # current implementation only allows for generating ends for one start index per example
    # to get end probs for k candidates for the start index, get_end_probs needs to be run k times
    def get_end_probs(self, data, answer_start_inds, batch_size):
        result = []
        idx = 0
        for batch in iterate_minibatches(data, batch_size, self.pad_value, with_answer_inds=False):
            questions, contexts, questions_char, contexts_char, bin_feats, \
                question_mask, context_mask, question_char_mask, context_char_mask = batch
            start_inds = answer_start_inds[idx:idx + batch_size]
            out = self.get_end_probs_fn(contexts, questions, contexts_char, questions_char, bin_feats, 
                                        context_mask, question_mask, context_char_mask, question_char_mask, start_inds)
            result.append(out)
            idx += batch_size
                
        return np.vstack(result)

            
    def train_one_epoch(self, train_data, batch_size, log_interval=10):
        train_err = 0.
        train_batches = 0
        start_time = time.time()

        for batch in iterate_minibatches(train_data, batch_size, self.pad_value, shuffle=True):
            questions, contexts, questions_char, contexts_char, bin_feats, \
                question_mask, context_mask, question_char_mask, context_char_mask, answer_inds = batch
            
            train_err += self.train_fn(contexts, questions, contexts_char, questions_char, bin_feats, 
                                       context_mask, question_mask, context_char_mask, question_char_mask, 
                                       answer_inds[:,0], answer_inds[:,1])
            train_batches += 1

            if not train_batches % log_interval:
                print "Done {} batches in {:.2f}s\ttraining loss:\t{:.6f}".format(
                    train_batches, time.time() - start_time, train_err / train_batches)

        return  train_err / train_batches
    
            
'''          
MODEL PARAMETERS (as in LL.get_params(train_net))

    TODO
'''
            

def _build_net(context_var, question_var, context_char_var, question_char_var, bin_feat_var, mask_context_var,
               mask_question_var, mask_context_char_var, mask_question_char_var, answer_starts_var, voc_size, alphabet_size,
               emb_size, emb_char_size, num_emb_char_filters, rec_size, 
               emb_char_filter_size=5, emb_init=None, train_emb=True, emb_char_init=None, train_emb_char=True, 
               **kwargs):
    
    batch_size = question_var.shape[0]
    context_len = context_var.shape[1]
    question_len = question_var.shape[1]
    context_word_len = context_char_var.shape[2]
    question_word_len = question_char_var.shape[2]
    
    ''' Inputs '''
    
    l_context = LL.InputLayer(shape=(None, None), input_var=context_var)
    l_question = LL.InputLayer(shape=(None, None), input_var=question_var)
    
    l_context_char = LL.InputLayer(shape=(None, None, None), input_var=context_char_var)
    l_question_char = LL.InputLayer(shape=(None, None, None), input_var=question_char_var)
        
    l_c_mask = None
    if mask_context_var is not None:
        l_c_mask = LL.InputLayer(shape=(None, None), input_var=mask_context_var)

    l_q_mask = None
    if mask_question_var is not None:
        l_q_mask = LL.InputLayer(shape=(None, None), input_var=mask_question_var)
        
    l_c_char_mask = LL.InputLayer(shape=(None, None, None), input_var=mask_context_char_var)
    l_q_char_mask = LL.InputLayer(shape=(None, None, None), input_var=mask_question_char_var)
    
    ''' Word embeddings '''
    
    # voc_size should be 1 larger, to make the pad_value of -1 have its own vector at the end, just to be sure
    if emb_init is None:
        l_c_emb = LL.EmbeddingLayer(l_context,
                                    input_size=voc_size,
                                    output_size=emb_size)
    else:
        l_c_emb = LL.EmbeddingLayer(l_context,
                                    input_size=voc_size,
                                    output_size=emb_size,
                                    W=emb_init)
        
    l_q_emb = LL.EmbeddingLayer(l_question,
                                input_size=voc_size,
                                output_size=emb_size,
                                W=l_c_emb.W)
    if not train_emb:
        l_c_emb.params[l_c_emb.W].remove('trainable')
        l_q_emb.params[l_q_emb.W].remove('trainable')
        
    ''' Char-embeddings '''
        
    if emb_char_init is None:
        l_c_char_emb = LL.EmbeddingLayer(l_context_char,
                                         input_size=alphabet_size,
                                         output_size=emb_char_size) 
    else:
        l_c_char_emb = LL.EmbeddingLayer(l_context_char,
                                         input_size=alphabet_size,
                                         output_size=emb_char_size,
                                         W=emb_char_init)
    # l_c_char_emb.shape is (batch_size x context_len x context_word_len x emb_char_size)
        
    l_q_char_emb = LL.EmbeddingLayer(l_question_char,
                                     input_size=alphabet_size,
                                     output_size=emb_char_size,
                                     W=l_c_char_emb.W)
    
    if not train_emb_char:
        l_c_char_emb.params[l_c_char_emb.W].remove('trainable')
        l_q_char_emb.params[l_q_char_emb.W].remove('trainable')
    
    # here I do multiplication of character embeddings with masks, because I want to pad them with constant zeros
    # (I don't want those padding zeros to change over time)
    
    l_c_char_mask = ForgetSizeLayer(LL.dimshuffle(l_c_char_mask, (0, 1, 2, 'x')))
    l_q_char_mask = ForgetSizeLayer(LL.dimshuffle(l_q_char_mask, (0, 1, 2, 'x')))
    
    l_c_char_emb = LL.ElemwiseMergeLayer([l_c_char_emb, l_c_char_mask], T.mul)
    l_q_char_emb = LL.ElemwiseMergeLayer([l_q_char_emb, l_q_char_mask], T.mul)
    
    #
        
    l_c_char_emb = LL.dimshuffle(LL.reshape(l_c_char_emb, (batch_size * context_len, context_word_len, emb_char_size)), 
                                 (0, 2, 1))    
    l_c_char_conv = LL.Conv1DLayer(l_c_char_emb,
                                   num_filters=num_emb_char_filters, 
                                   filter_size=emb_char_filter_size,
                                   nonlinearity=L.nonlinearities.tanh) 
    # (batch_size * context_len x num_filters x context_word_len - filter_size + 1)
    
    l_c_char_emb = LL.ExpressionLayer(l_c_char_conv, lambda X: X.max(2), output_shape='auto')    
    l_c_char_emb = LL.reshape(l_c_char_emb, (batch_size, context_len, num_emb_char_filters))    
    l_c_emb = LL.concat([l_c_emb, l_c_char_emb], axis=2)
    
    
    l_q_char_emb = LL.dimshuffle(LL.reshape(l_q_char_emb, (batch_size * question_len, question_word_len, emb_char_size)), 
                                 (0, 2, 1))    
    l_q_char_conv = LL.Conv1DLayer(l_q_char_emb,
                                   num_filters=num_emb_char_filters, 
                                   filter_size=emb_char_filter_size,
                                   nonlinearity=L.nonlinearities.tanh,
                                   W=l_c_char_conv.W,
                                   b=l_c_char_conv.b)
    # (batch_size * question_len x num_filters x question_word_len - filter_size + 1)
    
    l_q_char_emb = LL.ExpressionLayer(l_q_char_conv, lambda X: X.max(2), output_shape='auto')    
    l_q_char_emb = LL.reshape(l_q_char_emb, (batch_size, question_len, num_emb_char_filters))    
    l_q_emb = LL.concat([l_q_emb, l_q_char_emb], axis=2)
    
    ''' Highway layer allowing for interaction between embeddings '''
    
    l_c_P = LL.DenseLayer(LL.reshape(l_c_emb, (batch_size * context_len, emb_size + num_emb_char_filters)),
                          num_units=rec_size,
                          b=None,
                          nonlinearity=None)
    
    l_c_high = HighwayLayer(l_c_P)
    l_c_emb = LL.reshape(l_c_high, (batch_size, context_len, rec_size))
    
    l_q_P = LL.DenseLayer(LL.reshape(l_q_emb, (batch_size * question_len, emb_size + num_emb_char_filters)),
                          num_units=rec_size,
                          W=l_c_P.W,
                          b=None,
                          nonlinearity=None)
    
    l_q_high = HighwayLayer(l_q_P,
                            W1=l_c_high.W1,
                            b1=l_c_high.b1,
                            W2=l_c_high.W2,
                            b2=l_c_high.b2)
    l_q_emb = LL.reshape(l_q_high, (batch_size, question_len, rec_size))    
        
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
  
    inputs, inputs_char = inputs
    
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
        inputs = np.array(inputs)
        inputs_char = np.array(inputs_char)
    
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
            
        examples = inputs[excerpt]
        examples_char = inputs_char[excerpt]

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
        
        c_word_len = max(len(w) for e in examples_char for w in e[1])
        q_word_len = max(len(w) for e in examples_char for w in e[0])
        
        questions_char = []
        contexts_char = []
        
        for q, c in examples_char:
            q = q + [[]] * (question_len - len(q))
            c = c + [[]] * (context_len - len(c))            
            q = [w + [pad] * (q_word_len - len(w)) for w in q]
            if not all(len(x) == q_word_len for x in q):
                print map(len, q)
            c = [w + [pad] * (c_word_len - len(w)) for w in c]
            
            if not all(len(x) == c_word_len for x in c):
                print map(len, c)
            
            questions_char.append([q])
            contexts_char.append([c])
            
        if not all(len(x[0]) == question_len for x in questions_char):
            print [len(x[0]) for x in questions_char]
                
        if not all(len(x[0]) == context_len for x in contexts_char):
            print [len(x[0]) for x in contexts_char]
            
        questions_char = np.vstack(questions_char).astype(np.int32)
        contexts_char = np.vstack(contexts_char).astype(np.int32)      
        
        question_char_mask = (questions_char != pad).astype(np.float32)
        context_char_mask = (contexts_char != pad).astype(np.float32)
        
        if with_answer_inds:
            yield questions, contexts, questions_char, contexts_char, bin_feats, question_mask, context_mask, \
                    question_char_mask, context_char_mask, answer_inds
        else:
            yield questions, contexts, questions_char, contexts_char, bin_feats, question_mask, context_mask, \
                    question_char_mask, context_char_mask
