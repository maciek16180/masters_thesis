# similar to FastQA, https://arxiv.org/abs/1703.04816
# vectors can be stored in RAM instead of GPU memory, works with GloVe 840B

import numpy as np
import theano, os, io
import theano.tensor as T
import time

import lasagne as L
import lasagne.layers as LL

from layers import WeightedFeatureLayer
from layers import BatchedDotLayer
from layers import MaskedSoftmaxLayer
from layers import StartFeaturesLayer
from layers import EndFeaturesLayer
from layers import HighwayLayer
from layers import ForgetSizeLayer
from layers import TrainPartOfEmbsLayer


class QANet:

    def __init__(self, voc_size, alphabet_size, emb_size, emb_char_size, num_emb_char_filters, rec_size,
                 train_inds, emb_init, squad_path='/pio/data/data/squad/',
                 working_path='evaluate/glove6B/training/', dev_path='/pio/data/data/squad/glove6B/',
                 checkpoint_examples=64000, prefetch_word_embs=False, **kwargs):

        self.data_dev = None
        self.data_dev_num = None
        self.squad_path = squad_path
        self.working_path = working_path
        self.dev_path = dev_path

        self.checkpoint_examples = checkpoint_examples
        self.examples_since_last_checkpoint = 0
        self.dev_f1_log = []

        self.voc_size = voc_size
        self.emb_size = emb_size
        self.rec_size = rec_size

        self.train_inds = train_inds
        self.alphabet_size = alphabet_size
        self.emb_char_size = emb_char_size
        self.num_emb_char_filters = num_emb_char_filters

        self.prefetch_word_embs = prefetch_word_embs
        self.word_embeddings = emb_init

        if not self.prefetch_word_embs:
            self.context_var = T.imatrix('contexts')
            self.question_var = T.imatrix('questions')
        else:
            self.context_var = T.tensor3('contexts')
            self.question_var = T.tensor3('questions')

        self.mask_context_var = T.matrix('context_mask')
        self.mask_question_var = T.matrix('question_mask')

        self.context_char_var = T.itensor3('context_char')
        self.mask_context_char_var = T.tensor3('context_char_mask')

        self.question_char_var = T.itensor3('question_char')
        self.mask_question_char_var = T.tensor3('question_char_mask')

        self.bin_feat_var = T.matrix('bin_feat')
        self.answer_starts_var = T.ivector('answer_starts')
        self.answer_ends_var = T.ivector('answer_ends')

        self.aux1_var = T.matrix('aux1')
        self.aux2_var = T.matrix('aux2')

        # BUILD THE MODEL
        print 'Building the model...'

        self.intermediate_net = self._build_net(**kwargs)
        self.train_net = self._build_predictors(*self.intermediate_net)

        all_params = {x.name : x for x in L.layers.get_all_params(self.train_net)}
        self.pred_net = self._build_predictors_from_intermediate_results(all_params)

        # CALCULATE THE LOSS

        train_out = LL.get_output(self.train_net)
        test_out = LL.get_output(self.pred_net, deterministic=True)

        start_probs = train_out[0][T.arange(self.context_var.shape[0]), self.answer_starts_var]
        end_conditional_probs = train_out[1][T.arange(self.context_var.shape[0]), self.answer_ends_var]
        span_probs = start_probs * end_conditional_probs

        train_loss = -T.log(span_probs).mean()

        test_start_probs = test_out[0][T.arange(self.context_var.shape[0]), self.answer_starts_var]
        test_end_conditional_probs = test_out[1][T.arange(self.context_var.shape[0]), self.answer_ends_var]
        test_span_probs = test_start_probs * test_end_conditional_probs
        test_loss = -T.log(test_span_probs).mean()

        # CREATE UPDATE RULES

        params = LL.get_all_params(self.train_net, trainable=True)

        learning_rate_var = T.scalar('learning_rate')
        self.learning_rate = .001
        update_fn = lambda l, p: L.updates.adam(l, p, learning_rate=learning_rate_var)

        updates = update_fn(train_loss, params)

        # MAKE THEANO FUNCTIONS
        print 'Compiling theano functions:'

        compile_train_fn = not kwargs.get('skip_train_fn', False)
        if compile_train_fn:
            print '    train_fn...'
            self.train_fn = theano.function(
                [self.context_var, self.question_var, self.context_char_var, self.question_char_var, self.bin_feat_var,
                 self.mask_context_var, self.mask_question_var, self.mask_context_char_var, self.mask_question_char_var,
                 self.answer_starts_var, self.answer_ends_var, learning_rate_var],
                train_loss, updates=updates)

        print '    get_intermediate_results_fn...'
        self.get_intermediate_results_fn = theano.function(
            [self.context_var, self.question_var, self.context_char_var, self.question_char_var, self.bin_feat_var,
             self.mask_context_var, self.mask_question_var, self.mask_context_char_var, self.mask_question_char_var],
            LL.get_output(self.intermediate_net))

        print '    get_start_probs_fn...'
        self.get_start_probs_fn = theano.function(
            [self.context_var, self.mask_context_var, self.aux1_var, self.aux2_var],
            test_out[0])

        print '    get_end_probs_fn...'
        self.get_end_probs_fn = theano.function(
            [self.context_var, self.mask_context_var, self.aux1_var, self.aux2_var, self.answer_starts_var],
            test_out[1])

        # print '    get_start_probs_fn...'
        # self.get_start_probs_fn = theano.function(
        #     [self.context_var, self.question_var, self.context_char_var, self.question_char_var,
        #      self.bin_feat_var, self.mask_context_var, self.mask_question_var, self.mask_context_char_var,
        #      self.mask_question_char_var],
        #     test_out[0])

        # print '    get_end_probs_fn...'
        # self.get_end_probs_fn = theano.function(
        #     [self.context_var, self.question_var, self.context_char_var, self.question_char_var,
        #      self.bin_feat_var, self.mask_context_var, self.mask_question_var, self.mask_context_char_var,
        #      self.mask_question_char_var, self.answer_starts_var],
        #     test_out[1])


        print 'Done'


    def get_start_probs(self, data, batch_size, premade_bin_feats=False):
        result = []
        intermediate_results = []
        for batch in self.iterate_minibatches(data, batch_size, with_answer_inds=False, premade_bin_feats=premade_bin_feats):
            questions, contexts, questions_char, contexts_char, bin_feats, \
                question_mask, context_mask, question_char_mask, context_char_mask = batch
            inter_res = self.get_intermediate_results_fn(
                contexts, questions, contexts_char, questions_char, bin_feats,
                context_mask, question_mask, context_char_mask, question_char_mask)
            # inter_res = [inter_res[0].astype(theano.config.floatX), inter_res[1].astype(theano.config.floatX)]
            out = self.get_start_probs_fn(contexts, context_mask, inter_res[0], inter_res[1])
            # out = self.get_start_probs_fn(contexts, questions, contexts_char, questions_char, bin_feats,
            #                               context_mask, question_mask, context_char_mask, question_char_mask)
            intermediate_results.append(inter_res)
            result.append(out)

        return np.vstack(result), intermediate_results

    '''
    tu jest blad! rozne batche maja rozna dlugosc, trzeba to zrobic inaczej (albo wsadzac jednobatchowe porcje danych)
    (batchowanie na zewnatrz)
    '''

    # current implementation only allows for generating ends for one start index per example
    # to get end probs for k candidates for the start index, get_end_probs needs to be run k times
    def get_end_probs(self, data, answer_start_inds, batch_size, intermediate_results, premade_bin_feats=False):
        result = []
        idx = 0
        for i, batch in enumerate(self.iterate_minibatches(data, batch_size, with_answer_inds=False, premade_bin_feats=premade_bin_feats)):
            questions, contexts, questions_char, contexts_char, bin_feats, \
                question_mask, context_mask, question_char_mask, context_char_mask = batch
            inter_res = intermediate_results[i]
            start_inds = answer_start_inds[idx:idx + batch_size]
            out = self.get_end_probs_fn(contexts, context_mask, inter_res[0], inter_res[1], start_inds)
            # out = self.get_end_probs_fn(contexts, questions, contexts_char, questions_char, bin_feats,
            #                             context_mask, question_mask, context_char_mask, question_char_mask, start_inds)
            result.append(out)
            idx += batch_size

        return np.vstack(result)


    def train_one_epoch(self, train_data, batch_size, log_interval=10, premade_bin_feats=True):
        train_err = 0.
        train_batches = 0
        start_time = time.time()

        for batch in self.iterate_minibatches(train_data, batch_size, shuffle=True, premade_bin_feats=premade_bin_feats):
            questions, contexts, questions_char, contexts_char, bin_feats, \
                question_mask, context_mask, question_char_mask, context_char_mask, answer_inds = batch

            train_err += self.train_fn(contexts, questions, contexts_char, questions_char, bin_feats,
                                       context_mask, question_mask, context_char_mask, question_char_mask,
                                       answer_inds[:,0], answer_inds[:,1], self.learning_rate)
            train_batches += 1
            self.examples_since_last_checkpoint += batch_size

            if not train_batches % log_interval:
                print "Done {} batches in {:.2f}s\ttraining loss:\t{:.6f}".format(
                    train_batches, time.time() - start_time, train_err / train_batches)

            if self.examples_since_last_checkpoint > self.checkpoint_examples:
                checkpoint = len(self.dev_f1_log) + 1
                self.dev_f1_log.append(self._calc_dev_f1(checkpoint))
                if checkpoint > 1 and self.dev_f1_log[-1] < self.dev_f1_log[-2]:
                    print 'Lowering learning rate to ', self.learning_rate / 2
                    self.learning_rate /= 2
                self.examples_since_last_checkpoint = 0

                if len(self.dev_f1_log) > 10 and self.dev_f1_log[-11] > max(self.dev_f1_log[-10:]):
                    print '##################################'
                    print 'No improvement for 10 checkpoints!'
                    print '##################################'

        return train_err / train_batches


    def save_params(self, fname): # without the fixed word embeddings matrix
        params = L.layers.get_all_param_values(self.train_net)
        if not self.prefetch_word_embs:
            params = params[1:]
        np.savez(fname, *params)


    def load_params(self, fname, E=None): # E is the fixed word embeddings matrix
        assert self.prefetch_word_embs or E is not None
        with np.load(fname) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            if not self.prefetch_word_embs:
                param_values.insert(0, E)
            for i in range(len(param_values)):
                if param_values[i].dtype == 'float64':
                    param_values[i] = param_values[i].astype(theano.config.floatX)
            L.layers.set_all_param_values(self.train_net, param_values)


    # WARNING: I don't know if beam > 1 works anymore (I don't use it anyway)
    def _predict_spans(self, data, beam=1, batch_size=10):
        num_examples = len(data[0])

        start_probs, intermediate_results = self.get_start_probs(data, batch_size, premade_bin_feats=True)
        best_starts = start_probs.argpartition(-beam, axis=1)[:, -beam:].astype(np.int32)

        scores = start_probs[np.arange(num_examples)[:, np.newaxis], best_starts]
        scores = np.tile(scores[:, np.newaxis], (beam, 1)).transpose(0, 2, 1)

        best_ends_all = []
        for i in xrange(beam):
            end_probs = self.get_end_probs(data, best_starts[:, i], batch_size, intermediate_results, premade_bin_feats=True)
            best_ends = end_probs.argpartition(-beam, axis=1)[:, -beam:]
            scores[:, i, :] *= end_probs[np.arange(num_examples)[:, np.newaxis], best_ends]
            best_ends_all.append(best_ends)

        best_ends_all = np.hstack(best_ends_all)

        scores = scores.reshape(num_examples, beam**2)
        best_spans = scores.argmax(axis=1)
        starts = [i / beam for i in best_spans]

        starts = best_starts[np.arange(num_examples), starts]
        ends = best_ends_all[np.arange(num_examples), best_spans]

        return starts, ends


    def _calc_dev_f1(self, checkpoint, batch_size=10):
        if self.data_dev is None:
            self.data_dev = np.load(self.dev_path + 'dev.pkl')

            dev = np.load(self.dev_path + 'dev_words.pkl')
            dev_char = np.load(self.dev_path + 'dev_char_ascii.pkl')
            dev_bin_feats = np.load(self.dev_path + 'dev_bin_feats.pkl')
            self.data_dev_num = dev, dev_char, dev_bin_feats

        predicted_spans = []

        print "Calculating validation f1..."
        idx = 0
        while idx < len(self.data_dev):
            data_dev_batch = [self.data_dev_num[0][idx:idx + batch_size],
                              self.data_dev_num[1][idx:idx + batch_size],
                              self.data_dev_num[2][idx:idx + batch_size]]
            spans = self._predict_spans(data_dev_batch, beam=1)
            predicted_spans.append(np.vstack(spans))
            idx += batch_size
            if not idx % 1000:
                print 'Done %i examples' % idx

        print 'Predictions done'

        predicted_spans = np.hstack(predicted_spans).T

        path = self.working_path + 'pred_checkpoint%i.txt' % checkpoint

        with io.open(path, 'w', encoding='utf-8') as f:
            f.write(u'{')
            for i in xrange(len(self.data_dev)):
                ans = ' '.join(self.data_dev[i][2][predicted_spans[i][0]:predicted_spans[i][1] + 1])
                Id = self.data_dev[i][3]
                f.write(u'"{}": "{}"'.format(Id, ans))
                if i < len(self.data_dev) - 1:
                    f.write(u', ')
            f.write(u'}')

        res = os.system('python ' + self.squad_path + 'evaluate-v1.1.py ' + self.squad_path + 'dev-v1.1.json ' + path)

        f1 = np.load(path + '.pkl')['f1']
        print "F1: ", f1
        return f1


    def _build_net(self, emb_char_filter_size=5, emb_char_init=None, train_emb_char=True, emb_dropout=False, **kwargs):

        batch_size = self.context_var.shape[0]
        context_len = self.context_var.shape[1]
        question_len = self.question_var.shape[1]
        context_word_len = self.context_char_var.shape[2]
        question_word_len = self.question_char_var.shape[2]

        self.batch_size = batch_size
        self.context_len = context_len

        ''' Inputs '''

        l_context_char = LL.InputLayer(shape=(None, None, None), input_var=self.context_char_var)
        l_question_char = LL.InputLayer(shape=(None, None, None), input_var=self.question_char_var)

        l_c_mask = None
        if self.mask_context_var is not None:
            l_c_mask = LL.InputLayer(shape=(None, None), input_var=self.mask_context_var)

        l_q_mask = None
        if self.mask_question_var is not None:
            l_q_mask = LL.InputLayer(shape=(None, None), input_var=self.mask_question_var)

        l_c_char_mask = LL.InputLayer(shape=(None, None, None), input_var=self.mask_context_char_var)
        l_q_char_mask = LL.InputLayer(shape=(None, None, None), input_var=self.mask_question_char_var)

        ''' Word embeddings '''

        # word_keep_rate = .5 if emb_dropout else 1
        # if emb_dropout:
        #     print "Using word dropout with keep rate", word_keep_rate

        if not self.prefetch_word_embs:
            l_context = LL.InputLayer(shape=(None, None), input_var=self.context_var)
            l_question = LL.InputLayer(shape=(None, None), input_var=self.question_var)

            l_c_emb = TrainPartOfEmbsLayer(l_context,
                                           output_size=self.emb_size,
                                           input_size=self.voc_size,
                                           W=self.word_embeddings[self.train_inds],
                                           E=self.word_embeddings,
                                           train_inds=self.train_inds)

            l_q_emb = TrainPartOfEmbsLayer(l_question,
                                           output_size=self.emb_size,
                                           input_size=self.voc_size,
                                           W=l_c_emb.W,
                                           E=l_c_emb.E,
                                           train_inds=self.train_inds)
        else:
            l_c_emb = LL.InputLayer(shape=(None, None, self.emb_size), input_var=self.context_var)
            l_q_emb = LL.InputLayer(shape=(None, None, self.emb_size), input_var=self.question_var)

        ''' Char-embeddings '''

        if emb_char_init is None:
            l_c_char_emb = LL.EmbeddingLayer(l_context_char,
                                             input_size=self.alphabet_size,
                                             output_size=self.emb_char_size)
        else:
            l_c_char_emb = LL.EmbeddingLayer(l_context_char,
                                             input_size=self.alphabet_size,
                                             output_size=self.emb_char_size,
                                             W=emb_char_init)
        # l_c_char_emb.shape is (batch_size x context_len x context_word_len x emb_char_size)

        l_q_char_emb = LL.EmbeddingLayer(l_question_char,
                                         input_size=self.alphabet_size,
                                         output_size=self.emb_char_size,
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

        l_c_char_emb = LL.dimshuffle(LL.reshape(l_c_char_emb, (batch_size * context_len, context_word_len, self.emb_char_size)),
                                     (0, 2, 1))
        l_c_char_conv = LL.Conv1DLayer(l_c_char_emb,
                                       num_filters=self.num_emb_char_filters,
                                       filter_size=emb_char_filter_size,
                                       nonlinearity=L.nonlinearities.tanh)
        # (batch_size * context_len x num_filters x context_word_len - filter_size + 1)

        l_c_char_emb = LL.ExpressionLayer(l_c_char_conv, lambda X: X.max(2), output_shape='auto')
        l_c_char_emb = LL.reshape(l_c_char_emb, (batch_size, context_len, self.num_emb_char_filters))
        l_c_emb = LL.concat([l_c_emb, l_c_char_emb], axis=2)


        l_q_char_emb = LL.dimshuffle(LL.reshape(l_q_char_emb, (batch_size * question_len, question_word_len, self.emb_char_size)),
                                     (0, 2, 1))
        l_q_char_conv = LL.Conv1DLayer(l_q_char_emb,
                                       num_filters=self.num_emb_char_filters,
                                       filter_size=emb_char_filter_size,
                                       nonlinearity=L.nonlinearities.tanh,
                                       W=l_c_char_conv.W,
                                       b=l_c_char_conv.b)
        # (batch_size * question_len x num_filters x question_word_len - filter_size + 1)

        l_q_char_emb = LL.ExpressionLayer(l_q_char_conv, lambda X: X.max(2), output_shape='auto')
        l_q_char_emb = LL.reshape(l_q_char_emb, (batch_size, question_len, self.num_emb_char_filters))
        l_q_emb = LL.concat([l_q_emb, l_q_char_emb], axis=2)

        ''' Dropout at the embeddings '''

        if emb_dropout:
            print 'Using dropout.'
            l_c_emb = LL.dropout(l_c_emb)
            l_q_emb = LL.dropout(l_q_emb)

        ''' Highway layer allowing for interaction between embeddings '''

        l_c_P = LL.DenseLayer(LL.reshape(l_c_emb, (batch_size * context_len, self.emb_size + self.num_emb_char_filters)),
                              num_units=self.rec_size,
                              b=None,
                              nonlinearity=None)

        l_c_high = HighwayLayer(l_c_P)
        l_c_emb = LL.reshape(l_c_high, (batch_size, context_len, self.rec_size))

        l_q_P = LL.DenseLayer(LL.reshape(l_q_emb, (batch_size * question_len, self.emb_size + self.num_emb_char_filters)),
                              num_units=self.rec_size,
                              W=l_c_P.W,
                              b=None,
                              nonlinearity=None)

        l_q_high = HighwayLayer(l_q_P,
                                W1=l_c_high.W1,
                                b1=l_c_high.b1,
                                W2=l_c_high.W2,
                                b2=l_c_high.b2)
        l_q_emb = LL.reshape(l_q_high, (batch_size, question_len, self.rec_size))

        ''' Here we calculate wiq features from https://arxiv.org/abs/1703.04816 '''

        l_feat = WeightedFeatureLayer([l_c_emb, l_q_emb, l_c_mask])
        l_weighted_feat = LL.dimshuffle(l_feat, (0, 1, 'x'))

        l_bin_feat = LL.InputLayer(shape=(None, None), input_var=self.bin_feat_var)
        l_bin_feat = LL.dimshuffle(l_bin_feat, (0, 1, 'x'))

        l_c_emb = LL.concat([l_c_emb, l_bin_feat, l_weighted_feat], axis=2) # both features are concatenated to the embeddings
        l_q_emb = LL.pad(l_q_emb, width=[(0, 2)], val=L.utils.floatX(1), batch_ndim=2) # for the question we fix the features to 1

        ''' Context and question encoding using the same BiLSTM for both '''

        l_c_enc_forw = LL.LSTMLayer(l_c_emb, # output shape is (batch_size x context_len x rec_size)
                                    num_units=self.rec_size,
                                    grad_clipping=100,
                                    mask_input=l_c_mask)

        l_c_enc_back = LL.LSTMLayer(l_c_emb,
                                    num_units=self.rec_size,
                                    grad_clipping=100,
                                    mask_input=l_c_mask,
                                    backwards=True)

        l_q_enc_forw = LL.LSTMLayer(l_q_emb, # output shape is (batch_size x question_len x rec_size)
                                    num_units=self.rec_size,
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
                                    num_units=self.rec_size,
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
        l_c_proj = LL.DenseLayer(LL.reshape(l_c_enc, (-1, 2 * self.rec_size)), # batch_size * context_len x rec_size
                                 num_units=self.rec_size,
                                 W=np.vstack([np.eye(self.rec_size, dtype=theano.config.floatX),
                                              np.eye(self.rec_size, dtype=theano.config.floatX)]),
                                 b=None,
                                 nonlinearity=L.nonlinearities.tanh)

        # this is Z from the paper
        l_q_proj = LL.DenseLayer(LL.reshape(l_q_enc, (-1, 2 * self.rec_size)), # batch_size * question_len x rec_size
                                 num_units=self.rec_size,
                                 W=np.vstack([np.eye(self.rec_size, dtype=theano.config.floatX),
                                              np.eye(self.rec_size, dtype=theano.config.floatX)]),
                                 b=None,
                                 nonlinearity=L.nonlinearities.tanh)

        ''' Additional, weighted question encoding (alphas from https://arxiv.org/abs/1703.04816) '''

        l_alpha = LL.DenseLayer(l_q_proj, # batch_size * question_len x 1
                                num_units=1,
                                b=None,
                                nonlinearity=None)

        l_alpha = MaskedSoftmaxLayer([LL.reshape(l_alpha, (batch_size, -1)), l_q_mask])

        l_z_hat = BatchedDotLayer([LL.reshape(l_q_proj, (batch_size, -1, self.rec_size)), l_alpha]) # batch_size x rec_size

        return l_c_proj, l_z_hat


    def _build_predictors(self, l_c_proj, l_z_hat):

        l_c_mask = None
        if self.mask_context_var is not None:
            l_c_mask = LL.InputLayer(shape=(None, None), input_var=self.mask_context_var)

        ''' Answer span prediction '''

        # span start

        l_start_feat = StartFeaturesLayer([LL.reshape(l_c_proj, (self.batch_size, self.context_len, self.rec_size)), l_z_hat])

        l_start = LL.DenseLayer(LL.reshape(l_start_feat, (self.batch_size * self.context_len, 3 * self.rec_size)),
                                num_units=self.rec_size,
                                nonlinearity=L.nonlinearities.rectify,
                                name='start_dense') # batch_size * context_len x rec_size

        l_Vs = LL.DenseLayer(l_start, # batch_size * context_len x 1
                             num_units=1,
                             b=None,
                             nonlinearity=None,
                             name='Vs')

        # this is p_s from the paper
        l_start_soft = MaskedSoftmaxLayer([LL.reshape(l_Vs, (self.batch_size, self.context_len)), l_c_mask]) # batch_size x context_len

        # span end

        l_answer_starts = LL.InputLayer(shape=(None,), input_var=self.answer_starts_var)

        l_end_feat = EndFeaturesLayer([LL.reshape(l_c_proj, (self.batch_size, self.context_len, self.rec_size)), l_z_hat, l_answer_starts])

        l_end = LL.DenseLayer(LL.reshape(l_end_feat, (self.batch_size * self.context_len, 5 * self.rec_size)),
                              num_units=self.rec_size,
                              nonlinearity=L.nonlinearities.rectify,
                              name='end_dense') # batch_size * context_len x self.rec_size

        l_Ve = LL.DenseLayer(l_end, # batch_size * context_len x 1
                             num_units=1,
                             b=None,
                             nonlinearity=None,
                             name='Ve')

        # this is p_e from the paper
        l_end_soft = MaskedSoftmaxLayer([LL.reshape(l_Ve, (self.batch_size, self.context_len)), l_c_mask]) # batch_size x context_len

        return l_start_soft, l_end_soft


    def _build_predictors_from_intermediate_results(self, params):

        l_c_mask = None
        if self.mask_context_var is not None:
            l_c_mask = LL.InputLayer(shape=(None, None), input_var=self.mask_context_var)

        l_c_proj = LL.InputLayer(shape=(None, self.rec_size), input_var=self.aux1_var)
        l_z_hat = LL.InputLayer(shape=(None, self.rec_size), input_var=self.aux2_var)

        ''' Answer span prediction from itermediate results '''

        # span start

        l_start_feat = StartFeaturesLayer([LL.reshape(l_c_proj, (self.batch_size, self.context_len, self.rec_size)), l_z_hat])

        l_start = LL.DenseLayer(LL.reshape(l_start_feat, (self.batch_size * self.context_len, 3 * self.rec_size)),
                                num_units=self.rec_size,
                                W=params['start_dense.W'],
                                b=params['start_dense.b'],
                                nonlinearity=L.nonlinearities.rectify) # batch_size * context_len x rec_size

        l_Vs = LL.DenseLayer(l_start, # batch_size * context_len x 1
                             num_units=1,
                             W=params['Vs.W'],
                             b=None,
                             nonlinearity=None)

        # this is p_s from the paper
        l_start_soft = MaskedSoftmaxLayer([LL.reshape(l_Vs, (self.batch_size, self.context_len)), l_c_mask]) # batch_size x context_len

        # span end

        l_answer_starts = LL.InputLayer(shape=(None,), input_var=self.answer_starts_var)

        l_end_feat = EndFeaturesLayer([LL.reshape(l_c_proj, (self.batch_size, self.context_len, self.rec_size)), l_z_hat, l_answer_starts])

        l_end = LL.DenseLayer(LL.reshape(l_end_feat, (self.batch_size * self.context_len, 5 * self.rec_size)),
                              num_units=self.rec_size,
                              W=params['end_dense.W'],
                              b=params['end_dense.b'],
                              nonlinearity=L.nonlinearities.rectify) # batch_size * context_len x self.rec_size

        l_Ve = LL.DenseLayer(l_end, # batch_size * context_len x 1
                             num_units=1,
                             W=params['Ve.W'],
                             b=None,
                             nonlinearity=None)

        # this is p_e from the paper
        l_end_soft = MaskedSoftmaxLayer([LL.reshape(l_Ve, (self.batch_size, self.context_len)), l_c_mask]) # batch_size x context_len

        return l_start_soft, l_end_soft


    def iterate_minibatches(self, inputs, batch_size, pad=-1, with_answer_inds=True, shuffle=False, premade_bin_feats=False):

        assert not premade_bin_feats or len(inputs) == 3

        if not premade_bin_feats:
            inputs, inputs_char = inputs
        else:
            inputs, inputs_char, all_bin_feats = inputs

        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
            inputs = np.array(inputs)
            inputs_char = np.array(inputs_char)
            if premade_bin_feats:
                all_bin_feats = np.array(all_bin_feats)

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

            if premade_bin_feats:
                bin_feats_premade = all_bin_feats[excerpt]

            if with_answer_inds:
                answer_inds = []

            for l, (ans, q, c) in enumerate(examples):
                q_words = set(q)
                q = q + [pad] * (question_len - len(q))

                if not premade_bin_feats:
                    bin_feat = [w in q_words for w in c]
                else:
                    bin_feat = bin_feats_premade[l]

                bin_feat = bin_feat + [pad] * (context_len - len(bin_feat))
                c = c + [pad] * (context_len - len(c))

                if with_answer_inds:
                    answer_inds.append((min(ans[0]), max(ans[0])))

                questions.append(q)
                contexts.append(c)
                bin_feats.append(bin_feat)

            questions = np.vstack(questions).astype(np.int32)
            contexts  = np.vstack(contexts).astype(np.int32)
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

            if self.prefetch_word_embs:
                questions = self.word_embeddings[questions]
                contexts = self.word_embeddings[contexts]

            res = questions, contexts, questions_char, contexts_char, bin_feats, question_mask, context_mask, \
                    question_char_mask, context_char_mask

            if with_answer_inds:
                res = res + (answer_inds,)

            yield res
