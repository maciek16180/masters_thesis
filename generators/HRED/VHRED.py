from __future__ import print_function

# <unk> token is assumed to be at index 0!

import numpy as np
import theano
import theano.tensor as T
import time

import lasagne as L
import lasagne.layers as LL

import sys
sys.path.append('../SimpleRNNLM/')
sys.path.append('../')

from SimpleRNNLM import SimpleRNNLM

from layers import SampledSoftmaxDenseLayer
from layers import ShiftLayer
from layers import L2PoolingLayer
from layers import MultNormKLDivLayer
from layers import GaussianSampleLayer
from layers import WordDropoutLayer


class VHRED(SimpleRNNLM):

    def __init__(self, voc_size, emb_size, lv1_rec_size, lv2_rec_size,
                 out_emb_size, latent_size, mode='ssoft', n=3, pad_value=-1,
                 **kwargs):

        self.pad_value = pad_value
        self.voc_size = voc_size
        self.emb_size = emb_size
        self.lv1_rec_size = lv1_rec_size
        self.lv2_rec_size = lv2_rec_size
        self.out_emb_size = out_emb_size
        self.latent_size = latent_size

        self.input_var = T.imatrix('inputs')
        self.target_var = T.imatrix('targets')  # inputs shifted by 1
        self.mask_input_var = T.matrix('input_mask')
        mask_idx = self.mask_input_var.nonzero()

        self.emb_init = kwargs.get('emb_init', None)

        context_init = T.matrix('context_init')
        decoder_init = T.matrix('decoder_init')

        # BUILD THE MODEL
        print('Building the model...')

        assert mode in ['ssoft']

        self.kl_annealing = kwargs.get('kl_annealing', True)
        self.kl_annealing_max_examples = kwargs.get(
            'kl_annealing_max_examples', 500000)

        self.train_net = self._build_vhred(
            target_var=self.target_var, n=n, **kwargs)

        # CALCULATE THE LOSS

        train_out, train_kl = LL.get_output(self.train_net)
        test_out, test_kl = LL.get_output(self.train_net, deterministic=True)

        train_loss = (-T.log(train_out[mask_idx]).sum() + train_kl.sum()
                      ) / mask_idx[0].size
        test_loss = (-T.log(test_out[mask_idx]).sum() + test_kl.sum()
                     ) / mask_idx[0].size

        # MAKE TRAIN AND VALIDATION FUNCTIONS
        print('Compiling theano functions...')

        params = LL.get_all_params(self.train_net, trainable=True)

        update_fn = kwargs.get('update_fn', lambda l, p: L.updates.adam(
            l, p, learning_rate=.0001))
        updates = update_fn(train_loss, params)

        self.train_fn = theano.function(
            [self.input_var, self.target_var, self.mask_input_var],
            train_loss, updates=updates)
        self.val_fn = theano.function(
            [self.input_var, self.target_var, self.mask_input_var], test_loss)

        # BUILD NET FOR GENERATING, WITH SHARED PARAMETERS
        print('Building a network for generating...')

        all_params = {x.name: x for x in LL.get_all_params(self.train_net)}

        self.context_net = self._build_context_net_with_params(
            context_init, all_params)
        self.decoder_net = self._build_decoder_net_with_params(
            decoder_init, all_params)

        dec_net_out = LL.get_output(self.decoder_net, deterministic=True)
        new_con_init, z_prior = LL.get_output(self.context_net)

        self.get_probs_and_new_dec_init_fn = theano.function(
            [self.input_var, decoder_init], dec_net_out)
        self.get_new_con_init_fn = theano.function(
            [self.input_var, context_init], [new_con_init, z_prior])

        print('Done')

    def train_one_epoch(self, train_data, batch_size, log_interval=10):
        train_err = 0.
        train_batches = 0
        num_training_words = 0
        start_time = time.time()

        for batch in self.iterate_minibatches(train_data, batch_size,
                                              self.pad_value):
            inputs, targets, mask = batch

            num_batch_words = mask.sum()
            train_err += self.train_fn(inputs, targets, mask) * num_batch_words
            train_batches += 1
            num_training_words += num_batch_words

            if not train_batches % log_interval:
                print("Done {} batches in {:.2f}s\ttraining loss:\t{:.6f}".
                      format(train_batches, time.time() - start_time,
                             train_err / num_training_words))

            if self.kl_annealing:
                kl_annealing_param = [
                    x for x in LL.get_all_params(self.train_net)
                    if x.name == 'kl_annealing.scales'][0]
                val = np.array(kl_annealing_param.eval())
                if val < 1:
                    new_val = val + 1. / int(
                        self.kl_annealing_max_examples / batch_size)
                    new_val = np.array(new_val, dtype=theano.config.floatX)
                    new_val = min(
                        new_val, np.array(1, dtype=theano.config.floatX))
                    kl_annealing_param.set_value(new_val)

        return train_err / num_training_words

    def validate(self, val_data, batch_size):
        val_err = 0.
        val_batches = 0
        num_validate_words = 0
        start_time = time.time()

        for batch in self.iterate_minibatches(val_data, batch_size,
                                              self.pad_value):
            inputs, targets, mask = batch

            num_batch_words = mask.sum()
            val_err += self.val_fn(inputs, targets, mask) * num_batch_words
            val_batches += 1
            num_validate_words += num_batch_words

            if not val_batches % 100:
                print("Done {} batches in {:.2f}s".format(
                    val_batches, time.time() - start_time))

        return val_err / num_validate_words

    def load_params_from_HRED(self, hred_model_path, use_hred_emb=True):
        with np.load(hred_model_path) as f:
            param_values = [f['arr_%d' % i] for i in range(31)]
            params = LL.get_all_params(self.train_net)
            for i in range(1, 31):
                params[i].set_value(param_values[i])
            if use_hred_emb:
                params[0].set_value(param_values[0])

    '''
    MODEL PARAMETERS (as in LL.get_params(train_net))

     index - description

         0 - emb
      1:10 - GRU forward lv1
     11:20 - GRU backward lv1
     21:30 - GRU session

     31:32 - H1 posterior
     33:34 - H2 posterior
     35:36 - mu posterior
     37:38 - sigma posterior
        39 - scale posterior

     40:41 - dec init
     42:50 - GRU dec (without hid_init)
     51:52 - H0
        53 - E0
     54:56 - sampled softmax (p is unnecessary for generating)

     57:58 - H1 prior
     59:60 - H2 prior
     61:62 - mu prior
     63:64 - sigma prior
        65 - scale prior
        66 - kl_annealing

     context_net: emb, GRUs lv1, GRU ses (no hid_init), priors
     decoder_net: emb, GRU dec (no hid_init), H0, E0, softmax (full, no p from ssoft)
    '''

    # "n" argument below is the maximum number of utterances in the sequence (so n=3 for Movie Triples).
    # It doesn't affect number or shape of net parameters, so we can, for example, pretrain the net on short contexts
    # and then feed it with longer ones. We have to rebuild the net with different n for that though.
    # Because i use 2D input, batch_size has to be divisible by n.

    # TODO: make it so we don't have to rebuild the net to feed in context with different n.
    #       the input shape will be (batch_size x n x sequence_len)

    def _build_vhred(self, num_sampled, ssoft_probs=None, train_emb=True, n=3,
                     kl_annealing=True, **kwargs):

        batch_size = self.input_var.shape[0]
        sequence_len = self.input_var.shape[1]

        ''' Inputs '''

        l_in = LL.InputLayer(shape=(None, None), input_var=self.input_var)

        l_mask = LL.InputLayer(shape=(None, None),
                               input_var=self.mask_input_var)

        ''' Word embeddings '''

        if self.emb_init is None:
            l_emb = LL.EmbeddingLayer(l_in,
                                      input_size=self.voc_size,
                                      output_size=self.emb_size,
                                      name='emb')
        else:
            l_emb = LL.EmbeddingLayer(l_in,
                                      input_size=self.voc_size,
                                      output_size=self.emb_size,
                                      W=self.emb_init,
                                      name='emb')
        if not train_emb:
            l_emb.params[l_emb.W].remove('trainable')

        ''' Level 1 (sentence) BiGRU encoding with L2-pooling '''

        # batch_size x lv1_rec_size
        l_lv1_enc_forw = LL.GRULayer(l_emb,
                                     num_units=self.lv1_rec_size,
                                     grad_clipping=100,
                                     mask_input=l_mask,
                                     name='GRU1forw')

        # backward pass of encoder rnn, (batch_size x lv1_rec_size)
        l_lv1_enc_back = LL.GRULayer(l_emb,
                                     num_units=self.lv1_rec_size,
                                     grad_clipping=100,
                                     mask_input=l_mask,
                                     backwards=True,
                                     name='GRU1back')

        l2_pooled_forw = L2PoolingLayer(l_lv1_enc_forw)
        l2_pooled_back = L2PoolingLayer(l_lv1_enc_back)

        # concatenation of L2-pooled states
        l_lv1_enc = LL.ConcatLayer([l2_pooled_forw, l2_pooled_back], axis=1)

        ''' Level 2 (context) encoding '''

        l_resh = LL.ReshapeLayer(l_lv1_enc, (
            batch_size / n, n, 2 * self.lv1_rec_size))

        # batch_size / n x n x lv2_rec_size
        l_lv2_enc = LL.GRULayer(l_resh,
                                num_units=self.lv2_rec_size,
                                grad_clipping=100,
                                name='GRU2')

        ''' Prior and posterior probs calculation '''

        # we want to use i-th utterance summary as an init for decoding (i+1)th
        l_shift = ShiftLayer(l_lv2_enc)
        l_resh_shifted = LL.ReshapeLayer(l_shift, (
            batch_size, self.lv2_rec_size))

        # prior

        l_dense1prior = LL.DenseLayer(l_resh_shifted,
                                      self.latent_size,
                                      nonlinearity=L.nonlinearities.tanh,
                                      W=L.init.Normal(),
                                      name='h1pr')
        l_dense2prior = LL.DenseLayer(l_dense1prior,
                                      self.latent_size,
                                      nonlinearity=L.nonlinearities.tanh,
                                      W=L.init.Normal(),
                                      name='h2pr')
        l_mu_prior = LL.DenseLayer(l_dense2prior,
                                   self.latent_size,
                                   nonlinearity=None,
                                   name='mupr')
        # Each row of l_sigma_prior is a 1D representation
        # of a diagonal covariance matrix.
        l_sigma_prior = LL.DenseLayer(l_dense2prior,
                                      self.latent_size,
                                      nonlinearity=L.nonlinearities.softplus,
                                      name='sigpr')
        l_sigma_prior = LL.ScaleLayer(
            l_sigma_prior, L.init.Constant(.1), name='scalepr')
        l_sigma_prior.params[l_sigma_prior.scales].remove('trainable')
        # mu and sigma shape is (batch_size x latent_size)

        # posterior

        l_h_post = LL.ConcatLayer([l_resh_shifted, l_lv1_enc], axis=1)

        l_dense1post = LL.DenseLayer(l_h_post,
                                     self.latent_size,
                                     nonlinearity=L.nonlinearities.tanh,
                                     W=L.init.Normal(),
                                     name='h1po')
        l_dense2post = LL.DenseLayer(l_dense1post,
                                     self.latent_size,
                                     nonlinearity=L.nonlinearities.tanh,
                                     W=L.init.Normal(),
                                     name='h2po')
        l_mu_post = LL.DenseLayer(l_dense2post,
                                  self.latent_size,
                                  nonlinearity=None,
                                  name='mupo')
        l_sigma_post = LL.DenseLayer(l_dense2post,
                                     self.latent_size,
                                     nonlinearity=L.nonlinearities.softplus,
                                     name='sigpo')
        l_sigma_post = LL.ScaleLayer(l_sigma_post,
                                     L.init.Constant(.1), name='scalepo')
        l_sigma_post.params[l_sigma_post.scales].remove('trainable')

        l_z = GaussianSampleLayer(l_mu_post, l_sigma_post)
        # sample from N(mu, sigma) (train: post, test: prior)
        # l_z size: batch_size x latent_size

        l_kl_div = MultNormKLDivLayer(
            [l_mu_post, l_sigma_post, l_mu_prior, l_sigma_prior])
        # KL annealing, as in https://arxiv.org/pdf/1511.06349.pdf
        # multiplier is slowly raising until it reaches 1 after specified number of samples
        # TODO: this as a custom layer, when deterministic=True no scaling is
        # applied
        if kl_annealing:
            l_kl_div = LL.ScaleLayer(l_kl_div,
                                     L.init.Constant(0), name='kl_annealing')
            l_kl_div.params[l_kl_div.scales].remove('trainable')

        ''' Decoder '''

        l_dec_init_base = LL.ConcatLayer([l_resh_shifted, l_z], axis=1)

        # batch_size x lv1_rec_size
        l_dec_inits = LL.DenseLayer(l_dec_init_base,
                                    num_units=self.lv1_rec_size,
                                    nonlinearity=L.nonlinearities.tanh,
                                    name='dec_init')

        # word dropout layer, as proposed in https://arxiv.org/pdf/1511.06349.pdf
        # The layer randomly chooses some fraction of embeddings and replaces
        # them with <unk>

        # <unk> is assumed to be at index 0
        l_unk_emb = LL.InputLayer(shape=(self.emb_size,), input_var=l_emb.W[0])
        l_drop = WordDropoutLayer(l_emb, l_unk_emb, drop_rate=.25)

        # batch_size x sequence_len x lv1_rec_size
        l_dec = LL.GRULayer(l_drop,
                            num_units=self.lv1_rec_size,
                            grad_clipping=100,
                            mask_input=l_mask,
                            hid_init=l_dec_inits,
                            name='GRUdec')

        l_resh3 = LL.ReshapeLayer(l_dec, (
            batch_size * sequence_len, self.lv1_rec_size))

        l_H0 = LL.DenseLayer(l_resh3,
                             num_units=self.out_emb_size,
                             nonlinearity=None,
                             name='h0')

        l_resh4 = LL.ReshapeLayer(l_emb, (
            batch_size * sequence_len, self.emb_size))

        l_E0 = LL.DenseLayer(l_resh4,
                             num_units=self.out_emb_size,
                             b=None,
                             nonlinearity=None,
                             name='e0')

        l_soft_in = LL.ElemwiseSumLayer([l_H0, l_E0])

        l_ssoft = SampledSoftmaxDenseLayer(l_soft_in,
                                           num_sampled,
                                           self.voc_size,
                                           targets=self.target_var.ravel(),
                                           probs=ssoft_probs,
                                           sample_unique=False,
                                           name='soft')

        l_out = LL.ReshapeLayer(l_ssoft, shape=(batch_size, sequence_len))

        return l_out, l_kl_div

    def _build_context_net_with_params(self, context_init, params):

        l_in = LL.InputLayer(shape=(1, None), input_var=self.input_var)

        l_emb = LL.EmbeddingLayer(l_in,
                                  input_size=self.voc_size,
                                  output_size=self.emb_size,
                                  W=params['emb.W'])

        l_lv1_enc_forw = LL.GRULayer(
            l_emb,
            num_units=self.lv1_rec_size,
            grad_clipping=100,
            resetgate=LL.Gate(
                W_in=params['GRU1forw.W_in_to_resetgate'],
                W_hid=params['GRU1forw.W_hid_to_resetgate'],
                W_cell=None,
                b=params['GRU1forw.b_resetgate']),
            updategate=LL.Gate(
                W_in=params['GRU1forw.W_in_to_updategate'],
                W_hid=params['GRU1forw.W_hid_to_updategate'],
                W_cell=None,
                b=params['GRU1forw.b_updategate']),
            hidden_update=LL.Gate(
                W_in=params['GRU1forw.W_in_to_hidden_update'],
                W_hid=params['GRU1forw.W_hid_to_hidden_update'],
                W_cell=None,
                b=params['GRU1forw.b_hidden_update'],
                nonlinearity=L.nonlinearities.tanh),
            hid_init=params['GRU1forw.hid_init'])

        # backward pass of encoder rnn
        l_lv1_enc_back = LL.GRULayer(
            l_emb,
            num_units=self.lv1_rec_size,
            grad_clipping=100,
            backwards=True,
            resetgate=LL.Gate(
                W_in=params['GRU1back.W_in_to_resetgate'],
                W_hid=params['GRU1back.W_hid_to_resetgate'],
                W_cell=None,
                b=params['GRU1back.b_resetgate']),
            updategate=LL.Gate(
                W_in=params['GRU1back.W_in_to_updategate'],
                W_hid=params['GRU1back.W_hid_to_updategate'],
                W_cell=None,
                b=params['GRU1back.b_updategate']),
            hidden_update=LL.Gate(
                W_in=params['GRU1back.W_in_to_hidden_update'],
                W_hid=params['GRU1back.W_hid_to_hidden_update'],
                W_cell=None,
                b=params['GRU1back.b_hidden_update'],
                nonlinearity=L.nonlinearities.tanh),
            hid_init=params['GRU1back.hid_init'])

        l2_pooled_forw = L2PoolingLayer(l_lv1_enc_forw)
        l2_pooled_back = L2PoolingLayer(l_lv1_enc_back)

        # concatenation of L2-pooled states
        l_lv1_enc = LL.ConcatLayer([l2_pooled_forw, l2_pooled_back])

        l_resh = LL.ReshapeLayer(l_lv1_enc, (1, 1, 2 * self.lv1_rec_size))

        l_lv2_enc = LL.GRULayer(
            l_resh,
            num_units=self.lv2_rec_size,
            hid_init=context_init,
            grad_clipping=100,
            only_return_final=True,
            resetgate=LL.Gate(
                W_in=params['GRU2.W_in_to_resetgate'],
                W_hid=params['GRU2.W_hid_to_resetgate'],
                W_cell=None,
                b=params['GRU2.b_resetgate']),
            updategate=LL.Gate(
                W_in=params['GRU2.W_in_to_updategate'],
                W_hid=params['GRU2.W_hid_to_updategate'],
                W_cell=None,
                b=params['GRU2.b_updategate']),
            hidden_update=LL.Gate(
                W_in=params['GRU2.W_in_to_hidden_update'],
                W_hid=params['GRU2.W_hid_to_hidden_update'],
                W_cell=None,
                b=params['GRU2.b_hidden_update'],
                nonlinearity=L.nonlinearities.tanh))

        l_dense1prior = LL.DenseLayer(l_lv2_enc,
                                      self.latent_size,
                                      nonlinearity=L.nonlinearities.tanh,
                                      W=params['h1pr.W'],
                                      b=params['h1pr.b'])
        l_dense2prior = LL.DenseLayer(l_dense1prior,
                                      self.latent_size,
                                      nonlinearity=L.nonlinearities.tanh,
                                      W=params['h2pr.W'],
                                      b=params['h2pr.b'])
        l_mu_prior = LL.DenseLayer(l_dense2prior,
                                   self.latent_size,
                                   nonlinearity=None,
                                   W=params['mupr.W'],
                                   b=params['mupr.b'])
        l_sigma_prior = LL.DenseLayer(l_dense2prior,
                                      self.latent_size,
                                      nonlinearity=L.nonlinearities.softplus,
                                      W=params['sigpr.W'],
                                      b=params['sigpr.b'])
        l_sigma_prior = LL.ScaleLayer(l_sigma_prior, L.init.Constant(.1))

        l_z = GaussianSampleLayer(l_mu_prior, l_sigma_prior)

        return l_lv2_enc, l_z

    def _build_decoder_net_with_params(self, decoder_init, params):

        l_in = LL.InputLayer(shape=(None, None), input_var=self.input_var)

        l_emb = LL.EmbeddingLayer(l_in,
                                  input_size=self.voc_size,
                                  output_size=self.emb_size,
                                  W=params['emb.W'])

        l_dec_init = LL.InputLayer(shape=(None, self.lv1_rec_size),
                                   input_var=decoder_init)

        l_dec = LL.GRULayer(
            l_emb,
            num_units=self.lv1_rec_size,
            grad_clipping=100,
            hid_init=l_dec_init,
            only_return_final=True,
            resetgate=LL.Gate(
                W_in=params['GRUdec.W_in_to_resetgate'],
                W_hid=params['GRUdec.W_hid_to_resetgate'],
                W_cell=None,
                b=params['GRUdec.b_resetgate']),
            updategate=LL.Gate(
                W_in=params['GRUdec.W_in_to_updategate'],
                W_hid=params['GRUdec.W_hid_to_updategate'],
                W_cell=None,
                b=params['GRUdec.b_updategate']),
            hidden_update=LL.Gate(
                W_in=params['GRUdec.W_in_to_hidden_update'],
                W_hid=params['GRUdec.W_hid_to_hidden_update'],
                W_cell=None,
                b=params['GRUdec.b_hidden_update'],
                nonlinearity=L.nonlinearities.tanh))

        l_H0 = LL.DenseLayer(l_dec,
                             num_units=self.out_emb_size,
                             nonlinearity=None,
                             W=params['h0.W'],
                             b=params['h0.b'])

        l_slice = LL.SliceLayer(l_emb, indices=-1, axis=1)

        l_E0 = LL.DenseLayer(l_slice,
                             num_units=self.out_emb_size,
                             W=params['e0.W'],
                             b=None,
                             nonlinearity=None)

        l_soft_in = LL.ElemwiseSumLayer([l_H0, l_E0])

        l_soft = LL.DenseLayer(l_soft_in,
                               num_units=self.voc_size,
                               nonlinearity=L.nonlinearities.softmax,
                               W=params['soft.W'],
                               b=params['soft.b'])

        l_out = LL.ReshapeLayer(l_soft, (
            self.input_var.shape[0], self.voc_size))

        return l_out, l_dec  # l_out - probabilities, l_dec - new decoder init

    def iterate_minibatches(self, inputs, batch_size, pad=-1):
        for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
            excerpt = slice(start_idx, start_idx + batch_size)
            inp = inputs[excerpt]

            inp_max_len = max(map(len, inp))
            inp = map(lambda l: l + [pad] * (inp_max_len - len(l)), inp)
            inp = np.asarray(inp, dtype=np.int32)
            padding = np.zeros((batch_size, 1), dtype=np.int32) + pad
            tar = np.hstack([inp[:, 1:], padding])

            mask = (inp != pad).astype(np.float32)

            yield inp, tar, mask
