from __future__ import print_function

import numpy as np
import theano
import theano.tensor as T
import time

import lasagne as L

import sys
sys.path.append('../')

from layers import HierarchicalSoftmaxDenseLayer
from layers import SampledSoftmaxDenseLayer
from layers import NCEDenseLayer
from layers import ShiftLayer


class SimpleRNNLM(object):

    def __init__(self, voc_size, emb_size, rec_size, mode='ssoft', **kwargs):

        self.voc_size = voc_size
        self.emb_size = emb_size
        self.rec_size = rec_size

        self.input_var = T.imatrix('inputs')
        self.mask_input_var = T.matrix('input_mask')
        mask_idx = self.mask_input_var.nonzero()

        self.emb_init = kwargs.get('emb_init', None)

        # BUILD THE MODEL
        print('Building the model...')

        assert mode in ['full', 'ssoft', 'hsoft', 'nce']

        if mode == 'full':
            self.train_net = self._build_full_softmax_net(**kwargs)
        elif mode == 'ssoft':
            self.train_net = self._build_sampled_softmax_net(**kwargs)
        elif mode == 'hsoft':
            self.train_net = self._build_hierarchical_softmax_net(**kwargs)
        elif mode == 'nce':
            self.train_net = self._build_nce_net(**kwargs)

        # CALCULATE THE LOSS

        train_out = L.layers.get_output(self.train_net)
        test_out = L.layers.get_output(self.train_net, deterministic=True)

        if mode == 'full':
            train_loss = L.objectives.categorical_crossentropy(
                train_out[mask_idx], self.input_var[mask_idx]).mean()
            test_loss = L.objectives.categorical_crossentropy(
                test_out[mask_idx], self.input_var[mask_idx]).mean()
        elif mode in ['ssoft', 'hsoft']:
            train_loss = -T.log(train_out[mask_idx]).mean()
            test_loss = -T.log(test_out[mask_idx]).mean()
        elif mode == 'nce':
            # NCEDenseLayer uses logreg loss, so we don't -T.log here
            train_loss = train_out[mask_idx].mean()
            test_loss = -T.log(test_out[mask_idx]).mean()

        # MAKE TRAIN AND VALIDATION FUNCTIONS
        print('Compiling theano functions...')

        params = L.layers.get_all_params(self.train_net, trainable=True)

        if kwargs.has_key('update_fn'):
            update_fn = kwargs['update_fn']
        else:
            update_fn = lambda l, p: L.updates.adam(l, p, learning_rate=.0001)

        updates = update_fn(train_loss, params)

        self.train_fn = theano.function(
            [self.input_var, self.mask_input_var],
            train_loss, updates=updates)
        self.val_fn = theano.function(
            [self.input_var, self.mask_input_var], test_loss)

        # BUILD NET FOR GENERATING, WITH SHARED PARAMETERS
        print('Building a network for generating...')

        all_params = {x.name: x
                      for x in L.layers.get_all_params(self.train_net)}

        if mode in ['full', 'ssoft', 'nce']:
            self.gen_net = self._build_full_softmax_net_with_params(all_params)
        elif mode == 'hsoft':
            self.gen_net = self._build_hierarchical_softmax_net_with_params(
                all_params)

        probs = L.layers.get_output(self.gen_net)[:, -1, :]
        self.get_probs_fn = theano.function([self.input_var], probs)

        print('Done')

    def train_one_epoch(self, train_data, batch_size, log_interval=10):
        train_err = 0.
        train_batches = 0
        num_training_words = 0
        start_time = time.time()

        for batch in self.iterate_minibatches(train_data, batch_size):
            inputs, mask = batch

            num_batch_words = mask.sum()
            train_err += self.train_fn(inputs, mask) * num_batch_words
            train_batches += 1
            num_training_words += num_batch_words

            if not train_batches % log_interval:
                print("Done {} batches in {:.2f}s\ttraining loss:\t{:.6f}".
                      format(train_batches, time.time() - start_time,
                             train_err / num_training_words))

        return train_err / num_training_words

    def validate(self, val_data, batch_size):
        val_err = 0.
        val_batches = 0
        num_validate_words = 0
        start_time = time.time()

        for batch in self.iterate_minibatches(val_data, batch_size):
            inputs, mask = batch

            num_batch_words = mask.sum()
            val_err += self.val_fn(inputs, mask) * num_batch_words
            val_batches += 1
            num_validate_words += num_batch_words

            if not val_batches % 100:
                print("Done {} batches in {:.2f}s".format(
                    val_batches, time.time() - start_time))

        return val_err / num_validate_words

    def train_model(self, train_data, val_data, train_batch_size,
                    val_batch_size, num_epochs, save_params=False, path=None,
                    log_interval=10):
        if save_params:
            open(path, 'w').close()

        for epoch in range(num_epochs):
            start_time = time.time()

            train_err = self.train_one_epoch(
                train_data, train_batch_size, log_interval)
            val_err = self.validate(val_data, val_batch_size)

            print("Epoch {} of {} took {:.2f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err))
            print("  validation loss:\t\t{:.6f}".format(val_err))

        if save_params:
            self.save_params(path)

    def save_params(self, fname='model.npz'):
        np.savez(fname, *L.layers.get_all_param_values(self.train_net))

    def load_params(self, fname='model.npz'):
        with np.load(fname) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            L.layers.set_all_param_values(self.train_net, param_values)

    def _build_architecture(self, train_emb=True):
        l_in = L.layers.InputLayer(shape=(None, None),
                                   input_var=self.input_var)

        l_mask = None
        if self.mask_input_var is not None:
            l_mask = L.layers.InputLayer(shape=(None, None),
                                         input_var=self.mask_input_var)

        if self.emb_init is None:
            l_emb = L.layers.EmbeddingLayer(l_in,
                                            input_size=self.voc_size,
                                            output_size=self.emb_size,
                                            name='emb')
        else:
            l_emb = L.layers.EmbeddingLayer(l_in,
                                            input_size=self.voc_size,
                                            output_size=self.emb_size,
                                            W=self.emb_init,
                                            name='emb')
        if not train_emb:
            l_emb.params[l_emb.W].remove('trainable')

        l_lstm1 = L.layers.LSTMLayer(l_emb,
                                     num_units=self.rec_size,
                                     nonlinearity=L.nonlinearities.tanh,
                                     grad_clipping=100,
                                     mask_input=l_mask,
                                     name='LSTM1')

        l_resh = L.layers.ReshapeLayer(ShiftLayer(l_lstm1),
                                       shape=(-1, self.rec_size))

        return l_resh

    def _build_full_softmax_net(self, train_emb=True, **kwargs):
        l_resh = self._build_architecture(train_emb=train_emb)

        l_soft = L.layers.DenseLayer(l_resh,
                                     num_units=self.voc_size,
                                     nonlinearity=L.nonlinearities.softmax,
                                     name='soft')

        l_out = L.layers.ReshapeLayer(l_soft, tuple(
            self.input_var.shape) + (self.voc_size,))

        return l_out

    def _build_sampled_softmax_net(self, num_sampled, train_emb=True,
                                   ssoft_probs=None, sample_unique=False,
                                   **kwargs):
        l_resh = self._build_architecture(train_emb=train_emb)

        l_ssoft = SampledSoftmaxDenseLayer(l_resh, num_sampled, self.voc_size,
                                           targets=self.input_var.ravel(),
                                           probs=ssoft_probs,
                                           sample_unique=sample_unique,
                                           name='soft')

        l_out = L.layers.ReshapeLayer(l_ssoft, tuple(self.input_var.shape))
        return l_out

    def _build_nce_net(self, num_sampled, train_emb=True, noise_probs=None,
                       sample_unique=False, **kwargs):
        l_resh = self._build_architecture(train_emb=train_emb)

        l_ssoft = NCEDenseLayer(l_resh, num_sampled, self.voc_size,
                                targets=self.input_var.ravel(),
                                probs=noise_probs,
                                sample_unique=sample_unique,
                                name='soft')

        l_out = L.layers.ReshapeLayer(l_ssoft, tuple(self.input_var.shape))
        return l_out

    def _build_hierarchical_softmax_net(self, train_emb=True, **kwargs):
        l_resh = self._build_architecture(train_emb=train_emb)

        l_hsoft = HierarchicalSoftmaxDenseLayer(l_resh,
                                                num_units=self.voc_size,
                                                target=self.input_var.ravel(),
                                                name='soft')

        l_out = L.layers.ReshapeLayer(l_hsoft, tuple(self.input_var.shape))
        return l_out

    def _build_architecture_with_params(self, params):

        l_in = L.layers.InputLayer(shape=(None, None),
                                   input_var=self.input_var)

        l_emb = L.layers.EmbeddingLayer(l_in,
                                        input_size=self.voc_size,
                                        output_size=self.emb_size,
                                        W=params['emb.W'])

        l_lstm1 = L.layers.LSTMLayer(
            l_emb,
            num_units=self.rec_size,
            grad_clipping=100,
            mask_input=None,
            ingate=L.layers.Gate(
                W_in=params['LSTM1.W_in_to_ingate'],
                W_hid=params['LSTM1.W_hid_to_ingate'],
                W_cell=params['LSTM1.W_cell_to_ingate'],
                b=params['LSTM1.b_ingate']),
            forgetgate=L.layers.Gate(
                W_in=params['LSTM1.W_in_to_forgetgate'],
                W_hid=params['LSTM1.W_hid_to_forgetgate'],
                W_cell=params['LSTM1.W_cell_to_forgetgate'],
                b=params['LSTM1.b_forgetgate']),
            cell=L.layers.Gate(
                W_in=params['LSTM1.W_in_to_cell'],
                W_hid=params['LSTM1.W_hid_to_cell'],
                W_cell=None,
                b=params['LSTM1.b_cell'],
                nonlinearity=L.nonlinearities.tanh),
            outgate=L.layers.Gate(
                W_in=params['LSTM1.W_in_to_outgate'],
                W_hid=params['LSTM1.W_hid_to_outgate'],
                W_cell=params['LSTM1.W_cell_to_outgate'],
                b=params['LSTM1.b_outgate']),
            cell_init=params['LSTM1.cell_init'],
            hid_init=params['LSTM1.hid_init'])

        l_resh = L.layers.ReshapeLayer(l_lstm1, shape=(-1, self.rec_size))

        return l_resh

    def _build_full_softmax_net_with_params(self, params):

        l_resh = self._build_architecture_with_params(params)

        l_soft = L.layers.DenseLayer(l_resh,
                                     num_units=self.voc_size,
                                     nonlinearity=L.nonlinearities.softmax,
                                     W=params['soft.W'],
                                     b=params['soft.b'])

        l_out = L.layers.ReshapeLayer(l_soft, tuple(
            self.input_var.shape) + (self.voc_size,))

        return l_out

    def _build_hierarchical_softmax_net_with_params(self, params):

        l_resh = self._build_architecture_with_params(params)

        l_hsoft = HierarchicalSoftmaxDenseLayer(l_resh,
                                                num_units=self.voc_size,
                                                target=None,
                                                W1=params['soft.W1'],
                                                b1=params['soft.b1'],
                                                W2=params['soft.W2'],
                                                b2=params['soft.b2'])

        l_out = L.layers.ReshapeLayer(l_hsoft, tuple(
            self.input_var.shape) + (self.voc_size,))

        return l_out

    def iterate_minibatches(self, inputs, batch_size, pad=-1):
        for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
            excerpt = slice(start_idx, start_idx + batch_size)
            inp = inputs[excerpt]

            inp_max_len = max(map(len, inp))
            inp = map(lambda l: l + [pad] * (inp_max_len - len(l)), inp)
            inp = np.asarray(inp, dtype=np.int32)
            mask = (inp != pad).astype(np.float32)

            yield inp, mask
