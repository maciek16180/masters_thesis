# work in progress

import numpy as np
import theano
import theano.tensor as T
import time

import lasagne as L

import sys
sys.path.insert(0, '../HSoftmaxLayerLasagne/')

from HSoftmaxLayer import HierarchicalSoftmaxDenseLayer
from SampledSoftmaxLayer import SampledSoftmaxDenseLayer


class SimpleRNNLM(object):
    
    def __init__(self, voc_size, emb_size, rec_size, mode='ssoft', pad_value=-1, **kwargs):
        self.pad_value = pad_value

        input_var = T.imatrix('inputs')
        target_var = T.imatrix('targets')  # these will be inputs shifted by 1
        mask_input_var = T.matrix('input_mask')
        mask_idx = mask_input_var.nonzero()

        # BUILD THE MODEL

        assert mode in ['full', 'ssoft', 'hsoft']

        if mode == 'full':
            self.train_net = _build_full_softmax_net(input_var, mask_input_var, voc_size, emb_size, rec_size, **kwargs)
        elif mode == 'ssoft':
            num_sampled = kwargs['num_sampled']
            self.train_net = _build_sampled_softmax_net(input_var, mask_input_var, voc_size, emb_size, rec_size,
                                                        num_sampled, target_var=target_var, **kwargs)
        elif mode == 'hsoft':
            self.train_net = _build_hierarchical_softmax_net(input_var, mask_input_var, voc_size, emb_size, rec_size,
                                                             target_var=target_var, **kwargs)

        # CALCULATE THE LOSS

        train_out = L.layers.get_output(self.train_net)
        test_out = L.layers.get_output(self.train_net, deterministic=True)

        if mode == 'full':
            train_loss = L.objectives.categorical_crossentropy(train_out[mask_idx], target_var[mask_idx]).mean()
            test_loss = L.objectives.categorical_crossentropy(test_out[mask_idx], target_var[mask_idx]).mean()
        elif mode in ['ssoft', 'hsoft']:
            train_loss = -T.sum(T.log(train_out[mask_idx])) / T.sum(mask_input_var)
            test_loss = -T.sum(T.log(test_out[mask_idx])) / T.sum(mask_input_var)

        # MAKE TRAIN AND VALIDATION FUNCTIONS

        params = L.layers.get_all_params(self.train_net, trainable=True)
        updates = L.updates.adagrad(train_loss, params, learning_rate=.01)

        self.train_fn = theano.function([input_var, target_var, mask_input_var], train_loss, updates=updates)
        self.val_fn = theano.function([input_var, target_var, mask_input_var], test_loss)

    def train_one_epoch(self, train_data, batch_size):
        train_err = 0.
        train_batches = 0
        start_time = time.time()

        for batch in iterate_minibatches(train_data, batch_size, self.pad_value):
            inputs, targets, mask = batch

            train_err += self.train_fn(inputs, targets, mask)
            train_batches += 1

            if not train_batches % 10:
                print "Done {} batches in {:.2f}s\ttraining loss:\t{:.6f}".format(
                    train_batches, time.time() - start_time, train_err / train_batches)

        return  train_err / train_batches

    def validate(self, val_data, batch_size):
        val_err = 0.
        val_batches = 0
        start_time = time.time()

        for batch in iterate_minibatches(val_data, batch_size, self.pad_value):
            inputs, targets, mask = batch

            val_err += self.val_fn(inputs, targets, mask)
            val_batches += 1

            if not val_batches % 100:
                print "Done {} batches in {:.2f}s".format(val_batches, time.time() - start_time)

        return val_err / val_batches

    def train_model(self, train_data, val_data, train_batch_size, val_batch_size, num_epochs,
                    save_params=False, path=None):
        if save_params:
            open(path, 'w').close()

        for epoch in range(num_epochs):
            start_time = time.time()

            train_err = self.train_one_epoch(train_data, train_batch_size)
            val_err = self.validate(val_data, val_batch_size)

            print "Epoch {} of {} took {:.2f}s".format(epoch + 1, num_epochs, time.time() - start_time)
            print "  training loss:\t\t{:.6f}".format(train_err)
            print "  validation loss:\t\t{:.6f}".format(val_err)

        if save_params:
            self.save_params(path)

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


def iterate_minibatches(inputs, batch_size, pad=-1):
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        excerpt = slice(start_idx, start_idx + batch_size)
        inp = inputs[excerpt]

        inp_max_len = len(max(inp, key=len))
        inp = map(lambda l: l + [pad] * (inp_max_len - len(l)), inp)
        inp = np.asarray(inp, dtype=np.int32)
        tar = np.hstack([inp[:, 1:], np.zeros((batch_size, 1), dtype=np.int32) + pad])

        v_not_pad = np.vectorize(lambda x: x != pad, otypes=[np.float32])
        mask = v_not_pad(inp)  # there is no separate value for the end of an utterance, just pad

        yield inp, tar, mask