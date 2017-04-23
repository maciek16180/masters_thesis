import numpy as np
import theano
import theano.tensor as T
import time

import lasagne as L

import sys
sys.path.insert(0, '../HSoftmaxLayerLasagne/')

from HSoftmaxLayer import HierarchicalSoftmaxDenseLayer
from SampledSoftmaxLayer import SampledSoftmaxDenseLayer
from NCELayer import NCEDenseLayer


class SimpleRNNLM(object):
    
    def __init__(self, voc_size, emb_size, rec_size, mode='ssoft', pad_value=-1, **kwargs):
        self.pad_value = pad_value
        self.voc_size = voc_size
        self.emb_size = emb_size
        self.rec_size = rec_size

        input_var = T.imatrix('inputs')
        target_var = T.imatrix('targets')  # these will be inputs shifted by 1
        mask_input_var = T.matrix('input_mask')
        mask_idx = mask_input_var.nonzero()

        # BUILD THE MODEL
        print 'Building the model...'

        assert mode in ['full', 'ssoft', 'hsoft', 'nce']

        if mode == 'full':
            self.train_net = _build_full_softmax_net(input_var, mask_input_var, voc_size, emb_size, rec_size, **kwargs)
        elif mode == 'ssoft':
            self.train_net = _build_sampled_softmax_net(input_var, mask_input_var, voc_size, emb_size, rec_size,
                                                        target_var=target_var, **kwargs)
        elif mode == 'hsoft':
            self.train_net = _build_hierarchical_softmax_net(input_var, mask_input_var, voc_size, emb_size, rec_size,
                                                             target_var=target_var, **kwargs)
        elif mode == 'nce':
            self.train_net = _build_nce_net(input_var, mask_input_var, voc_size, emb_size, rec_size,
                                            target_var=target_var, **kwargs)

        # CALCULATE THE LOSS

        train_out = L.layers.get_output(self.train_net)
        test_out = L.layers.get_output(self.train_net, deterministic=True)

        if mode == 'full':
            train_loss = L.objectives.categorical_crossentropy(train_out[mask_idx], target_var[mask_idx]).mean()
            test_loss = L.objectives.categorical_crossentropy(test_out[mask_idx], target_var[mask_idx]).mean()
        elif mode in ['ssoft', 'hsoft']:
            train_loss = -T.log(train_out[mask_idx]).mean()
            test_loss = -T.log(test_out[mask_idx]).mean()
        elif mode == 'nce':
            train_loss = train_out[mask_idx].mean() # NCEDenseLayer uses logreg loss, so we don't -T.log here
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

        if mode in ['full', 'ssoft', 'nce']:
            pars = L.layers.get_all_params(self.train_net)
            if mode in ['ssoft', 'nce']:
                del pars[-1] # remove ssoft/noise probs, not needed for generation
            self.gen_net = _build_full_softmax_net_with_params(input_var, voc_size, emb_size, rec_size, pars)
        elif mode == 'hsoft':
            self.gen_net = _build_hierarchical_softmax_net_with_params(input_var, voc_size, emb_size, rec_size,
                                                                       L.layers.get_all_params(self.train_net))

        probs = L.layers.get_output(self.gen_net)[:, -1, :]
        self.get_probs_fn = theano.function([input_var], probs)
        
        print 'Done'

    def train_one_epoch(self, train_data, batch_size, log_interval=10):
        train_err = 0.
        train_batches = 0
        num_training_words = 0
        start_time = time.time()

        for batch in iterate_minibatches(train_data, batch_size, self.pad_value):
            inputs, targets, mask = batch

            num_batch_words = mask.sum()
            train_err += self.train_fn(inputs, targets, mask) * num_batch_words
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
            inputs, targets, mask = batch

            num_batch_words = mask.sum()
            val_err += self.val_fn(inputs, targets, mask) * num_batch_words
            val_batches += 1
            num_validate_words += num_batch_words

            if not val_batches % 100:
                print "Done {} batches in {:.2f}s".format(val_batches, time.time() - start_time)

        return val_err / num_validate_words

    def train_model(self, train_data, val_data, train_batch_size, val_batch_size, num_epochs,
                    save_params=False, path=None, log_interval=10):
        if save_params:
            open(path, 'w').close()

        for epoch in range(num_epochs):
            start_time = time.time()

            train_err = self.train_one_epoch(train_data, train_batch_size, log_interval)
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


def __build_architecture(input_var, mask_input_var, voc_size, emb_size, rec_size,
                         emb_init=None, train_emb=True):
    l_in = L.layers.InputLayer(shape=(None, None), input_var=input_var)

    l_mask = None
    if mask_input_var is not None:
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
    l_resh = __build_architecture(input_var, mask_input_var, voc_size, emb_size, rec_size,
                                  emb_init=emb_init, train_emb=train_emb)

    l_soft = L.layers.DenseLayer(l_resh,
                                 num_units=voc_size,
                                 nonlinearity=L.nonlinearities.softmax)

    l_out = L.layers.ReshapeLayer(l_soft, shape=(input_var.shape[0], input_var.shape[1], voc_size))

    return l_out


def _build_sampled_softmax_net(input_var, mask_input_var, voc_size, emb_size, rec_size, num_sampled,
                               emb_init=None, train_emb=True, target_var=None,
                               ssoft_probs=None, sample_unique=False, **kwargs):
    l_resh = __build_architecture(input_var, mask_input_var, voc_size, emb_size, rec_size,
                                  emb_init=emb_init, train_emb=train_emb)

    if target_var is not None:
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


def _build_nce_net(input_var, mask_input_var, voc_size, emb_size, rec_size, num_sampled,
                   emb_init=None, train_emb=True, target_var=None,
                   noise_probs=None, sample_unique=False, **kwargs):
    l_resh = __build_architecture(input_var, mask_input_var, voc_size, emb_size, rec_size,
                                  emb_init=emb_init, train_emb=train_emb)

    if target_var is not None:
        target_var = target_var.ravel()

    l_ssoft = NCEDenseLayer(l_resh, num_sampled, voc_size,
                            targets=target_var,
                            probs=noise_probs,
                            sample_unique=sample_unique)

    if target_var is not None:
        l_out = L.layers.ReshapeLayer(l_ssoft, shape=(input_var.shape[0], input_var.shape[1]))
    else:
        l_out = L.layers.ReshapeLayer(l_ssoft, shape=(input_var.shape[0], input_var.shape[1], voc_size))

    return l_out


def _build_hierarchical_softmax_net(input_var, mask_input_var, voc_size, emb_size, rec_size,
                                    emb_init=None, train_emb=True, target_var=None, **kwargs):
    l_resh = __build_architecture(input_var, mask_input_var, voc_size, emb_size, rec_size,
                                  emb_init=emb_init, train_emb=train_emb)

    if target_var is not None:
        target_var = target_var.ravel()

    l_hsoft = HierarchicalSoftmaxDenseLayer(l_resh,
                                            num_units=voc_size,
                                            target=target_var)

    if target_var is not None:
        l_out = L.layers.ReshapeLayer(l_hsoft, shape=(input_var.shape[0], input_var.shape[1]))
    else:
        l_out = L.layers.ReshapeLayer(l_hsoft, shape=(input_var.shape[0], input_var.shape[1], voc_size))

    return l_out


def __build_architecture_with_params(input_var, voc_size, emb_size, rec_size, params):
    assert len(params) == 35
    params = [params[:1], params[1:18], params[18:35]]
    em, r1, r2 = map(lambda p: {x.name: x for x in p}, params)

    l_in = L.layers.InputLayer(shape=(None, None), input_var=input_var)

    l_emb = L.layers.EmbeddingLayer(l_in,
                                    input_size=voc_size,
                                    output_size=emb_size,
                                    W=em['W'])

    l_lstm1 = L.layers.LSTMLayer(l_emb,
                                 num_units=rec_size,
                                 grad_clipping=100,
                                 mask_input=None,
                                 ingate=L.layers.Gate(W_in=r1['W_in_to_ingate'],
                                                      W_hid=r1['W_hid_to_ingate'],
                                                      W_cell=r1['W_cell_to_ingate'],
                                                      b=r1['b_ingate']),
                                 forgetgate=L.layers.Gate(W_in=r1['W_in_to_forgetgate'],
                                                          W_hid=r1['W_hid_to_forgetgate'],
                                                          W_cell=r1['W_cell_to_forgetgate'],
                                                          b=r1['b_forgetgate']),
                                 cell=L.layers.Gate(W_in=r1['W_in_to_cell'],
                                                    W_hid=r1['W_hid_to_cell'],
                                                    W_cell=None,
                                                    b=r1['b_cell'],
                                                    nonlinearity=L.nonlinearities.tanh),
                                 outgate=L.layers.Gate(W_in=r1['W_in_to_outgate'],
                                                       W_hid=r1['W_hid_to_outgate'],
                                                       W_cell=r1['W_cell_to_outgate'],
                                                       b=r1['b_outgate']),
                                 cell_init=r1['cell_init'],
                                 hid_init=r1['hid_init'])

    l_lstm2 = L.layers.LSTMLayer(l_lstm1,
                                 num_units=rec_size,
                                 grad_clipping=100,
                                 mask_input=None,
                                 ingate=L.layers.Gate(W_in=r2['W_in_to_ingate'],
                                                      W_hid=r2['W_hid_to_ingate'],
                                                      W_cell=r2['W_cell_to_ingate'],
                                                      b=r2['b_ingate']),
                                 forgetgate=L.layers.Gate(W_in=r2['W_in_to_forgetgate'],
                                                          W_hid=r2['W_hid_to_forgetgate'],
                                                          W_cell=r2['W_cell_to_forgetgate'],
                                                          b=r2['b_forgetgate']),
                                 cell=L.layers.Gate(W_in=r2['W_in_to_cell'],
                                                    W_hid=r2['W_hid_to_cell'],
                                                    W_cell=None,
                                                    b=r2['b_cell'],
                                                    nonlinearity=L.nonlinearities.tanh),
                                 outgate=L.layers.Gate(W_in=r2['W_in_to_outgate'],
                                                       W_hid=r2['W_hid_to_outgate'],
                                                       W_cell=r2['W_cell_to_outgate'],
                                                       b=r2['b_outgate']),
                                 cell_init=r2['cell_init'],
                                 hid_init=r2['hid_init'])

    l_resh = L.layers.ReshapeLayer(l_lstm2, shape=(-1, rec_size))

    return l_resh


def _build_full_softmax_net_with_params(input_var, voc_size, emb_size, rec_size, params):
    assert len(params) == 37
    sm = {x.name : x for x in params[-3:]}
    l_resh = __build_architecture_with_params(input_var, voc_size, emb_size, rec_size, params[:-2])

    l_soft = L.layers.DenseLayer(l_resh,
                                 num_units=voc_size,
                                 nonlinearity=L.nonlinearities.softmax,
                                 W=sm['W'],
                                 b=sm['b'])

    l_out = L.layers.ReshapeLayer(l_soft, shape=(input_var.shape[0], input_var.shape[1], voc_size))

    return l_out


def _build_hierarchical_softmax_net_with_params(input_var, voc_size, emb_size, rec_size, params):
    assert len(params) == 39
    sm = {x.name: x for x in params[-4:]}
    l_resh = __build_architecture_with_params(input_var, voc_size, emb_size, rec_size, params[:-4])

    l_hsoft = HierarchicalSoftmaxDenseLayer(l_resh,
                                            num_units=voc_size,
                                            target=None,
                                            W1=sm['W1'],
                                            b1=sm['b1'],
                                            W2=sm['W2'],
                                            b2=sm['b2'])

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