# work in progress, clearly

import numpy as np
import theano
import theano.tensor as T

import lasagne as L

import sys
sys.path.insert(0, '../HSoftmaxLayerLasagne/')

from HSoftmaxLayer import HierarchicalSoftmaxDenseLayer
from SampledSoftmaxLayer import SampledSoftmaxDenseLayer


class SimpleRNNLM(object):
    
    def __init__(self, mode='ssoft'):
        pass

    @staticmethod
    def __build_net(input_var, voc_mask_var, voc_size, emb_size, rec_size,
                    mask_input_var= None, target_var=None, mode='ssoft', gclip=100):

        l_in = L.layers.InputLayer(shape=(None, None), input_var=input_var)
        batch_size, seq_len = l_in.input_var.shape

        l_mask = None
        if mask_input_var is not None:
            print 'setting up input mask...'
            l_mask = L.layers.InputLayer(shape=(batch_size, seq_len), input_var=mask_input_var)

        l_emb = L.layers.EmbeddingLayer(l_in,
                                        input_size=voc_size+1,
                                        output_size=emb_size)

        l_lstm1 = L.layers.LSTMLayer(l_emb,
                                     num_units=rec_size,
                                     nonlinearity=L.nonlinearities.tanh,
                                     grad_clipping=gclip,
                                     mask_input=l_mask)

        l_lstm2 = L.layers.LSTMLayer(l_lstm1,
                                     num_units=rec_size,
                                     nonlinearity=L.nonlinearities.tanh,
                                     grad_clipping=gclip,
                                     mask_input=l_mask)

        l_resh = L.layers.ReshapeLayer(l_lstm2, shape=(-1, rec_size))

        l_tar = None
        if target_var is not None:
            print 'setting up targets for sampled softmax...'
            l_tar = L.layers.InputLayer(shape=(-1,), input_var=target_var.reshape(shape=(batch_size * seq_len,)))

        if mode == 'ssoft':
            l_soft = SampledSoftmaxDenseLayer(l_resh, voc_mask_var, voc_size, targets=l_tar)
        elif mode == 'hsoft':
            l_soft = HierarchicalSoftmaxDenseLayer(l_resh, voc_size, target=l_tar)
        else:
            raise NameError('Mode not recognised.')

        if target_var is not None:
            l_out = L.layers.ReshapeLayer(l_soft, shape=(batch_size, seq_len))
        else:
            l_out = L.layers.ReshapeLayer(l_soft, shape=(batch_size, seq_len, voc_size))

        return l_out


    def save_params(self, fname='model.npz'):
        np.savez(fname, *self.params)

    def load_params(fname='model.npz'):
        with np.load(fname) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            L.layers.set_all_param_values(net, param_values)
            
    @staticmethod
    def rnd_next_word(probs, size=1):
        return np.random.choice(np.arange(probs.shape[0], dtype=np.int32), size=size, p=probs)

    def beam_search(get_probs_fun, beam=10, init_seq='', mode='rr'):
        utt = [1] + map(lambda w: mt_w_to_i.get(w, mt_w_to_i['<unk>']), init_seq.split())
        utt = np.asarray(utt, dtype=np.int32)[np.newaxis]

        if mode[0] == 's':
            words = get_probs_fun(utt)[0].argpartition(-beam)[-beam:].astype(np.int32)
        elif mode[0] == 'r':
            words = rnd_next_word(get_probs_fun(utt)[0], beam)

        candidates = utt.repeat(beam, axis=0)
        candidates = np.hstack([candidates, words[np.newaxis].T])
        scores = np.zeros(beam)

        while 0 not in candidates[:,-1] and candidates.shape[1] < 100:

            if mode[1] == 's':
                log_probs = np.log(get_probs_fun(candidates))
                tot_scores = log_probs + scores[np.newaxis].T

                idx = tot_scores.ravel().argpartition(-beam)[-beam:]
                i,j = idx / tot_scores.shape[1], (idx % tot_scores.shape[1]).astype(np.int32)

                scores = tot_scores[i,j]

                candidates = np.hstack([candidates[i], j[np.newaxis].T])

            elif mode[1] == 'r':
                probs = get_probs_fun(candidates)
                words = []
                for k in xrange(beam):
                    words.append(rnd_next_word(probs[k], beam)) # this doesn't have to be exactly 'beam'
                words = np.array(words)
                idx = np.indices((beam, words.shape[1]))[0]
                tot_scores = scores[np.newaxis].T + np.log(probs)[idx, words]

                idx = tot_scores.ravel().argpartition(-beam)[-beam:]
                i,j = idx / tot_scores.shape[1], (idx % tot_scores.shape[1])

                scores = tot_scores[i,j]

                candidates = np.hstack([candidates[i], words[i,j][np.newaxis].T])

        return candidates[candidates[:,-1] == 0][0]