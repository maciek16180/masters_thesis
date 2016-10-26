# work in progress, clearly

import numpy as np
import theano
import theano.tensor as T

import lasagne as L

sys.path.insert(0, '../HSoftmaxLayerLasagne/')

import HSoftmaxLayer


class SimpleRNN(object):
    
    def __init__(self):
        
    
    def save_params(self, fname='model.npz'):
        np.savez(fname, *self.params)
        
    def load_params(self, fname='model.npz'):
        with np.load(fname) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            L.layers.set_all_param_values(net, param_values)

    