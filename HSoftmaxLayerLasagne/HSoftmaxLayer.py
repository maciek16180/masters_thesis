import numpy as np
import theano.tensor as T
from lasagne import init
from lasagne.layers import Layer, MergeLayer, InputLayer, flatten


class HierarchicalSoftmaxDenseLayer(MergeLayer):
    """
    
    Wraps theano.tensor.nnet.h_softmax for a more convenient usage as lasagne layer.
    
    :param incoming: incoming lasagne layer
    :param num_units: the number of outputs
    :param n_classes: the number of intermediate classes of the two-layer hierarchical softmax.
        It corresponds to the number of outputs of the first softmax. See note at
        the end.  Defaults to sqrt(num_units) or can be inferred from n_outputs_per_class.
    :param n_outputs_per_class: the number of outputs per intermediate class. 
        See note at the end. int, can be inferred
    :param W1: lasagne init or a tensor of shape (number of features of the input x, n_classes)
        the weight matrix of the first softmax, which maps the input x to the
        probabilities of the classes.
    :param b1: lasagne init or a tensor of shape (n_classes,)
        the bias vector of the first softmax layer.
    :param W2: lasagne init or a tensor of shape 
        (n_classes, number of features of the input x, n_outputs_per_class)
        the weight matrix of the second softmax, which maps the input x to
        the probabilities of the outputs.
    :param b2: tensor of shape (n_classes, n_outputs_per_class)
        the bias vector of the second softmax layer.
    :param target: lasagne layer or tensor of shape either (batch_size,) or (batch_size, 1)
        (optional, default None)
        contains the indices of the targets for the minibatch
        input x. For each input, the function computes the output for its
        corresponding target. If target is None, then all the outputs are
        computed for each input.
    
    Notes
    -----
    The product of n_outputs_per_class and n_classes has to be greater or equal
    to n_outputs. If it is strictly greater, then the irrelevant outputs will
    be ignored.
    n_outputs_per_class and n_classes have to be the same as the corresponding
    dimensions of the tensors of W1, b1, W2 and b2.
    The most computational efficient configuration is when n_outputs_per_class
    and n_classes are equal to the square root of n_outputs.
    
        
        """
    def __init__(self,incoming,num_units,
                 n_classes='auto',
                 n_outputs_per_class='auto',
                 W1=init.GlorotUniform(),
                 b1=init.Constant(0),
                 W2=init.GlorotUniform(),
                 b2=init.Constant(0),
                 target=None,
                 **kwargs):

        
        
        #flatten input layer if it has higher dimensionality
        if len(incoming.output_shape) != 2:
            assert len(incoming.output_shape) >= 2
            incoming = flatten(incoming)
        
        incomings = [incoming]
        
        #add target if provided (as theano tensor or lasagne layer)
        if target is not None:
            
            #convert tensor to layer
            if not isinstance(target, Layer):
                assert target.ndim <= 2
                if target.ndim == 1:
                    target_shape = (incoming.shape[0],)
                else:
                    target_shape = (incoming.shape[0], 1)
                    
                target = InputLayer(target_shape, input_var=target, name="target inputlayer")
            
            #check shape
            assert len(target.output_shape) <= 2
            if len(target.output_shape) == 2:
                assert target.output_shape[1] == 1
            
            incomings.append(target)
        
        super(HierarchicalSoftmaxDenseLayer, self).__init__(incomings, **kwargs)
        
        #infer classes
        if n_classes == 'auto':
            if n_outputs_per_class == 'auto':
                n_classes = int(np.ceil(num_units**.5))
            else:
                n_classes = int(np.ceil(float(num_units) / n_outputs_per_class))
        if n_outputs_per_class == 'auto':
            assert n_classes != 'auto'
            n_outputs_per_class = int(np.ceil(float(num_units) / n_classes))
        
        assert n_classes * n_outputs_per_class >= num_units
        
        #remember dimensions
        self.num_units = num_units
        self.n_classes = n_classes
        self.n_outputs_per_class = n_outputs_per_class
        
        #create params
        n_inputs = incoming.output_shape[1]
        self.W1 = self.add_param(W1, (n_inputs, self.n_classes),
                                 name="W1")
        self.b1 = self.add_param(b1, (self.n_classes,),
                                 name="b1", regularizable=False)
        self.W2 = self.add_param(W2, (self.n_classes,n_inputs, self.n_outputs_per_class),
                                 name="W2")
        self.b2 = self.add_param(b2, (self.n_classes, self.n_outputs_per_class),
                                 name="b2", regularizable=False)
        
    def get_output_for(self, inputs, **kwargs):
        """
        Returns
        -------
        output_probs: tensor of shape (batch_size, n_outputs) or (batch_size)
            Output of the two-layer hierarchical softmax for input x. If target is
            not specified (None), then all the outputs are computed and the
            returned tensor has shape (batch_size, n_outputs). Otherwise, when
            target is specified, only the corresponding outputs are computed and
            the returned tensor has thus shape (batch_size,). 
            
            
        """
        
        input_ = inputs[0]
        
        if len(inputs) == 1:
            target = None
        else:
            assert len(inputs) == 2
            target = inputs[1]
            
        return T.nnet.h_softmax(input_, input_.shape[0],
                                self.num_units, self.n_classes,
                                self.n_outputs_per_class,
                                W1=self.W1, b1=self.b1,
                                W2=self.W2, b2=self.b2,
                                target=target)

    def get_output_shape_for(self, input_shapes, **kwargs):
        if len(input_shapes) == 1:
            return input_shapes[0][0], self.num_units
        else:
            return input_shapes[0][0],