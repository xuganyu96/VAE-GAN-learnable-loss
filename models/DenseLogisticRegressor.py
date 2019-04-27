# Import the basic packages
import mxnet as mx
from mxnet import nd, gluon
from mxnet.gluon import nn, loss as gloss
import numpy as np
import d2l
CTX = d2l.try_gpu()

# This will be a simple logistic regression discriminator
class DenseLogisticRegressor(gluon.Block):
    
    def __init__(self, n_hlayers = 0,
                 n_hnodes = 10,
                 n_classes = 1):
        
        super(DenseLogisticRegressor, self).__init__()
        
        # Construct the simple logistic regression network
        with self.name_scope():
            self.discriminator = nn.Sequential()
            
            for hlayer in range(n_hlayers):
                self.discriminator.add(nn.Dense(n_hnodes, activation='relu'))
                
            # The final output layer will be logits, not probabilities
            # so no sigmoid transformation is needed for the activation
            # the probability predictions will be obtained when
            # the loss function is computed (with from_sigmoid = False)
            self.discriminator.add(nn.Dense(n_classes))
            
    def forward(self, x):
        # The input data is 4-dimensional image arrays, but the dimensionalities
        # don't really matter; the input array will simply be flattened into
        # shape (n_batch, n_pixels)
        x_flattened = x.reshape((x.shape[0], -1))
        
        # Feed the input into the network
        logit_preds = self.discriminator(x_flattened)
        
        # Return the logits
        return logit_preds
        