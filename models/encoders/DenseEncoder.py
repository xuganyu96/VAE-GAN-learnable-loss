import mxnet as mx
from mxnet import nd, init, gluon, autograd, image
from mxnet.gluon import data as gdata, loss as gloss, nn
import numpy as np
import d2l

class DenseEncoder(nn.Block):
    
    # An NN block that is the encoder network within a variational autoencoder
    #
    # a DenseEncoder() network consists solely of dense layers
    def __init__(self, n_latent = 10,
                 n_hlayers = 10,
                 n_hnodes = 400):
        super(DenseEncoder, self).__init__()
        
        # Store model hyperparameters
        self.n_latent = 10
        self.n_hlayers = 10
        self.n_hnodes = 400
        
        # Define the network
        with self.name_scope():
            self.encoder = nn.Sequential(prefix='encoder')
            
            # Add the hidden dense layers
            for i in range(n_hlayers):
                self.encoder.add(nn.Dense(n_hnodes, activation='relu'))
                
            # Add the output layer that has 2 * n_latent nodes to 
            # accommodate latent mean and latent log variances
            self.encoder.add(nn.Dense(2 * n_latent))
            
    # The forward() method
    def forward(self, x):
        # input data is image array
        # so we need to flatten x before proceeding
        batch_size = x.shape[0]
        x_flattened = x.reshape(batch_size, -1)
        
        # Return the output of the latent layer
        return self.encoder(x_flattened)
            
                 