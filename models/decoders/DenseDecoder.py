import mxnet as mx
from mxnet import nd, init, gluon, autograd, image
from mxnet.gluon import data as gdata, loss as gloss, nn
import numpy as np
import d2l

class DenseDecoder(nn.Block):
    
    # An NN block that is the decoder network of a variational autoencoder
    #
    # A DenseDecoder() network consists solely of dense layers
    
    def __init__(self, n_hlayers = 10,
                 n_hnodes = 400,
                 n_out_channels = 1,
                 out_width = 28,
                 out_height = 28):
        
        # the number of latent variables shouldn't be a concern
        # on the other hand, the number of output channels, the width
        # and the heights are both concerns
        
        super(DenseDecoder, self).__init__()
        self.n_hlayers = n_hlayers
        self.n_hnodes = n_hnodes
        self.n_out_channels = n_out_channels
        self.out_width = out_width
        self.out_height = out_height
        
        # Define the decoder network
        with self.name_scope():
            self.decoder = nn.Sequential(prefix='decoder')
            
            for i in range(n_hlayers):
                # Add the hidden dense layers
                self.decoder.add(nn.Dense(n_hnodes, activation='relu'))
                
            # Add the output layer
            # At this point the output layer is a flattened image
            # but we will do with a reshape in forward()
            self.decoder.add(nn.Dense(n_out_channels * out_width * out_height,
                                     activation='sigmoid'))
            
    def forward(self, x):
        # x is the latent variable, and thus of shape (batch_size, n_latent)
        
        # Compute the flattened output
        x_hat_flattened = self.decoder(x)
        # Inflate the output into 4-dimensional image array
        x_hat = x_hat_flattened.reshape(-1, self.n_out_channels,
                                        self.out_width,
                                        self.out_height)
        
        return x_hat