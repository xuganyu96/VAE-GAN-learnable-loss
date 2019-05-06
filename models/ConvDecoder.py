# Import the basic packages
import mxnet as mx
from mxnet import nd, gluon
from mxnet.gluon import nn, loss as gloss
import numpy as np
import d2l
CTX = d2l.try_gpu()

# A gluon block that is the decoder of ConvVAE

class ConvDecoder(gluon.Block):
    
    def __init__(self, n_latent = 512,
                 n_channels = 3,
                 out_width = 64,
                 out_height = 64,
                 n_base_channels = 16):
        super(ConvDecoder, self).__init__()
        
        # Store some of the hyper paramteres
        self.n_latent = n_latent
        self.n_channels = n_channels
        self.out_width = out_width
        self.out_height = out_height
        self.n_base_channels = n_base_channels
        
        # Construct the decoder network
        with self.name_scope():
            
            # Construct the decoder network
            # For the decoder network, its input is 4-dimensional arrays
            # of shape (batch_size, n_latent, 1, 1)
            # Decoder's architecture came from Deep Convolutional
            # Generative Adversarial Network tutorial from MXNet
            self.decoder = nn.Sequential(prefix='decoder')
            # Add convolution layers with decreasing number of channels
            self.decoder.add(nn.Conv2DTranspose(n_base_channels*8, 4, 1, 0, use_bias=False),
                             nn.BatchNorm(),
                             nn.Activation('relu'))
            self.decoder.add(nn.Conv2DTranspose(n_base_channels*4, 4, 2, 1, use_bias=False),
                             nn.BatchNorm(),
                             nn.Activation('relu'))
            self.decoder.add(nn.Conv2DTranspose(n_base_channels*2, 4, 2, 1, use_bias=False),
                             nn.BatchNorm(),
                             nn.Activation('relu'))
            self.decoder.add(nn.Conv2DTranspose(n_base_channels*1, 4, 2, 1, use_bias=False),
                             nn.BatchNorm(),
                             nn.Activation('relu'))
            # The last layer uses sigmoid because I want image pixel data between 0 and 1
            # if it doesn't work I will use tanh as the tutorial did
            self.decoder.add(nn.Conv2DTranspose(self.n_channels, 4, 2, 1, use_bias=False),
                             nn.Activation('sigmoid'))
            
    def forward(self, x):
        # x must be 4-dimensional array of shape (batch_size, n_latent, 1, 1)
        
        return self.decoder(x)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    