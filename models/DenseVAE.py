# Import the basic packages
import mxnet as mx
from mxnet import nd, gluon
from mxnet.gluon import nn
import numpy as np
import d2l
CTX = d2l.try_gpu()

# Import the Encoder and Decoder networks. All paths are
# relative to the main directory
import sys
sys.path.insert(0, "./models/encoders")
sys.path.insert(0, "./models/decoders")
from DenseEncoder import DenseEncoder
from DenseDecoder import DenseDecoder

# This script provides a subclass of gluon.Block that is the
# VAE network. The implementation is identical to that of the demo
# provided in https://gluon.mxnet.io/chapter13_unsupervised-learning/vae-gluon.html

class DenseVAE(gluon.Block):
    
    def __init__(self, n_latent = 10,
                 n_hlayers = 10,
                 n_hnodes = 400,
                 n_out_channels = 1,
                 out_width = 28,
                 out_height = 28):
        
        # Store some hyperparameters
        self.n_latent = n_latent
        self.n_hlayers = n_hlayers
        self.n_hnodes = n_hnodes
        self.n_out_channels = n_out_channels
        self.out_width = out_width
        self.out_height = out_height
        
        # Initialize the super class
        super(DenseVAE, self).__init__()
        
        # Define the networks: encoder and decoder
        with self.name_scope():
            self.encoder = DenseEncoder(n_latent = n_latent,
                                        n_hlayers = n_hlayers,
                                        n_hnodes = n_hnodes)
            
            self.decoder = DenseDecoder(n_hlayers = n_hlayers,
                                        n_hnodes = n_hnodes,
                                        n_out_channels = n_out_channels,
                                        out_width = out_width,
                                        out_height = out_height)
            
    def forward(self, x):
        # x should be image arrays (4-dimensional) but encoder should be able 
        # to handle this so I am not going flatten it
        
        # Use the encoder network to compute the values of latent layers
        latent_layer = self.encoder(x)
        
        # Split the latent layer into latent means and latent log vars
        latent_mean = nd.split(latent_layer, axis=1, num_outputs = 2)[0]
        latent_logvar = nd.split(latent_layer, axis=1, num_outputs = 2)[1]
        
        # Use the reparametrization trick to ensure differentiability of the latent
        # variable
        eps = nd.random_normal(loc=0,
                               scale=1,
                               shape=(x.shape[0], self.n_latent),
                              ctx=CTX)
        latent_z = latent_mean + nd.exp(0.5 * latent_logvar) * eps
        
        # Use the decoder to generate output, then flatten it to compute loss
        x_hat_flattened = self.decoder(latent_z).reshape(x.shape[0], -1)
        # Flatten input image array for computing loss
        x_flattened = x.reshape(x.shape[0], -1)
        
        # Compute the KL_Divergence between latent variable and standard normal
        KL_div_loss = - 0.5 * nd.sum(1 + latent_logvar - latent_mean * latent_mean - nd.exp(latent_logvar),
                                   axis=1)
        
        # Compute the content loss that is the cross entropy between the original image 
        # and the generated image
        # Add 1e-10 to prevent log(0) from happening
        logloss = -nd.sum(x_flattened*nd.log(x_hat_flattened+1e-10)+
                          (1-x_flattened)*nd.log(1-x_hat_flattened+1e-10), axis=1)
        
        # Sum up the loss
        loss = KL_div_loss + logloss
        return loss
    
    def generate(self, x):
        # Because forward() returns the loss values, we still need a method that returns the generated image
        # Which is basically the forward process, up to (not including) the flattening of x_hat
        
        # x should be image arrays (4-dimensional) but encoder should be able 
        # to handle this so I am not going flatten it
        
        # Use the encoder network to compute the values of latent layers
        latent_layer = self.encoder(x)
        
        # Split the latent layer into latent means and latent log vars
        latent_mean = nd.split(latent_layer, axis=1, num_outputs = 2)[0]
        latent_logvar = nd.split(latent_layer, axis=1, num_outputs = 2)[1]
        
        # Use the reparametrization trick to ensure differentiability of the latent
        # variable
        eps = nd.random_normal(loc=0,
                               scale=1,
                               shape=(x.shape[0], self.n_latent),
                              ctx=CTX)
        latent_z = latent_mean + nd.exp(0.5 * latent_logvar) * eps
        
        # Use the decoder to generate output, then flatten it to compute loss
        return self.decoder(latent_z)
            
            
            