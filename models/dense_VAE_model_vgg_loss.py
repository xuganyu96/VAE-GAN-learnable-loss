import mxnet as mx
from mxnet import nd, init, gluon, autograd, image
from mxnet.gluon import data as gdata, loss as gloss, nn
import numpy as np
import d2l


CTX = d2l.try_gpu()

# This script provides a subclass of gluon.Block that is the
# VAE network. The implementation is identical to that of the demo
# provided in https://gluon.mxnet.io/chapter13_unsupervised-learning/vae-gluon.html

class dense_VAE(gluon.Block):
    
    def __init__(self, n_latent = 2,
                 n_hlayers = 3,
                 n_hnodes = 400,
                 out_n_channels = 1,
                 out_width = 28,
                 out_height = 28,
                 batch_size = 64,
                kernel_size = 5):
        
        # Store some hyperparameters
        self.n_latent = n_latent
        self.n_hlayers = n_hlayers
        self.n_hnodes = n_hnodes
        self.out_n_channels = out_n_channels
        self.out_width = out_width
        self.out_height = out_height
        self.batch_size = batch_size
        self.kernel_size = 5
        
        # Initialize the super class
        super(dense_VAE, self).__init__()
        
        # Define the networks: encoder and decoder
        with self.name_scope():
            self.encoder = nn.Sequential(prefix='encoder')
            
            # The input of encoder networks are images so they will be of the
            # shape (n_batch, n_channels, width, height)
            for i in range(n_hlayers):
                # Choice of number of channels is suggested in the VAE_GAN paper
                # the first hidden layer will have 2^3 channels
                # and each hidden_layer after will have double the number of 
                # channels of the previous hidden layer
                self.encoder.add(nn.Dense(self.n_hnodes, activation='relu'))
                
                
            # Finally add the output layer that is the latent variables
            # But keep 2 * latent nodes because the first n_latent of them are
            # the latent means and the second n_latent of them are 
            # log variances
            self.encoder.add(nn.Dense(2 * n_latent))
            
            # Define the decoder network
            self.decoder = nn.Sequential(prefix='decoder')
            
            # The input of decoder network is latent space NDArray of shape
            # (n_batch, n_latent)
            for i in range(n_hlayers):
                # Add Dense layers with BatchNorm() and Relu activation
                self.decoder.add(nn.Dense(self.n_hnodes, activation='relu'))
            # Add the output layer that is another Dense layer but with sigmoid
            # transformation to keep the values between (0, 1)
            self.decoder.add(nn.Dense(self.out_n_channels * self.out_width * self.out_height,
                                      activation='sigmoid'))
            
    def forward(self, x):
        # x is input of shape (n_batch, n_channels, width, height)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        
        
        # Get the latent layer
        latent_vals = self.encoder(x)
        
        # Split the latent layer into latent means and latent log vars
        latent_mean = nd.split(latent_vals, axis=1, num_outputs = 2)[0]
        latent_logvar = nd.split(latent_vals, axis=1, num_outputs = 2)[1]
        
        # Use the reparametrization trick to ensure differentiability of the latent
        # variable
        eps = nd.random_normal(loc=0,
                               scale=1,
                               shape=(batch_size, self.n_latent),
                              ctx=CTX)
        latent_z = latent_mean + nd.exp(0.5 * latent_logvar) * eps
        
        # Use the decoder to generate output
        x_hat = self.decoder(latent_z)        
        self.x_hat = x_hat
        
        # Use the vgg loss net to compute the loss
        loss = self.loss_net(x, x_hat)
        return loss
            
            
            