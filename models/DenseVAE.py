# Import the basic packages
import mxnet as mx
from mxnet import nd, gluon
from mxnet.gluon import nn, loss as gloss
import numpy as np
import d2l
CTX = d2l.try_gpu()

# This script provides a subclass of gluon.Block that is the
# VAE network. The implementation is identical to that of the demo
# provided in https://gluon.mxnet.io/chapter13_unsupervised-learning/vae-gluon.html

class DenseVAE(gluon.Block):
    
    def __init__(self, n_latent = 2,
                 n_hlayers = 3,
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
            self.decoder.add(nn.Dense(self.n_out_channels * self.out_width * self.out_height,
                                      activation='sigmoid'))
            
    def forward(self, x):
        # x is input of shape (n_batch, n_channels, width, height)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        
        
        # Get the latent layer
        latent_layer = self.encoder(x)
        
        # Split the latent layer into latent means and latent log vars
        latent_mean = nd.split(latent_layer, axis=1, num_outputs = 2)[0]
        latent_logvar = nd.split(latent_layer, axis=1, num_outputs = 2)[1]
        
        # Use the reparametrization trick to ensure differentiability of the latent
        # variable
        eps = nd.random_normal(loc=0,
                               scale=1,
                               shape=(batch_size, self.n_latent),
                              ctx=CTX)
        latent_z = latent_mean + nd.exp(0.5 * latent_logvar) * eps
        
        # Use the decoder to generate output
        x_hat = self.decoder(latent_z)        
        
        # Compute the KL_Divergence between latent variable and standard normal
        self.KL_div_loss = -0.5 * nd.sum(1 + latent_logvar - latent_mean * latent_mean - nd.exp(latent_logvar),
                                   axis=1)
        
        # Compute the content loss that is the cross entropy between the original image 
        # and the generated image
        # content_loss = gloss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)(x_hat, x.reshape(batch_size, -1))
        
        # Add 1e-10 to prevent log(0) from happening
        self.logloss = - nd.sum(x*nd.log(x_hat + 1e-10)+ (1-x)*nd.log(1-x_hat + 1e-10), axis=1)
        
        # Sum up the loss
        loss = self.KL_div_loss + self.logloss
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
        return self.decoder(latent_z).reshape(-1, self.n_out_channels, self.out_width, self.out_height)
            
            
            