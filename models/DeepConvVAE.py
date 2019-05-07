# Import the basic packages
import mxnet as mx
from mxnet import nd, gluon
from mxnet.gluon import nn, loss as gloss
import numpy as np
import d2l
CTX = d2l.try_gpu()

# A variational autoencoder whose autoencoder uses convolutional 
# layers

class DeepConvVAE(gluon.Block):
    
    def __init__(self, n_latent = 5,
                n_channels = 3,
                out_width = 64,
                out_height = 64,
                n_base_channels = 16,
                pbp_weight=1):
        super(DeepConvVAE, self).__init__()
        
        # Record the model hyperparameters
        self.n_latent = n_latent
        self.n_channels = n_channels
        self.out_width = out_width
        self.out_height = out_height
        self.n_base_channels = n_base_channels
        self.pbp_weight = pbp_weight
        
        
        # Construct the encoder and decoder network
        with self.name_scope():
            
            # Construct the encoder network
            self.encoder = nn.Sequential(prefix='encoder')
            # Add convolution layers with increasing number of channels
            self.encoder.add(nn.Conv2D(n_base_channels * 1, kernel_size=4, use_bias=False),
                             nn.BatchNorm(),
                             nn.Activation('relu'))
            self.encoder.add(nn.Conv2D(n_base_channels * 2, kernel_size=4, use_bias=False),
                             nn.BatchNorm(),
                             nn.Activation('relu'))
#             self.encoder.add(nn.Conv2D(n_base_channels * 4, kernel_size=4, use_bias=False),
#                              nn.BatchNorm(),
#                              nn.Activation('relu'))
            # Add a final output layer that is 2 times the number of 
            # latent variables
            self.encoder.add(nn.Dense(2 * n_latent),
                             nn.BatchNorm(),
                             nn.Activation('relu'))
            
            # Construct the decoder network
            # For the decoder network, its input is 4-dimensional arrays
            # of shape (batch_size, n_latent, 1, 1)
            # Decoder's architecture came from Deep Convolutional
            # Generative Adversarial Network tutorial from MXNet
            self.decoder = nn.Sequential(prefix='decoder')
            # Add convolution layers with decreasing number of channels
            self.decoder.add(nn.Conv2DTranspose(64*8, 4, 1, 0, use_bias=False),
                             nn.BatchNorm(),
                             nn.Activation('relu'))
            self.decoder.add(nn.Conv2DTranspose(64*4, 4, 2, 1, use_bias=False),
                             nn.BatchNorm(),
                             nn.Activation('relu'))
            self.decoder.add(nn.Conv2DTranspose(64*2, 4, 2, 1, use_bias=False),
                             nn.BatchNorm(),
                             nn.Activation('relu'))
            self.decoder.add(nn.Conv2DTranspose(64*1, 4, 2, 1, use_bias=False),
                             nn.BatchNorm(),
                             nn.Activation('relu'))
            # The last layer uses sigmoid because I want image pixel data between 0 and 1
            # if it doesn't work I will use tanh as the tutorial did
            self.decoder.add(nn.Conv2DTranspose(self.n_channels, 4, 2, 1, use_bias=False),
                             nn.Activation('sigmoid'))
            
    def forward(self, x):
        # Because this encoder decoder setup uses convolutional layers 
        # There is no need to flatten anything
        # x.shape = (batch_size, n_channels, width, height)
        
        # Get the latent layer
        latent_layer = self.encoder(x)
        
        # Split the latent layer into latent means and latent log vars
        latent_mean = nd.split(latent_layer, axis=1, num_outputs=2)[0]
        latent_logvar = nd.split(latent_layer, axis=1, num_outputs=2)[1]
        
        # Compute the latent variable with reparametrization trick applied
        eps = nd.random_normal(0, 1, shape=(x.shape[0], self.n_latent), ctx=CTX)
        latent_z = latent_mean + nd.exp(0.5 * latent_logvar) * eps
        
        # Compute the KL Divergence between latent variable and standard normal
        kl_div_loss = -0.5 * nd.sum(1 + latent_logvar - latent_mean * latent_mean - nd.exp(latent_logvar),
                                         axis=1)
        
        # Use the decoder to generate output
        x_hat = self.decoder(latent_z.reshape((x.shape[0], self.n_latent, 1, 1)))
        
        # Compute the pixel-by-pixel loss; this requires that x and x_hat be flattened
        x_flattened = x.reshape((x.shape[0], -1))
        x_hat_flattened = x_hat.reshape((x_hat.shape[0], -1))
        logloss = - nd.sum(x_flattened*nd.log(x_hat_flattened + 1e-10) +
                                (1-x_flattened)*nd.log(1-x_hat_flattened+1e-10),
                                axis=1)
        
        # Sum up the loss
        loss = kl_div_loss + logloss * self.pbp_weight
        
        return loss
    
    def generate(self, x):
        # Generate an image given the input
        # input is
        # x.shape = (batch_size, n_channels, width, height)
        
        # Get the latent layer
        latent_layer = self.encoder(x)
        
        # Split the latent layer into latent means and latent log vars
        latent_mean = nd.split(latent_layer, axis=1, num_outputs=2)[0]
        latent_logvar = nd.split(latent_layer, axis=1, num_outputs=2)[1]
        
        # Compute the latent variable with reparametrization trick applied
        eps = nd.random_normal(0, 1, shape=(x.shape[0], self.n_latent), ctx=CTX)
        latent_z = latent_mean + nd.exp(0.5 * latent_logvar) * eps
        
        # Use the decoder to generate output
        x_hat = self.decoder(latent_z.reshape((x.shape[0], self.n_latent, 1, 1)))
        return x_hat
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
