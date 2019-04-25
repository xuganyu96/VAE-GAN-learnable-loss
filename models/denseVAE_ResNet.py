# Import the basic packages
import mxnet as mx
from mxnet import nd, init, gluon, autograd, image
from mxnet.gluon import data as gdata, loss as gloss, nn
import numpy as np
import d2l
import time
import matplotlib.pyplot as plt

# Import the discriminator network
# Note that this script will be run from the main directory
# from the training script, so all paths are relative
# to the main directory
import sys
sys.path.insert(0, "./models/discriminators")
from ResNet import ResNet

CTX = d2l.try_gpu()

# Define the GAN network architecture
class DenseVAE_ResNet(nn.Block):
    
    # The GAN network needs to know all of the VAE network's 
    # hyperparameters
    def __init__(self, n_latent = 2,
                 n_hlayers = 3,
                 n_hnodes = 400,
                 out_n_channels = 1,
                 out_width = 28,
                 out_height = 28):
        
        # Store the model hyperparameters
        self.n_latent = n_latent
        self.n_hlayers = n_hlayers
        self.n_hnodes = n_hnodes
        self.out_n_channels = out_n_channels
        self.out_width = out_width
        self.out_height = out_height
        
        # Initialize the super class
        super(DenseVAE_ResNet, self).__init__()
        
        # I will construct the network with encoder, decoder, and 
        # discriminator separately
        with self.name_scope():
            # Construct the encoder in the same way as in denseVAE
            self.encoder = nn.Sequential(prefix='encoder')
            
            # Add hidden dense layers to the encoder
            for i in range(self.n_hlayers):
                self.encoder.add(nn.Dense(self.n_hnodes, activation='relu'))
                
            # Add a final encoder layer that is the latent layer
            # latent layer will contain 2 * n_latent nodes the first n_latent
            # of which are latent means and the rest are latent log variances
            self.encoder.add(nn.Dense(2 * self.n_latent))
            
            # Construct the decoder
            self.decoder = nn.Sequential(prefix = 'decoder')
            
            # Add hidden dense layers to the decoder
            for i in range(self.n_hlayers):
                self.decoder.add(nn.Dense(self.n_hnodes, activation='relu'))
                
            # Add the final output layer; use sigmoid transformation to make
            # the color value ranges within [0, 1]
            self.decoder.add(nn.Dense(self.out_n_channels * self.out_width * self.out_height,
                                      activation='sigmoid'))
            
            # Construct the discriminator, which is just a ResNet
            # Because we are concerned with discriminating only the genuine images
            # from the generated images, we only need 1 class.
            self.discriminator = ResNet(n_classes = 1)
            
    # Define the forwarding process
    def forward(self, x, first_cycle = False):
        # input x is image and thus 4-dimensional ndarray
        batch_size, n_channels_in, input_width, input_height = x.shape
        
        # First run it through the encoder
        
        x_flattened = x.reshape(batch_size, -1)
        latent_layer = self.encoder(x_flattened)
        
        # Split latent layer into latent mean and latent log variances
        latent_mean = nd.split(latent_layer, axis=1, num_outputs = 2)[0]
        latent_logvar = nd.split(latent_layer, axis=1, num_outputs = 2)[1]
        
        # Compute the latent variable's value using the reparametrization trick
        eps = nd.random_normal(loc=0, scale=1, shape=(batch_size, self.n_latent))
        latent_z = latent_mean + nd.exp(0.5 * latent_logvar) * eps
        
        # At this point, also compute the KL_Divergence between latent variable and 
        # Gaussian(0, 1)
        KL_div_loss = -0.5 * nd.sum(1 + latent_logvar - latent_mean * latent_mean - nd.exp(latent_logvar),
                                   axis=1)
        
        # Run the latent variable through the decoder to get the flattened generated image
        x_hat_flattened = self.decoder(latent_z)
        
        # Inflate the flattened output to be fed into the discriminator
        x_hat = x_hat_flattened.reshape(batch_size, n_channels_in, input_width, input_height)
        
        # Content loss is given by the resnet
        # In later training process we will feed the discriminator genuine and generated images
        # with genuine images labeled 1 and generated images labeled 0
        # in this case a higher value in ResNet's output indicate higher confidence of
        # an image's realness; therefore we want to reduce the negative of the ResNet's output
        content_loss = - self.discriminator(x_hat)
        
        # For the first training cycle, resnet is completely not trained
        # so we will not use the resnet as a content loss metric; instead we will use
        # the logloss as a content loss
        if first_cycle:
            content_loss = nd.sum(x_flattened * nd.log(x_hat_flattened + 1e-10) +
                                  (1-x_flattened)*nd.log(1-x_hat_flattened + 1e-10), 
                                  axis=1)
        
        # Loss is the sum of KL_Divergence and the content loss
        loss = KL_div_loss + content_loss
        
        return loss
    
    def generate(self, x):
        # Repeat the process of forward, but stop at x_hat and return it
        # input x is image and thus 4-dimensional ndarray
        batch_size, n_channels_in, input_width, input_height = x.shape
        
        # First run it through the encoder
        
        x_flattened = x.reshape(batch_size, -1)
        latent_layer = self.encoder(x_flattened)
        
        # Split latent layer into latent mean and latent log variances
        latent_mean = nd.split(latent_layer, axis=1, num_outputs = 2)[0]
        latent_logvar = nd.split(latent_layer, axis=1, num_outputs = 2)[1]
        
        # Compute the latent variable's value using the reparametrization trick
        eps = nd.random_normal(loc=0, scale=1, shape=(batch_size, self.n_latent))
        latent_z = latent_mean + nd.exp(0.5 * latent_logvar) * eps
        
        # At this point, also compute the KL_Divergence between latent variable and 
        # Gaussian(0, 1)
        KL_div_loss = -0.5 * nd.sum(1 + latent_logvar - latent_mean * latent_mean - nd.exp(latent_logvar),
                                   axis=1)
        
        # Run the latent variable through the decoder to get the flattened generated image
        x_hat_flattened = self.decoder(latent_z)
        
        # Inflate the flattened output to be fed into the discriminator
        x_hat = x_hat_flattened.reshape(batch_size, n_channels_in, input_width, input_height)
        
        return x_hat
        
        
        
        
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            