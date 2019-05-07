# Import the basic packages
import mxnet as mx
from mxnet import nd, init, gluon, autograd, image
from mxnet.gluon import data as gdata, loss as gloss, nn
import numpy as np
import d2l
CTX = d2l.try_gpu()
import time
import matplotlib.pyplot as plt
import os
os.system('export MXNET_CUDNN_AUTOTUNE_DEFAULT=0')

# Import the ConvVAE and ResNet
import sys
sys.path.insert(0, "./models")
from ConvVAE import ConvVAE
from ResNet import ResNet

# Import the VAE_GAN training method
from train_VAE_GAN import train_VAE_GAN

##########################################################################################
## DATA PREPARATION
##########################################################################################
# Because training data iterator is defined in the train_VAE_GAN method
# we don't need to define it here
print("[STATE]: Loading data onto context")
print('[STATE]: Random seed chosen is 0')
mx.random.seed(0)
all_features = nd.load('../project_data/anime_faces.ndy')[0]
all_features = nd.shuffle(all_features)

# Use 80% of the data as training data
# since the anime faces have no particular order, just take the first
# 80% as training set
n_train = int(all_features.shape[0] * 0.8)
train_features = all_features[0:n_train]
test_features = all_features[n_train:]
batch_size = 64
_, n_channels, width, height = train_features.shape

##########################################################################################
## MODEL PREPARATION
##########################################################################################
# Instantiate the gluon.Block instances
# Do not initialize them or get trainers; initialization and trainer
# are done in the VAE_GAN_train method
n_latent = 1024
n_base_channels = 32
pbp_weight = 1
conv_vae = ConvVAE(n_latent=n_latent,
                   n_channels=n_channels,
                   out_width=width,
                   out_height=height,
                   n_base_channels=n_base_channels,
                  pbp_weight=pbp_weight)
resnet = ResNet(n_classes=1)

##########################################################################################
## ADDITIONAL TRAINING HYPERPARAMETERS
##########################################################################################
test_results_dir = './results/images/ConvVAE_ResNet_on_anime/1024_32_200_10_1_initlr2e-4/'
vae_parameters_path = '../project_data/model_parameters/ConvVAE_against_ResNet_1024_32_200_10_1_initlr2e-4.params'
n_epochs=200
n_solo_epochs=100
max_disc_loss=999
variable_pbp_weight='constant'
pbp_weight_decay = 0.95
constant_pbp_weight = 1
constant_disc_loss_mul = 10

##########################################################################################
## Training
##########################################################################################

train_VAE_GAN(vae_net = conv_vae,
              disc_net = resnet,
              train_features = train_features,
              test_features = test_features,
              test_results_dir = test_results_dir,
              vae_parameters_path = vae_parameters_path,
              batch_size = batch_size,
              init_lr = 0.0002,
              pbp_weight = constant_pbp_weight,
              disc_loss_mul = constant_disc_loss_mul,
              n_epochs = n_epochs,
              n_solo_epochs = n_solo_epochs,
              max_disc_loss = max_disc_loss,
              variable_pbp_weight = variable_pbp_weight,
              pbp_weight_decay = pbp_weight_decay,
              CTX = d2l.try_gpu())

# Print training statistics for verifying that training statistics is successfully generated
import pandas as pd
tr_stats = pd.read_csv(test_results_dir + 'training_statistics.csv')
print(tr_stats.shape)
print(tr_stats.describe())



                   
































