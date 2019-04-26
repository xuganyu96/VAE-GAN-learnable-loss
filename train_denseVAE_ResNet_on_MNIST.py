# Import basic packages
import mxnet as mx
from mxnet import nd, init, gluon, autograd, image
from mxnet.gluon import data as gdata, loss as gloss, nn
import numpy as np
import d2l
CTX = d2l.try_gpu()
import time
import matplotlib.pyplot as plt

# Import the model
import sys
sys.path.insert(0, "./models")
from denseVAE_ResNet import DenseVAE_ResNet

# Import the ResNet block for a second fixable resnet
sys.path.insert(0, "./models/discriminators")
from ResNet import ResNet

# Get training features and train_iterators
mnist = mx.test_utils.get_mnist()
train_features = nd.array(mnist['train_data'], ctx=CTX)
batch_size = 64
train_iter = gdata.DataLoader(train_features,
                                  batch_size,
                                 shuffle=True,
                                 last_batch='keep')

# Get the model hyperparameters regarding the input data 
# shape
_, n_channels, width, height = train_features.shape

# Instantiate the model
vae_resnet = DenseVAE_ResNet(n_latent = 5,
                                 n_hlayers = 10,
                                 n_hnodes = 400,
                                 out_n_channels = n_channels,
                                 out_width = width,
                                 out_height = height)

# Instantiate a second ResNet to be used as a fixable discriminator 
# in the GAN training scheme
resnet_2 = ResNet(n_classes = 1)

# Initialize the VAE_ResNet net and the resnet
print("[STATE]: Initializing model parameters")
vae_resnet.initialize(mx.init.Xavier(), ctx=CTX)
resnet_2.initialize(mx.init.Xavier(), ctx=CTX)

# Specify the vae_resnet trainer and the resnet_2 trainer
vae_resnet_trainer = gluon.Trainer(vae_resnet.collect_params(),
                                   'adam',
                                   {'learning_rate': .001})
resnet_２_trainer = gluon.Trainer(resnet_2.collect_params(),
                                   'adam',
                                   {'learning_rate': .001})

# Specify training parameters
n_cycles = 10
n_epochs = 50

# Specify the discriminator training's scheme's loss function
disc_loss_func = gloss.SigmoidBinaryCrossEntropyLoss()

print("[STATE]: Ready to train")

for cycle in range(n_cycles):
    
    # Train the VAE
#     print("cycle " + str(cycle) + " Training VAE")
    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        batch_losses = []
        for train_batch in train_iter:
            train_batch = train_batch.as_in_context(CTX)

            with autograd.record():
                loss = vae_resnet.forward(train_batch,
                                              first_cycle = (cycle == 0))
            loss.backward()
            vae_resnet_trainer.step(train_batch.shape[0], ignore_stale_grad=True)
            batch_losses.append(nd.mean(loss).asscalar())

        epoch_train_loss = np.mean(batch_losses)
        epoch_stop_time = time.time()
        time_consumed = epoch_stop_time - epoch_start_time
        print('VAE, Cycle{}, Epoch{}, Training loss {:.10f}, Time used {:.2f}'.format(cycle,
                                                                                      epoch, 
                                                                       epoch_train_loss, 
                                                                       time_consumed))
        
    # Train the discriminator
    # Before we can train the discriminator we need to prepare our data set
    print("Concatenating discriminator training data")
    genuine_features = train_features
    generated_features = vae_resnet.generate(genuine_features)
    disc_training_features = nd.concat(genuine_features, generated_features, dim=0)
    # Generate the labels for genuine and fake images
    sample_size = genuine_features.shape[0]
    genuine_labels = nd.ones(shape=(sample_size,), ctx=CTX)
    generated_labels = nd.zeros(shape=(sample_size,), ctx=CTX)
    disc_training_labels = nd.concat(genuine_labels, generated_labels, dim=0)
    # Load the training features and training labels into ArrayDataset
    disc_training_set = gdata.ArrayDataset(disc_training_features, disc_training_labels)
    disc_train_iter = gdata.DataLoader(disc_training_set, batch_size * 2, shuffle=True)
    
    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        batch_losses = []
        for batch_features, batch_labels in disc_train_iter:
            # Load them into GPU
            batch_features = batch_features.as_in_context(CTX)
            batch_labels = batch_labels.as_in_context(CTX)

            with autograd.record():
                loss = disc_loss_func(resnet_2(batch_features), batch_labels)

            loss.backward()
            resnet_2_trainer.step(batch_features.shape[0])
            batch_losses.append(nd.mean(loss).asscalar())

        epoch_train_loss = np.mean(batch_losses)
        epoch_stop_time = time.time()
        time_consumed = epoch_stop_time - epoch_start_time
        print('ResNet, Cycle{}, Epoch{}, Training loss {:.10f}, Time used {:.2f}'.format(cycle,
                                                                                         epoch, 
                                                                       epoch_train_loss, 
                                                                       time_consumed))
        
    # At the end of this cycle, resnet_2 is properly trained, transfer the parameters to
    # vae_resnet_net's discriminator
    # First save resnet_2's parameters to file
    print("Transferring paramters")
    resnet_2.resnet.save_parameters('cache.params')
    # Then let vae_resnet_net's discriminator pick it up
    vae_resnet.discriminator.resnet.load_parameters('cache.params')
        
# Validation
img_arrays = vae_resnet.generate(nd.array(mnist['test_data'], ctx=CTX)).asnumpy()
# img_arrays = nd.array(mnist['test_data'], ctx=CTX).asnumpy()

for i in range(50):
    img_array = img_arrays[i]
    fig = plt.figure()
    plt.imshow(img_array.reshape(width, height))
    plt.savefig('./results/images/denseVAE_ResNet_MNIST/validation/' + str(i) + '.png')
    plt.close()

