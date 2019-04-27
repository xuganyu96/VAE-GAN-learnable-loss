# Import the basic packages
import mxnet as mx
from mxnet import nd, init, gluon, autograd, image
from mxnet.gluon import data as gdata, loss as gloss, nn
import numpy as np
import d2l
CTX = d2l.try_gpu()
import time
import matplotlib.pyplot as plt

# Import the DenseVAE and ResNet network classes
import sys
sys.path.insert(0, "./models")
from DenseVAE import DenseVAE
from ResNet import ResNet

# Prepare the training data and training data iterator
mnist = mx.test_utils.get_mnist()
train_features = nd.array(mnist['train_data'], ctx=CTX)
batch_size = 64
train_iter = gdata.DataLoader(train_features,
                                  batch_size,
                                 shuffle=True,
                                 last_batch='keep')


# Extract the training image's shape
_, n_channels, width, height = train_features.shape

# Instantiate the VAE model, then build the trainer and 
# initialize the parameters
n_latent = 2
n_hlayers = 10
n_hnodes = 400
dense_vae = DenseVAE(n_latent = n_latent,
                    n_hlayers = n_hlayers,
                    n_hnodes = n_hnodes,
                    n_out_channels = n_channels,
                    out_width = width,
                    out_height = height)
dense_vae.collect_params().initialize(mx.init.Normal(0.02), ctx=CTX)
vae_trainer = gluon.Trainer(dense_vae.collect_params(), 
                        'adam', 
                        {'learning_rate': .001})

# Instantiate the ResNet network, initialize it
# and build the trainer instance
resnet = ResNet(n_classes=1)
resnet.collect_params().initialize(mx.init.Normal(0.02), ctx=CTX)
resnet_trainer = gluon.Trainer(resnet.collect_params(),
                               'adam',
                               {'learning_rate': 0.01})

# The MXNet GAN implementation used a SigmoidBinaryCrossEntropyLOss
# I will write it out here and figure out what it means later
loss_func = gloss.SigmoidBinaryCrossEntropyLoss()

# Specify the directory to which validation images and training
# report (with training errors and time for each epoch) will be
# saved
result_dir = './results/images/DenseVAE_ResNet_on_MNIST/2_10_400_50/'
# Open a file to write to for training reports
readme = open(result_dir + 'README.md', 'w')
readme.write('Number of latent variables \t' + str(n_latent) + '\n\n')
readme.write('Number of hidden layers \t' + str(n_hlayers) + '\n\n')
readme.write('Number of hidden nodes per layer \t' + str(n_hnodes) + '\n\n')

n_epochs = 50
n_sub_epochs = 50
readme.write('Number of epochs trained \t' + str(n_epochs) + '\n\n')

for epoch in range(n_epochs):
    
    # Start recording trainig time
    start_time = time.time()
    # Initialize a list that records the average loss within
    # each batch
    vae_batch_losses = []
    resnet_batch_losses = []
    
    
    for batch_features in train_iter:
        batch_features = batch_features.as_in_context(CTX)
        batch_size = batch_features.shape[0]
        
        # Generate some real labels so the generated_features
        real_labels = nd.ones((batch_size,), ctx=CTX)
        fake_labels = nd.zeros((batch_size,), ctx=CTX)
        
        # According to the tutorial, we first update the discriminator network
        with autograd.record():
            # Train with genuine images
            disc_preds = resnet(batch_features)
            genuine_loss = loss_func(disc_preds, real_labels)
            
            # Train with generated images
            generated_features = dense_vae.generate(batch_features)
            disc_preds = resnet(generated_features)
            generated_loss = loss_func(disc_preds, fake_labels)
            
            # Total loss is loss with real and loss with fake images
            loss = genuine_loss + generated_loss
            loss.backward()
            resnet_batch_losses.append(nd.mean(loss).asscalar())
            
        # Update the discriminator trainer with batch_size
        resnet_trainer.step(batch_size)
        
    for batch_features in train_iter:
        batch_features = batch_features.as_in_context(CTX)
        batch_size = batch_features.shape[0]

        # Generate some real labels so the generated_features
        real_labels = nd.ones((batch_size,), ctx=CTX)

        # Update the generator network, which is the VAE
        with autograd.record():
            # Generate the fake images
            generated_features = dense_vae.generate(batch_features)
            # Get the predicted logits from resnet
            disc_preds = resnet(generated_features)
            # Compute the content loss using the prediction from 
            # the ResNet
            content_loss = loss_func(disc_preds, real_labels)

            # Make a pass on the VAE network to get the KL Divergence loss
            dense_vae.forward(batch_features)
            kl_div_loss = dense_vae.kl_div_loss

            # Total loss is loss with content and loss with kl_div between 
            # latent variable and standard normal
            loss = content_loss #+ kl_div_loss
            loss.backward()
            vae_batch_losses.append(nd.mean(loss).asscalar())

        # Update the generator (VAE) network
        vae_trainer.step(batch_size)
    
    # End of an epoch, counting batch losses and time used
    stop_time = time.time()
    epoch_vae_train_loss = np.mean(vae_batch_losses)
    epoch_resnet_train_loss = np.mean(resnet_batch_losses)
    time_consumed = stop_time - start_time
    
    # Generate the epoch report, write it to the report file and print it
    epoch_report_str = 'Epoch{}, VAE Training loss {:.5f}, ResNet Training loss {:.10f}, Time used {:.2f}'.format(epoch,
                                                                                                    epoch_vae_train_loss,
                                                                                                    epoch_resnet_train_loss,
                                                                                                    time_consumed)
    readme.write(epoch_report_str + '\n\n')
    print(epoch_report_str)
    
# Validation
img_arrays = dense_vae.generate(nd.array(mnist['test_data'], ctx=CTX)).asnumpy()

# Define the number of validation images to generate (and display in the README.md)
n_validations = 10

for i in range(n_validations):
    # Add a line that writes to the report to display the images
    readme.write('!['+str(i)+'](./'+str(i)+'.png)')
    img_array = img_arrays[i]
    fig = plt.figure()
    plt.imshow(img_array.reshape(width, height))
    plt.savefig(result_dir + str(i) + '.png')
    plt.close()
    
readme.close()

