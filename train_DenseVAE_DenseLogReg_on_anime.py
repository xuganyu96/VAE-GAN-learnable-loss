# Import the basic packages
import mxnet as mx
from mxnet import nd, init, gluon, autograd, image
from mxnet.gluon import data as gdata, loss as gloss, nn
import numpy as np
import d2l
CTX = d2l.try_gpu()
import time
import matplotlib.pyplot as plt

# Import the DenseVAE and the DenseLogisticRegressor models
import sys
sys.path.insert(0, "./models")
from DenseVAE import DenseVAE
from DenseLogisticRegressor import DenseLogisticRegressor as DenseLogReg

# Prepare the training data and training data iterator
print("[STATE]: Loading data onto context")
mx.random.seed(0)
all_features = nd.load('../project_data/anime_faces.ndy')[0]
all_features = nd.shuffle(all_features)

# Use 80% of the data as training data
# since the anime faces have no particular order, just take the first
# 80% as training set
# Prepare the training data and training data iterator
n_train = int(all_features.shape[0] * 0.8)
train_features = all_features[0:n_train]
test_features = all_features[n_train:]
batch_size = 64
train_iter = gdata.DataLoader(train_features,
                                  batch_size,
                                 shuffle=True,
                                 last_batch='keep')
print("[STATE]: Data loaded onto context")

# Extract the training image's shape
_, n_channels, width, height = train_features.shape

# Instantiate the VAE model, then build the trainer and 
# initialize the parameters
n_latent = 512
n_hlayers = 5
n_hnodes = 1024
dense_vae = DenseVAE(n_latent = n_latent,
                    n_hlayers = n_hlayers,
                    n_hnodes = n_hnodes,
                    n_out_channels = n_channels,
                    out_width = width,
                    out_height = height)
dense_vae.collect_params().initialize(mx.init.Xavier(), ctx=CTX)
dense_vae_trainer = gluon.Trainer(dense_vae.collect_params(), 
                        'adam', 
                        {'learning_rate': .001})

# Instantiate the logistic regression model, initialize its paramters
# and instantiate the trainer instance
logreg_n_hlayers = 1
logreg_n_hnodes = 1024
logreg = DenseLogReg(n_hlayers = logreg_n_hlayers,
                    n_hnodes = logreg_n_hnodes)
logreg.collect_params().initialize(mx.init.Xavier(), ctx=CTX)
logreg_trainer = gluon.Trainer(logreg.collect_params(),
                               'adam',
                               {'learning_rate': 0.001})

# Define a discriminator amplifier constant that multiplies the discriminator
# loss in the generator training cycle
disc_loss_multiplier = 10

# Specify the directory to which validation images and training
# report (with training errors and time for each epoch) will be
# saved
result_dir = './results/images/DenseVAE_DenseLogReg_on_anime/512_5_1024_1_1024_200_10_seeded/'

# Open a file to write to for training reports
readme = open(result_dir + 'README.md', 'w')
readme.write('VAE number of latent variables \t' + str(n_latent) + '\n\n')
readme.write('VAE number of hidden layers \t' + str(n_hlayers) + '\n\n')
readme.write('VAE number of hidden nodes per layer \t' + str(n_hnodes) + '\n\n')
readme.write('LogReg number of hidden layers \t' + str(logreg_n_hlayers) + '\n\n')
readme.write('LogReg number of hidden nodes per layer \t' + str(logreg_n_hnodes) + '\n\n')

# Define the loss function for training the discriminator (the logreg)
disc_loss_func = gloss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)

# Define the number of epochs to train
n_epochs = 200
readme.write('Number of epochs trained \t' + str(n_epochs) + '\n\n')

print("[STATE]: Training started")
for epoch in range(n_epochs):
    
    # Start recording epoch training time
    start_time = time.time()
    
    # Initialize a list that records the average loss within each batch
    dense_vae_batch_losses = []
    logreg_batch_losses = []
    
    # Iterate through all possible batches
    for batch_features in train_iter:
        batch_features = batch_features.as_in_context(CTX)
        batch_size = batch_features.shape[0]
        
        # Generate the labels of 1 and 0s, with 1 representing an image
        # being genuine while the 0 representing an image being
        # generated
        genuine_labels = nd.ones((batch_size,), ctx=CTX)
        generated_labels = nd.zeros((batch_size,), ctx=CTX)
        
        ############################################################################
        # UPDATE THE DISCRIMINATOR NETWORK
        ############################################################################
        with autograd.record():
            
            # Train with genuine images: make predictions on genuine images
            genuine_logit_preds = logreg(batch_features)
            genuine_loss = disc_loss_func(genuine_logit_preds, genuine_labels)
            
            # Train with generated images: make predictions on generated images
            generated_features = dense_vae.generate(batch_features)
            generated_logit_preds = logreg(generated_features)
            generated_loss = disc_loss_func(generated_logit_preds, generated_labels)
            
            # Total loss is loss with genuine and with generated images
            disc_loss = genuine_loss + generated_loss
            disc_loss.backward()
            logreg_batch_losses.append(nd.mean(disc_loss).asscalar())
            
        # update the parameters in the logreg
        logreg_trainer.step(batch_size)
        
        ############################################################################
        # UPDATE THE VAE NETWORK
        ############################################################################
        with autograd.record():
            
#             # Make a pass on "forward", which will get the kl_div loss 
#             # and the logloss to be assigned into instance attributes
#             dense_vae.forward(batch_features)
#             batch_kl_div_loss = dense_vae.KL_div_loss
#             batch_pbp_loss = dense_vae.logloss
            
            # Compute the content loss by letting the logreg network make predictions
            # on the generated images
            generated_features = dense_vae.generate(batch_features)
            generated_logit_preds = logreg(generated_features)
            batch_disc_loss = disc_loss_func(generated_logit_preds, genuine_labels)
            
            # Sum up the VAE loss and the discriminator loss (with multiplier)
            gen_loss = dense_vae(batch_features) + batch_disc_loss * disc_loss_multiplier
            gen_loss.backward()
            dense_vae_batch_losses.append(nd.mean(gen_loss).asscalar())
            
        # Update the parameters in the dense vae
        dense_vae_trainer.step(batch_size)
            
    ############################################################################
    # NEAR THE END OF THIS EPOCH
    ############################################################################

    # Compute some summarical metrics of this epoch
    stop_time = time.time()
    time_consumed = stop_time - start_time
    epoch_logreg_train_loss = np.mean(logreg_batch_losses)
    epoch_dense_vae_train_loss = np.mean(dense_vae_batch_losses)
    
    # Generate the epoch report
    epoch_report = 'Epoch{}, VAE Training loss {:.5f}, LogReg Training loss {:.10f}, Time used {:.2f}'
    epoch_report = epoch_report.format(epoch,
                                       epoch_dense_vae_train_loss,
                                       epoch_logreg_train_loss,
                                       time_consumed)
    readme.write(epoch_report + '\n\n')
    print(epoch_report)
    
# Define the number of validation images to generate (and display in the README.md)
n_validations = 10
    
# Validation
img_arrays = dense_vae.generate(test_features[0:n_validations].as_in_context(CTX)).asnumpy()



for i in range(n_validations):
    # Add a line that writes to the report to display the images
    readme.write('!['+str(i)+'](./'+str(i)+'.png)')
    readme.write('!['+str(i)+'](./test_'+str(i)+'.png)')
    img_array = img_arrays[i]
    fig = plt.figure()
    plt.imshow(img_array.reshape(width, height, n_channels))
    plt.savefig(result_dir + str(i) + '.png')
    plt.close()
    plt.imshow(test_features[i].reshape((width, height, n_channels)).asnumpy())
    plt.savefig(result_dir + 'test_' + str(i) + '.png')
    plt.close()
    
readme.close()
        
        
    










































