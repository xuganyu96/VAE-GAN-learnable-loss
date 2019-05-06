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
from ConvDecoder import ConvDecoder
from ResNet import ResNet

# Prepare the training data and training data iterator
print("[STATE]: Loading data onto context")
print('[STATE]: Random seed chosen is 0')
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
print("[STATE]: Data loaded onto context: {}".format(CTX))

# Extract the training image's shape
_, n_channels, width, height = train_features.shape

# Instantiate the decoder network, then build the trainer and 
# initialize the parameters
n_latent = 512
n_base_channels = 32
conv_dec = ConvDecoder(n_latent=n_latent,
                       n_channels=n_channels,
                       out_width=width,
                       out_height=height,
                       n_base_channels=n_base_channels)
conv_dec.collect_params().initialize(mx.init.Xavier(), ctx=CTX)
conv_dec_trainer = gluon.Trainer(conv_dec.collect_params(),
                                 'adam',
                                 {'learning_rate': 0.001})

# Instantiate the ResNet discriminator model, initialize its paramters
# and instantiate the trainer instance
resnet = ResNet(n_classes = 1)
resnet.collect_params().initialize(mx.init.Xavier(), ctx=CTX)
# conv_disc = conv_disc.cast('float16')
resnet_trainer = gluon.Trainer(resnet.collect_params(),
                               'adam',
                               {'learning_rate': 0.001})

# Specify the directory to which validation images and training
# report (with training errors and time for each epoch) will be
# saved
test_results_dir = './results/images/ConvDecoder_ResNet_on_anime/512_32_200/'

# Define training hyperparamteres:
n_epochs = 200
disc_loss_func = gloss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)

#############################################################################
## Output file writer initialization
#############################################################################
#
# Open a file to write to for training statistics; the training statistics csv
# will be written to the results directory
try:
    csv_writer = open(test_results_dir + 'training_statistics.csv', 'w')
    print('[STATE]: Writing training statistics to ' + test_results_dir + 'training_statistics.csv')
except:
    print('[ERROR]: test results directory not valid, writing training statistics to main directory')
    csv_writer = open('./training_statistics.csv', 'w')
# CSV file needs to open with a header that is the column names
csv_writer.write('epoch,gen_loss,disc_loss,time_consumed\n')

# Open a file to write README.md for displaying validation images; the README
# file will be written to the results directory
try:
    readme_writer = open(test_results_dir + 'README.md', 'w')
    print('[STATE]: Writing README report to ' + test_results_dir + 'README.md')
except:
    print('[ERROR]: test results directory not valid, writing readme to main directory')
    csv_writer = open('./README.md', 'w')
# Write a few lines on README to indicate the hyper parameters
readme_writer.write('n_latent:{} \n\n'.format(conv_dec.n_latent))
readme_writer.write('n_base_channels:{} \n\n'.format(conv_dec.n_base_channels))
readme_writer.write('n_epochs:{} \n\n'.format(n_epochs))

print('[STATE]: Training started')
for epoch in range(n_epochs):
    start_time = time.time()
    
    # Initialize a list for recording the average loss within each batch
    gen_batch_losses = []
    disc_batch_losses = []
    
    # Iterate through all batches
    for batch_features in train_iter:
        # Load the batch into the appropriate context
        batch_features = batch_features.as_in_context(CTX)
        # Record the actual batch size; this may not be the same the specified
        # desired batch size
        act_batch_size = batch_features.shape[0]
        
        # Generate 1s and 0s as labels for genuine and generated images
        genuine_labels = nd.ones((act_batch_size,), ctx=CTX)
        generated_labels = nd.zeros((act_batch_size,), ctx=CTX)
        
        ############################################################################
        # UPDATE DISCRIMINATOR NETWORK
        ############################################################################
        with autograd.record():
            
            # Train with genuine images
            genuine_logit_preds = resnet(batch_features)
            genuine_loss = disc_loss_func(genuine_logit_preds,
                                          genuine_labels)
            
            # Train with generated images; generated images are generated from
            # random_normal(0, 1) and the decoder network
            latent_z = nd.random_normal(0, 1, shape=(act_batch_size, n_latent, 1, 1), ctx=CTX)
            generated_features = conv_dec(latent_z)
            generated_logit_preds = resnet(generated_features)
            generated_loss = disc_loss_func(generated_logit_preds,
                                            generated_labels)
            
            # Loss of discriminator cycle is sum of genuine loss and generated loss
            disc_loss = genuine_loss + generated_loss
            disc_loss.backward()
            disc_batch_losses.append(nd.mean(disc_loss).asscalar())
        
        # Update the parameters of the discriminator network
        resnet_trainer.step(act_batch_size)
        
        ############################################################################
        # UPDATE GENERATOR NETWORK
        ############################################################################
        with autograd.record():
            
            # Compute the discriminator loss by letting the discriminator make
            # predictions on the generated images
            latent_z = nd.random_normal(0, 1, shape=(act_batch_size, n_latent, 1, 1), ctx=CTX)
            generated_features = conv_dec(latent_z)
            generated_logit_preds = resnet(generated_features)
            gen_loss = disc_loss_func(generated_logit_preds,
                                      genuine_labels)
            gen_loss.backward()
            gen_batch_losses.append(nd.mean(gen_loss).asscalar())
            
        # Update the generator network's parameters
        conv_dec_trainer.step(act_batch_size)
        
    ############################################################################
    # NEAR THE END OF AN EPOCH
    ############################################################################
    # We have gone through all the batches of this epoch
    # Compute the summarical metrics
    stop_time = time.time()
    time_consumed = stop_time - start_time
    epoch_disc_train_loss = np.mean(disc_batch_losses)
    epoch_gen_train_loss = np.mean(gen_batch_losses)
    
    # Generate the README line and the csv line, and write them
    epoch_README_report = 'Epoch{}, ConvDecoder Training loss {:.5f}, ResNet Training loss {:.10f}, Time used {:.2f}'
    epoch_README_report = epoch_README_report.format(epoch,
                                                     epoch_gen_train_loss,
                                                     epoch_disc_train_loss,
                                                     time_consumed)
    epoch_CSV_report = '{},{:.10f},{:.10f},{:.2f}'.format(epoch,
                                                          epoch_gen_train_loss,
                                                          epoch_disc_train_loss,
                                                          time_consumed)
    readme_writer.write(epoch_README_report + '\n\n')
    csv_writer.write(epoch_CSV_report + '\n')
    print(epoch_README_report)
    
############################################################################
# END OF TRAINING
############################################################################
# Close the CSV writer because there is nothing left to write
csv_writer.close()

# Save model parameters; if vae_parameters_path is not valid, do not save it
conv_dec.save_parameters('../project_data/model_parameters/ConvDecoder_against_ResNet_512_32.params')
resnet.save_parameters('../project_data/model_parameters/ResNet_agaisnt_ConvDecoder_512_32.params')

# VALIDATION
# Define the number of validation images to generate
# then use the vae_net to generate them
n_validations = 10
latent_z = nd.random_normal(0, 1, shape=(n_validations, n_latent, 1, 1))
img_arrays = conv_dec(latent_z.as_in_context(CTX)).asnumpy()

for i in range(n_validations):
    # Write a line in the README report the displaying the generated images
    readme_writer.write('!['+str(i)+'](./'+str(i)+'.png)')

    # Reshape the output from (n_channels, width, height) to (width, height, n_channels)
    # Note that the vae_net instance already has such information regarding
    # the training images
    img_array = img_arrays[i].reshape((conv_dec.out_width,
                                    conv_dec.out_height,
                                    conv_dec.n_channels))

    # Show the plot, save it. If test_results_dir is not valid,
    # save it to main directory
    plt.imshow(img_array)
    try:
        plt.savefig(test_results_dir + str(i) + '.png')
        print('[STATE]: ' + test_results_dir + str(i) + '.png' + ' saved')

    except:
        print('[ERROR]: test results directory not valid, saving images to main directory')
        plt.savefig('./' + str(i) + '.png')
    plt.close()

# Close the README writer
readme_writer.close()
    
            
    

























