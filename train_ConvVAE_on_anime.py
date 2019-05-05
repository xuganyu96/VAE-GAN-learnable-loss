# Import the basic packages
import mxnet as mx
from mxnet import nd, init, gluon, autograd, image
from mxnet.gluon import data as gdata, loss as gloss, nn
import numpy as np
import d2l
CTX = d2l.try_gpu()
import time
import matplotlib.pyplot as plt

# Import the DenseVAE model
import sys
sys.path.insert(0, "./models")
from ConvVAE import ConvVAE

# Set seed to 0 for consistent testing set images throughout run and run
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
# Extract the training image's shape
_, n_channels, width, height = train_features.shape

# Instantiate the model, then build the trainer and 
# initialize the parameters
n_latent = 512
n_base_channels = 32
conv_vae = ConvVAE(n_latent = n_latent,
                   n_channels = n_channels,
                   out_width = width,
                   out_height = height,
                   n_base_channels=n_base_channels)
conv_vae.collect_params().initialize(mx.init.Xavier(), ctx=CTX)
trainer = gluon.Trainer(conv_vae.collect_params(), 
                        'adam', 
                        {'learning_rate': .001})

# Specify the directory to which validation images and training
# report (with training errors and time for each epoch) will be
# saved
result_dir = './results/images/ConvVAE_on_anime/512_32_200_seeded/'

# Open a file to write to for training reports
readme = open(result_dir + 'README.md', 'w')
readme.write('Number of latent variables \t' + str(n_latent) + '\n\n')
readme.write('Number of base channels \t' + str(n_base_channels) + '\n\n')

n_epoch = 200
readme.write('Number of epochs trained \t' + str(n_epoch) + '\n\n')
for epoch in range(n_epoch):
    
    batch_losses = []
    epoch_start_time = time.time()
    
    for train_batch in train_iter:
        train_batch = train_batch.as_in_context(CTX)
        
        with autograd.record():
            loss = conv_vae(train_batch)
        loss.backward()
        trainer.step(train_batch.shape[0])
        batch_losses.append(nd.mean(loss).asscalar())
   
    epoch_train_loss = np.mean(batch_losses)
    epoch_stop_time = time.time()
    time_consumed = epoch_stop_time - epoch_start_time
    
    # Generate the epoch report, write it to the report file and print it
    epoch_report_str = 'Epoch{}, Training loss {:.10f}, Time used {:.2f}'.format(epoch,
                                                     epoch_train_loss,
                                                     time_consumed)
    readme.write(epoch_report_str + '\n\n')
    print(epoch_report_str)
    
# Define the number of validation images to generate (and display in the README.md)
n_validations = 10
    
# Validation
img_arrays = conv_vae.generate(test_features[0:n_validations].as_in_context(CTX)).asnumpy()



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
