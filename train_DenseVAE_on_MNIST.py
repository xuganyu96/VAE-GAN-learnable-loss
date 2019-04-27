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
from DenseVAE import DenseVAE

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

# Instantiate the model, then build the trainer and 
# initialize the parameters
n_latent = 5
n_hlayers = 3
n_hnodes = 400
dense_vae = DenseVAE(n_latent = n_latent,
                    n_hlayers = n_hlayers,
                    n_hnodes = n_hnodes,
                    n_out_channels = n_channels,
                    out_width = width,
                    out_height = height)
dense_vae.collect_params().initialize(mx.init.Xavier(), ctx=CTX)
trainer = gluon.Trainer(dense_vae.collect_params(), 
                        'adam', 
                        {'learning_rate': .001})

# Specify the directory to which validation images and training
# report (with training errors and time for each epoch) will be
# saved
result_dir = './results/images/DenseVAE_on_MNIST/5_3_400_40/'

# Open a file to write to for training reports
readme = open(result_dir + 'README.md', 'w')
readme.write('Number of latent variables \t' + str(n_latent) + '\n\n')
readme.write('Number of hidden layers \t' + str(n_hlayers) + '\n\n')
readme.write('Number of hidden nodes per layer \t' + str(n_hnodes) + '\n\n')

# Define the number of epochs
n_epoch = 40
readme.write('Number of epochs trained \t' + str(n_epoch) + '\n\n')
for epoch in range(n_epoch):
    
    batch_losses = []
    epoch_start_time = time.time()
    
    for train_batch in train_iter:
        train_batch = train_batch.as_in_context(CTX)
        
        with autograd.record():
            loss = dense_vae(train_batch)
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
    
# Validation and output validation images
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
