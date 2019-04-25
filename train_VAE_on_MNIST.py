import sys
sys.path.insert(0, "./models")
sys.path.insert(0, "./data")
from dense_VAE_model import dense_VAE
import mxnet as mx
from mxnet import nd, init, gluon, autograd, image
from mxnet.gluon import data as gdata, loss as gloss, nn
import numpy as np
import d2l
from conv_VAE_model import conv_VAE
from dense_VAE_model import dense_VAE
import time
import matplotlib.pyplot as plt

CTX = d2l.try_gpu()

mnist = mx.test_utils.get_mnist()
train_features = nd.array(mnist['train_data'], ctx=CTX)

_, n_channels, width, height = train_features.shape
batch_size = 64

vae_net = dense_VAE(n_latent = 5,
                    n_hlayers = 10,
                    n_hnodes = 400,
                    out_n_channels = n_channels,
                    out_width = width,
                    out_height = height)

vae_net.collect_params().initialize(mx.init.Xavier(), ctx=CTX)
trainer = gluon.Trainer(vae_net.collect_params(), 
                        'adam', 
                        {'learning_rate': .001})
train_iter = gdata.DataLoader(train_features,
                                  batch_size,
                                 shuffle=True,
                                 last_batch='keep')


n_epoch = 50
for epoch in range(n_epoch):
    
    batch_losses = []
    epoch_start_time = time.time()
    
    for train_batch in train_iter:
        train_batch = train_batch.as_in_context(CTX)
        
        with autograd.record():
            loss = vae_net(train_batch)
#             print(loss)
        loss.backward()
        trainer.step(train_batch.shape[0])
        batch_losses.append(nd.mean(loss).asscalar())
   
    epoch_train_loss = np.mean(batch_losses)
    epoch_stop_time = time.time()
    time_consumed = epoch_stop_time - epoch_start_time
    
    print('Epoch{}, Training loss {:.10f}, Time used {:.2f}'.format(epoch, 
                                                                   epoch_train_loss, 
                                                                   time_consumed))
    
# END OF TRAINING, save the model paramters
param_filename = 'MNIST_5_10_400_50.params'
vae_net.save_parameters('./results/params/' + param_filename)
    
# Validation
vae_net(nd.array(mnist['test_data'], ctx=CTX))
img_arrays = vae_net.x_hat.reshape(-1, width, height, n_channels).asnumpy()

for i in range(50):
    img_array = img_arrays[i]
    fig = plt.figure()
    plt.imshow(img_array.reshape(width, height))
    plt.savefig('./results/images/dense_VAE_l2/validation/' + str(i) + '.png')
    plt.close()
