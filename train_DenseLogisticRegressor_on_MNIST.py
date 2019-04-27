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
from DenseLogisticRegressor import DenseLogisticRegressor

# Prepare the training data and training data iterator
# Note that we are going to train on only the images of digits 0 and 1
mnist = mx.test_utils.get_mnist()
train_features = nd.array(mnist['train_data'][mnist['train_label'] < 2], ctx=CTX)
train_labels = nd.array(mnist['train_label'][mnist['train_label'] < 2], ctx=CTX)
batch_size = 64
train_dataset = gdata.ArrayDataset(train_features, train_labels)
train_iter = gdata.DataLoader(train_dataset,
                                  batch_size,
                                 shuffle=True,
                                 last_batch='keep')

# Instantiate the model, initialize the parameters, and instantiate the trainer
log_reg = DenseLogisticRegressor(n_hlayers = 1)
log_reg.collect_params().initialize(mx.init.Xavier(), ctx=CTX)
log_reg_trainer = gluon.Trainer(log_reg.collect_params(),
                                'adam',
                                {'learning_rate': 0.001})

# Specify loss function; in this case because we use a single class
# for the logistic regression, we will use SigmoidBinaryCrossEntropyLoss()
# with from_sigmoid = False
loss_func = gloss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)

# Specify the number of epochs and start training
n_epochs = 50
for epoch in range(n_epochs):
    
    batch_losses = []
    
    for batch_features, batch_labels in train_iter:
        batch_features = batch_features.as_in_context(CTX)
        batch_labels = batch_labels.as_in_context(CTX)
        
        with autograd.record():
            logit_preds = log_reg(batch_features)
            loss = loss_func(logit_preds, batch_labels)
            batch_losses.append(nd.mean(loss).asscalar())
            loss.backward()
            
        log_reg_trainer.step(batch_features.shape[0])
        
    epoch_train_loss = np.mean(batch_losses)
    train_preds = nd.round(nd.sigmoid(log_reg(train_features))).reshape((-1,))
    train_acc = nd.mean(train_preds == train_labels).asscalar()
    
    epoch_report_str = 'Epoch{}, Training loss {:.10f}, Training accuracy {:.3f}'.format(epoch,
                                                                                 epoch_train_loss,
                                                                                 train_acc)
    print(epoch_report_str)
        
        
        
        