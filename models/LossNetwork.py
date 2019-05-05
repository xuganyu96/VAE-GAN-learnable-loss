import mxnet as mx
import numpy as np
from mxnet import gluon
from mxnet.gluon import data as gdata, loss as gloss, nn
from mxnet.gluon import nn, Block, HybridBlock, Parameter
import mxnet.ndarray as F
import utilities
import os


class LossNet(Block):
    def __init__(self, vgg, content_weight, style_weight, batch_size, loss=gluon.loss.L2Loss()):
        super(LossNet, self).__init__()
        self.vgg = vgg
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.batch_size = batch_size
        self.loss = loss

    def forward(self, original_img, generated_img):
        # normalization:
        x = utils.subtract_img_mean_batch(original_img)
        y = utils.subtract_img_mean_batch(generated_img)

        features_x = self.vgg(x)
        features_y = self.vgg(y)

        gram_style = [self.gram_matrix(feature) for feature in featured_x]

        # compute content loss:
        vgg_x = features_x[1]
        content_loss = self.content_weight * self.loss(features_y[1], vgg_x)

        # compute style loss:
        style_loss = 0.0
        for m in range(len(features_y)):
            gram_y = self.gram_matrix(features_y[m])
            _, C, _ = gram_y[m].shape
            gram_s = F.expand_dims(gram_style[m], 0).broadcast_to((self.batch_size, 1, C, C))
            style_loss += self.style_weight * self.loss(gram_y, gram_s[:self.batch_size, :, :])
        return content_loss + style_loss

    def gram_matrix(self, x):
        (b, ch, h, w) = x.shape
        features = x.reshape((b, ch, w*h))
        return F.batch_dot(features, features, transpose_b=True) / (ch*h*w)


class Vgg16(Block):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2D(in_channels=3, channels=64, kernel_size=3, strides=1, padding=1)
        self.conv1_2 = nn.Conv2D(in_channels=64, channels=64, kernel_size=3, strides=1, padding=1)

        self.conv2_1 = nn.Conv2D(in_channels=64, channels=128, kernel_size=3, strides=1, padding=1)
        self.conv2_2 = nn.Conv2D(in_channels=128, channels=128, kernel_size=3, strides=1, padding=1)

        self.conv3_1 = nn.Conv2D(in_channels=128, channels=256, kernel_size=3, strides=1, padding=1)
        self.conv3_2 = nn.Conv2D(in_channels=256, channels=256, kernel_size=3, strides=1, padding=1)
        self.conv3_3 = nn.Conv2D(in_channels=256, channels=256, kernel_size=3, strides=1, padding=1)

        self.conv4_1 = nn.Conv2D(in_channels=256, channels=512, kernel_size=3, strides=1, padding=1)
        self.conv4_2 = nn.Conv2D(in_channels=512, channels=512, kernel_size=3, strides=1, padding=1)
        self.conv4_3 = nn.Conv2D(in_channels=512, channels=512, kernel_size=3, strides=1, padding=1)

        self.conv5_1 = nn.Conv2D(in_channels=512, channels=512, kernel_size=3, strides=1, padding=1)
        self.conv5_2 = nn.Conv2D(in_channels=512, channels=512, kernel_size=3, strides=1, padding=1)
        self.conv5_3 = nn.Conv2D(in_channels=512, channels=512, kernel_size=3, strides=1, padding=1)

        self.dense1 = nn.Dense()

    def forward(self, X):
        h = F.Activation(self.conv1_1(X), act_type='relu')
        h = F.Activation(self.conv1_2(h), act_type='relu')
        relu1_2 = h
        h = F.Pooling(h, pool_type='max', kernel=(2, 2), stride=(2, 2))

        h = F.Activation(self.conv2_1(h), act_type='relu')
        h = F.Activation(self.conv2_2(h), act_type='relu')
        relu2_2 = h
        h = F.Pooling(h, pool_type='max', kernel=(2, 2), stride=(2, 2))

        h = F.Activation(self.conv3_1(h), act_type='relu')
        h = F.Activation(self.conv3_2(h), act_type='relu')
        h = F.Activation(self.conv3_3(h), act_type='relu')
        relu3_3 = h
        h = F.Pooling(h, pool_type='max', kernel=(2, 2), stride=(2, 2))

        h = F.Activation(self.conv4_1(h), act_type='relu')
        h = F.Activation(self.conv4_2(h), act_type='relu')
        h = F.Activation(self.conv4_3(h), act_type='relu')
        relu4_3 = h

        return [relu1_2, relu2_2, relu3_3, relu4_3]

    def init_vgg_params(self, model_folder, ctx, fixed_params=True):
        if not os.path.exists(os.path.join(model_folder, 'mxvgg.params')):
            os.system('wget https://www.dropbox.com/s/7c92s0guekwrwzf/mxvgg.params?dl=1 -O' + os.path.join(model_folder,
                                                                                                           'mxvgg.params'))
        self.collect_params().load(os.path.join(model_folder, 'mxvgg.params'), ctx=ctx)
        for param in self.collect_params().values():
            if fixed_params:
                param.grad_req = 'null'
            else:
                param.grad_req = 'write'

