# Import the basic packages
import mxnet as mx
from mxnet import nd, gluon
from mxnet.gluon import nn, loss as gloss
import numpy as np
import d2l
CTX = d2l.try_gpu()

# An implementation of Patch GAN 