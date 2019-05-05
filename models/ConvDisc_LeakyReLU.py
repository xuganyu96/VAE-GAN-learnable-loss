import mxnet as mx
from mxnet import nd, gluon, init
from mxnet.gluon import nn

class ConvDisc_LeakyReLU(gluon.Block):
    
    def __init__(self, n_classes=1,
                n_base_channels=32):
        
        super(ConvDisc_LeakyReLU, self).__init__()
        
        self.n_classes = n_classes
        self.n_base_channels = n_base_channels
        
        with self.name_scope():
            self.discriminator = nn.Sequential(prefix='discriminator')
            
            self.discriminator.add(nn.Conv2D(n_base_channels, 4, 2, 1, use_bias=False),
                                   nn.LeakyReLU(0.2))
            self.discriminator.add(nn.Conv2D(n_base_channels*2, 4, 2, 1, use_bias=False),
                                   nn.BatchNorm(),
                                   nn.LeakyReLU(0.2))
            self.discriminator.add(nn.Conv2D(n_base_channels*4, 4, 2, 1, use_bias=False),
                                   nn.BatchNorm(),
                                   nn.LeakyReLU(0.2))
            self.discriminator.add(nn.Dense(n_classes))
            
    def forward(self, x):
        # input x has shape
        # x.shape = (batch_size, n_channels, width, height)
        
        return self.discriminator(x)
            