import mxnet as mx
from mxnet import nd, gluon, init
from mxnet.gluon import nn

# The residual block implementation is directly taken from 
# D2L's website at http://d2l.ai/chapter_convolutional-modern/resnet.html
class Residual(nn.Block):
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1,
                               strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1,
                                   strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def forward(self, X):
        Y = nd.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return nd.relu(Y + X)
    
# The implementation of a ResNet is taken from the D2L website
# at http://d2l.ai/chapter_convolutional-modern/resnet.html
class ResNet(nn.Block):
    def __init__(self, n_classes = 1):
        super(ResNet, self).__init__()
        
        # This method is copied and pasted from the website
        # It is really sad how I need to implement nested method but I don't know
        # of better ways
        def resnet_block(num_channels, num_residuals, first_block=False):
            blk = nn.Sequential()
            for i in range(num_residuals):
                if i == 0 and not first_block:
                    blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
                else:
                    blk.add(Residual(num_channels))
            return blk
        
        
        with self.name_scope():
            # Construct the network
            self.resnet = nn.Sequential()
            
            # As specified on the website we begin with a 7*7 convolution
            # with 64 output channels, followed by batch normalization,
            # relu, and 3 * 3 max pooling with strides 2
            self.resnet.add(nn.Conv2D(64, kernel_size = 7,
                                      strides = 2,
                                      padding = 3),
                            nn.BatchNorm(),
                            nn.Activation('relu'),
                            nn.MaxPool2D(pool_size = 3,
                                         strides = 2,
                                         padding = 1))
            
            # Then we add the residual blocks defined by resnet_block
            self.resnet.add(resnet_block(64, 2, first_block=True),
                            resnet_block(128, 2),
                            resnet_block(256, 2),
                            resnet_block(512, 2))
            
            # Finally add global pooling and Dense layer (for classification)
            # 
            self.resnet.add(nn.GlobalAvgPool2D(), nn.Dense(n_classes))
            
    def forward(self, x):
        # x is image NDArray that is
        # x.shape = (batch_size, n_channels, width, height)
        
        return self.resnet(x)
        