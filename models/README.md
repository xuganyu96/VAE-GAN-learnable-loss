# Specification of network components

For simplier deployment, the three components of our VAE-GAN network are to be separately written in sub-modules. Here are the specifications of individual network components.

## Encoders
Encoders must be subclasses of `mxnet.gluon.nn.Block` with a `__init__()` method and a `forward(self, x)` method implemented.  

**__init__()** method must contain a section of code that is as follows:
```python
with self.name_scope():
    self.encoder = nn.Sequential(prefix='encoder')
    # continuing code that builds the encoder network
```

**I/O Shapes** must be as follows: an encoder must be able to accept image arrays (4-dimensional NDArray object whose shape is `(batch_size, n_channels, width, height)`) and output a latent layer with twice as many nodes as there are latent variables to accommodate latent mean and latent log variances

**forward()** method must return the latent layer that is of the shape `(batch_size, 2 * n_latent)`

## Decoders
Decoders must be subclasses of `mxnet.gluon.nn.Block` with a `__init__()` method, a `forward(self, x)` method implemented.

**__init__()** method must contain a section of code that is as follows:
```python
with self.name_scope():
    self.decoder = nn.Sequential(prefix='decoder')
    # continuing code that builds the decoder network
    #
    # The final dense layer that is the output layer must have a sigmoid activation
    # to make sure output values are between 0 and 1, inclusively
```

**I/O Shapes** must be as follows: a decoder must be able to accept latent variable array of the shape (batch_size, n_latent) and output an image array (4-dimensional NDArray object whose shape is `(batch_size, n_channels, width, height)`); the output is to be returned by `forward()`.

**forward()** method must return the image array that is of the shape `(batch_size, n_channels, width, height)`

## VAE
VAE must be subclasses of `mxnet.gluon.nn.Block` with a `__init__()` method, a `forward(self, x)` method, and a `generate(self, x)` method implemented.

**__init__()** method must contain a section of code that is as follows:
```python
with self.name_scope():
    self.encoder = Encoder()
    self.decoder = Decoder()
    # for some choices of Encoder and Decoder blocks
```

**I/O Shapes** must accept image arrays and return image arrays

**forward()** method must return the loss of the VAE

**generate()** method must return the generated image array using the VAE