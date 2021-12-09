# https://github.com/manicman1999/StyleGAN2-Tensorflow-2.0/blob/master/conv_mod.py

import tensorflow as tf
from tensorflow.keras import layers

class ColorLayer(layers.Layer):
  def __init__(self, filters, **kwargs):
    super(ColorLayer, self).__init__(**kwargs)
    self.filters = filters
    self.kernel_size = (3, 3)
    self.input_spec = [layers.InputSpec(ndim = 4), layers.InputSpec(ndim = 4)]
  
  def build(self, input_shape):
    channel_axis = -1
    if input_shape[0][channel_axis] is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    
    input_dim = input_shape[0][channel_axis]
    kernel_shape = self.kernel_size + (input_dim, self.filters)

    if input_shape[1][-1] != input_dim:
      raise ValueError('The last dimension of modulation input should be equal to input dimension.')
    
    self.kernel = self.add_weight(shape=kernel_shape,
                                  initializer="he_uniform",
                                  name='kernel')
    
    self.input_spec = [layers.InputSpec(ndim=4, axes={channel_axis: input_dim}), layers.InputSpec(ndim=4)]

    self.built = True
  
  def call(self, inputs):
    # from channels last to channels after batch BxHxWxC to BxCxHxW
    x = tf.transpose(inputs[0], [0, 3, 1, 2])

    # style weigths Bx1x1xinput_dim to Bx1x1xinput_dimx1
    w = tf.expand_dims(inputs[1], axis = -1)

    # add batch to weights
    wo = tf.expand_dims(self.kernel, axis = 0)

    # modulate
    weights = wo * (w+1) # Bx3x3xinput_dimxfilters

    # demodulate

    demodulate = tf.sqrt(tf.reduce_sum(tf.square(weights), axis=[1,2,3], keepdims = True) + 1e-8)
    weights = weights / demodulate # Bx3x3xinput_dimxfilters

    # from BxCxHxW to 1, B*C ,H, W
    x = tf.reshape(x, [1, -1, x.shape[2], x.shape[3]])

    # from B x 3 x 3 x input_dim x filters to 3 x 3 x input_dim x B x filters to 3 x 3 x input_dim x B * filters
    w = tf.reshape(tf.transpose(weights, [1, 2, 3, 0, 4]), [weights.shape[1], weights.shape[2], weights.shape[3], -1])

    x = tf.nn.conv2d(x, w, strides=(1, 1), padding="SAME", data_format="NCHW")

    # from 1, B*C, H, W to B, C, H, W
    x = tf.reshape(x, [-1, self.filters, x.shape[2], x.shape[3]])

    # B, C, H, W to B, H, W, C
    x = tf.transpose(x, [0, 2, 3, 1])

    return x

class ResBlock(tf.keras.Model):
  def __init__(self, filters):
    super(ResBlock, self).__init__()
    self.conv_1 = layers.Conv2D(filters, (3, 3), strides=(1, 1), padding="SAME")
    self.conv_2 = layers.Conv2D(filters, (3, 3), strides=(1, 1), padding="SAME")
    self.norm_1 = layers.BatchNormalization()
    self.norm_2 = layers.BatchNormalization()
  
  def call(self, input_x):
    x = self.conv_1(input_x)
    x = self.norm_1(x)
    x = layers.ReLU()(x)
    x = self.conv_2(x)
    x = self.norm_2(x)

    x = layers.add([x, input_x])
    x = layers.ReLU()(x)

    return x

class ResBlockShortCut(tf.keras.Model):
  def __init__(self, filters):
    super(ResBlockShortCut, self).__init__()
    self.conv_1 = layers.Conv2D(filters, (3, 3), strides=(1, 1), padding="SAME")
    self.conv_2 = layers.Conv2D(filters, (3, 3), strides=(1, 1), padding="SAME")
    self.norm_1 = layers.BatchNormalization()
    self.norm_2 = layers.BatchNormalization()

    self.conv = layers.Conv2D(filters, (1, 1), strides=(1, 1))
  
  def call(self, input_x):
    x = self.conv_1(input_x)
    x = self.norm_1(x)
    x = layers.ReLU()(x)
    x = self.conv_2(x)
    x = self.norm_2(x)

    input_x = self.conv(input_x)

    x = layers.add([x, input_x])
    x = layers.ReLU()(x)

    return x

class DownSamplingBlock(tf.keras.Model):
  def __init__(self, filters):
    super(DownSamplingBlock, self).__init__()
    self.conv = layers.Conv2D(filters, (4, 4), strides=(2, 2), padding="SAME") #downsampling_block
    self.res_block = ResBlock(filters)
  
  def call(self, input_x):
    x = self.conv(input_x)
    x = self.res_block(x)

    return x

class UpSamplingBlock(tf.keras.Model):
  def __init__(self, filters):
    super(UpSamplingBlock, self).__init__()
    self.up = layers.UpSampling2D(size=(2, 2))
    self.res_block = ResBlockShortCut(filters)
  
  def call(self, input_x, input_down):
    x = self.up(input_x)
    x = tf.keras.layers.Concatenate()([x, input_down])
    x = self.res_block(x)

    return x

