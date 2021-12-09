import tensorflow as tf
from tensorflow.keras import layers
from networks.model_blocks import DownSamplingBlock, UpSamplingBlock, ColorLayer, ResBlock

class Generator(tf.keras.Model):
  def __init__(self):
    super(Generator, self).__init__()
    self.conv = layers.Conv2D(64, (3, 3), strides=(1, 1), padding="SAME")

    self.conv_1 = layers.Conv2D(256, (1, 1), strides=(1, 1), kernel_initializer="he_uniform")
    self.conv_2 = layers.Conv2D(128, (1, 1), strides=(1, 1), kernel_initializer="he_uniform")
    self.conv_3 = layers.Conv2D(64, (1, 1), strides=(1, 1), kernel_initializer="he_uniform")
    self.conv_4 = layers.Conv2D(2, (3, 3), strides=(1, 1), padding="SAME")

    self.color_1 = ColorLayer(512)
    self.color_2 = ColorLayer(256)
    self.color_3 = ColorLayer(128)
    self.color_4 = ColorLayer(64)

    self.res_block = ResBlock(64)

    self.down_1 = DownSamplingBlock(128)
    self.down_2 = DownSamplingBlock(256)
    self.down_3 = DownSamplingBlock(512)
    self.down_4 = DownSamplingBlock(512)

    self.up_1 = UpSamplingBlock(512)
    self.up_2 = UpSamplingBlock(256)
    self.up_3 = UpSamplingBlock(128)
    self.up_4 = UpSamplingBlock(64)
  
  def call(self, input_x, style_input):
    x = self.conv(input_x)

    x1 = self.res_block(x) # 256
    x2 = self.down_1(x1) # 128
    x3 = self.down_2(x2) # 64
    x4 = self.down_3(x3) # 32
    x5 = self.down_4(x4) # 16

    x = self.up_1(x5, x4)
    x = self.color_1([x, style_input])

    style = self.conv_1(style_input)

    x = self.up_2(x, x3)
    x = self.color_2([x, style])

    style = self.conv_2(style_input)

    x = self.up_3(x, x2)
    x = self.color_3([x, style])

    style = self.conv_3(style_input)

    x = self.up_4(x, x1)
    x = self.color_4([x, style])

    x = self.conv_4(x)

    return x

class Vgg19(tf.keras.Model):
  def __init__(self):
    super(Vgg19, self).__init__()
    original_vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    original_vgg.trainable = False
    layer = 'block5_conv2' 
    
    output = original_vgg.get_layer(layer).output
    self.model = tf.keras.Model([original_vgg.input], output)
    self.model.trainable = False
  
  def call(self, x):
    x = tf.keras.applications.vgg19.preprocess_input(x * 255.0)
    return self.model(x)

class ColorEncoder(tf.keras.Model):
  def __init__(self):
    super(ColorEncoder, self).__init__()
    original_vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    original_vgg.trainable = True
    layer = 'block5_conv2' 
    
    output = original_vgg.get_layer(layer).output

    self.model = tf.keras.Model([original_vgg.input], output)

    self.conv = layers.Conv2D(512, (3, 3), strides=(2, 2))

    self.conv_1 = layers.Conv2D(512, (1, 1), strides=(1, 1))
    self.conv_2 = layers.Conv2D(512, (1, 1), strides=(1, 1))
    self.conv_3 = layers.Conv2D(512, (1, 1), strides=(1, 1))
    self.conv_4 = layers.Conv2D(512, (1, 1), strides=(1, 1))

    self.global_avg = layers.Lambda(lambda t4d: tf.reduce_mean(t4d, axis=(1,2), keepdims=True), name='GlobalAverage2D')
  
  def call(self, x):
    x = tf.keras.applications.vgg19.preprocess_input(x * 255.0)
    x = self.model(x)
    x = self.conv(x)
    x = self.global_avg(x)

    x = self.conv_1(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = self.conv_2(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = self.conv_3(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = self.conv_4(x)
    x = tf.keras.layers.LeakyReLU()(x)

    return x

class CompleteModel(tf.keras.Model):
  def __init__(self):
    super(CompleteModel, self).__init__()
    self.color_encoder = ColorEncoder()
    self.generator = Generator()
  
  def call(self, input_x, input_z):
    z = self.color_encoder(input_z)
    x = self.generator(input_x, z)

    return x

