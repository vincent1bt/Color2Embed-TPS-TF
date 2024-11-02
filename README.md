# Color2Embedding and Thin Plate Spline

Color2Embedding is a framework to colorize black and white images. It's training using a pair of reference and target images, from reference images we obtain the color information to inject such information into the target image. 

We use the TPS augmentation to transform the reference image. You can read more about TPS and the Color2Embedding framework in [this blog post](https://vincentblog.link/posts/thin-plate-splines-and-its-implementation-as-data-augmentation-technique).

The TPS augmentation implementation is inside *data_generator/data_utils.py*.

This network is implemented using *TensorFlow*, the original code uses *PyTorch* and you can find it [here](https://github.com/zhaohengyuan1/Color2Embed).

The project is based on the [Color2Embed: Fast Exemplar-Based Image Colorization using Color Embeddings](https://arxiv.org/abs/2106.08017) paper.


## Training

First you need to install **tensorflow_addons**:

```
pip install tfa-nightly
```

To train the model you need to download:

```
https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz
```

and extract the images inside the *data* folder.

You can train the model using:

```
python train.py --epochs=300
```

There is a bug where the loss could lead to *nan*. If this happens you can stop the training, delete the recent checkpoints and continue from an older checkpoint.

## Weight modulation/demodulation

Weight modulation and demodulation is implemented using the code from [https://github.com/manicman1999/StyleGAN2-Tensorflow-2.0/blob/master/conv_mod.py](https://github.com/manicman1999/StyleGAN2-Tensorflow-2.0/blob/master/conv_mod.py).

## LAB images

The code to translate images from and to LAB format is from [https://github.com/xahidbuffon/TF_RGB_LAB/blob/master/rgb_lab_formulation.py](https://github.com/xahidbuffon/TF_RGB_LAB/blob/master/rgb_lab_formulation.py).

