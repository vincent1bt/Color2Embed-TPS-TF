import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras import layers
from data_generator.data_utils import rgb_to_lab, create_tps, tps_augmentation

data_path = os.path.abspath("./data/clean_image_paths.parquet")
df_image_paths = pd.read_parquet(data_path)

image_height = 256
image_width = 256

def load_image(image_path):
  img = tf.io.read_file(image_path)
  img = tf.image.decode_jpeg(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  size = tf.shape(img)
  width, height = size[0], size[1]
  max_size = tf.minimum(width, height)
  img = tf.image.random_crop(img, size=(max_size, max_size, 3))

  img = tf.image.resize(img, (image_height, image_width))

  return img

rotation_layers = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomRotation(0.05, fill_mode="reflect"),
])

def load_train_image(image_path):
  img = load_image(image_path)

  tps = create_tps(image_height, image_width)
  reference = tps_augmentation(img, tps, height=image_height, width=image_width)

  if tf.random.uniform([]) > 0.5:
    img = tf.image.flip_left_right(img)
  
  if tf.random.uniform([]) > 0.5:
    img = rotation_layers(tf.expand_dims(img, axis=0))[0]

  if tf.random.uniform([]) > 0.5:
    reference = tf.image.flip_left_right(reference)

  if tf.random.uniform([]) > 0.5:
    reference = rotation_layers(tf.expand_dims(reference, axis=0))[0]

  noise = tf.random.normal(shape=tf.shape(reference), mean=0.0, stddev=0.02, dtype=tf.float32)
  reference = tf.add(reference, noise)
  reference = tf.clip_by_value(reference, 0.0, 1.0)

  # lab image
  lab_img = rgb_to_lab(img)

  return lab_img, img, reference

image_paths = df_image_paths["path"].tolist()

train_dataset = tf.data.Dataset.from_tensor_slices((image_paths))
train_dataset = train_dataset.shuffle(len(image_paths))
train_dataset = train_dataset.map(load_train_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((image_paths))
test_dataset = test_dataset.map(load_train_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

