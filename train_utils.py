import tensorflow as tf
from networks.loss_functions import compute_loss
from data_generator.data_utils import lab_to_rgb

def inner_step(lab_images, 
               images, 
               reference_images, 
               main_model,
               vgg,
               optimizer):
  l_images = lab_images[:, :, :, :1] # Bx256x256x1
  ab_images = lab_images[:, :, :, 1:] # Bx256x256x2

  # a_n = (ab_images[:, :, :, :1] + 86.185) / 184.439
  # b_n = (ab_images[:, :, :, 1:] + 107.863) / 202.345

  l_normalized = l_images / 100.0

  # ab_images = tf.concat([a_n, b_n], axis=-1)

  with tf.GradientTape() as tape:
    # ab_pred_images = main_model([l_images, reference_images])
    # style_z = color_encoder(reference_images)
    # ab_pred_images = model(l_normalized, style_z)

    ab_pred_images = main_model(l_normalized, reference_images)

    a = tf.clip_by_value(ab_pred_images[:, :, :, :1], -86.185, 98.254)
    b = tf.clip_by_value(ab_pred_images[:, :, :, 1:], -107.863, 94.482)

    ab_range = tf.concat([a, b], axis=-1) # original range

    # a_n = (a + 86.185) / 184.439
    # b_n = (b + 107.863) / 202.345

    # ab_pred_normalized = tf.concat([a_n, b_n], axis=-1)

    feature_map_true = vgg(images) # Bx16x16x512
    
    fake_lab_images = tf.concat([l_images, ab_range], axis=-1) # Bx256x256x3
    fake_rgb_images = lab_to_rgb(fake_lab_images) # returns stats between 0-1

    # print(fake_rgb_images.numpy().max())

    feature_map_fake = vgg(fake_rgb_images) # Bx16x16x512

    loss, rec_loss, perc_loss = compute_loss(ab_images, ab_range, feature_map_true, feature_map_fake)
  
  # color_gradients = color_tape.gradient(loss, color_encoder.trainable_variables)
  gradients = tape.gradient(loss, main_model.trainable_variables)

  # color_optimizer.apply_gradients(zip(color_gradients, color_encoder.trainable_variables))
  optimizer.apply_gradients(zip(gradients, main_model.trainable_variables))

  return loss, rec_loss, perc_loss

