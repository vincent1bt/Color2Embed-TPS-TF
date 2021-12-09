import tensorflow as tf

l1_smooth = tf.keras.losses.Huber()

def compute_reconstruction_loss(y_true, y_fake, HUBER_DELTA=1):
  return l1_smooth(y_true, y_fake)

def perceptual_loss(feature_map_true, feature_map_fake):
  return tf.reduce_mean(tf.math.abs(feature_map_true - feature_map_fake))

def compute_loss(ab_images, ab_pred_images, feature_map_true, feature_map_fake, lambda_perc=0.1):
  rec_loss = compute_reconstruction_loss(ab_images, ab_pred_images)

  perc_loss = perceptual_loss(feature_map_true, feature_map_fake) * lambda_perc

  loss = rec_loss + perc_loss

  return loss, rec_loss, perc_loss

