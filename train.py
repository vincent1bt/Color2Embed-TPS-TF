import tensorflow as tf
import time
import os
from train_utils import inner_step

from networks.models import Vgg19, CompleteModel

from data_generator.data_generator import train_dataset, image_paths 

import argparse
import builtins

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model')
parser.add_argument('--batch_size', type=int, default=16, help='Number batches')
parser.add_argument('--continue_training', default=False, action='store_true', help='Use training checkpoints to continue training the model')
parser.add_argument('--save_generator', default=False, action='store_true', help='Save the generator after training')

using_notebook = getattr(builtins, "__IPYTHON__", False)

opts = parser.parse_args([]) if using_notebook else parser.parse_args()

batch_size = opts.batch_size
epochs = opts.epochs
lr = 1e-4

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

vgg = Vgg19() # output shape 1, 16, 16, 512 for image 256x256
vgg.trainable = False

main_model = CompleteModel()

train_generator = train_dataset.batch(batch_size)

train_steps = int(len(image_paths) / batch_size) + 1

if using_notebook:
  print(train_steps)







checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(model=main_model, optimizer=optimizer)

if opts.continue_training:
  print("loading training checkpoints: ")                   
  print(tf.train.latest_checkpoint(checkpoint_dir))
  checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

loss_metric = tf.keras.metrics.Mean()
loss_results = []

@tf.function
def train_step(lab_images, images, reference_images):
  return inner_step(lab_images, 
               images, 
               reference_images, 
               main_model,
               vgg,
               optimizer)

def train(epochs):
  for epoch in range(epochs):
    batch_time = time.time()
    epoch_time = time.time()
    step = 0
    epoch_count = f"0{epoch + 1}/{epochs}" if epoch < 9 else f"{epoch + 1}/{epochs}"

    loss_metric.reset_states()

    for lab_img, img, reference_img in train_generator:
      loss, rec_loss, perc_loss = train_step(lab_img, img, reference_img)
      
      loss_metric.update_state(loss)
      loss = loss_metric.result().numpy()
      step += 1

      loss_results.append(loss)

      print('\r', 'Epoch', epoch_count, '| Step', f"{step}/{train_steps}",
            '| Loss:', f"{loss:.5f}", "| Step Time:", f"{time.time() - batch_time:.2f}", end='')    
        
      batch_time = time.time()

    checkpoint.save(file_prefix=checkpoint_prefix)

    print('\r', 'Epoch', epoch_count, '| Step', f"{step}/{train_steps}",
          '| Loss:', f"{loss:.5f}", "| Epoch Time:", f"{time.time() - epoch_time:.2f}")

train(epochs)

if opts.save_generator:
  main_model.save('saved_model/color_model')