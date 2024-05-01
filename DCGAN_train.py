import tensorflow as tf
import tensorflow_datasets as tfds
from model.generative import DCGAN

dataset, info = tfds.load('lsun/dining_room', split='train', shuffle_files=True, with_info=True)

BATCH_SIZE = 256
EPOCHS = 50
noise_dim = 100

dcgan = DCGAN((64, 64, 3), 'Dining_Room')

dataset = dataset.map(lambda x: dcgan.preprocess_image(x['image'], normalize=True))
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

dcgan.fit(dataset, EPOCHS, noise_dim, save_every=1, batch_size=BATCH_SIZE, generate_images=True, n_examples=16)