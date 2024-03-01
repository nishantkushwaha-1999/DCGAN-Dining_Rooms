import tensorflow as tf
import tensorflow_datasets as tfds
from model.generative import DCGANMNIST, DCGAN

# BUFFER_SIZE = 60000
# BATCH_SIZE = 256
EPOCHS = 50
noise_dim = 100

dcgan = DCGAN((64, 64, 3), 'Faces')
dataset = dcgan.process_from_path("./images/", batch_size=1, shuffle=True, buffer_size=None)
dcgan.fit(dataset, EPOCHS, noise_dim, save_every=1, batch_size=1)

# dcgan = DCGANMNIST()
# dataset = dcgan.to_Dataset(train_images, BUFFER_SIZE, BATCH_SIZE)
# dcgan.fit_mnist(dataset, EPOCHS, noise_dim, save_every=5, batch_size=BATCH_SIZE, generate_images=True, n_examples=16)