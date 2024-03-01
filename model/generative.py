import os
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt

def is_power_of_two(n):
   return (n != 0) and (n & (n - 1) == 0)

class DCGANMNIST():
   def __init__(self, name='DCGAN_MNIST'):
      self.name = name
      self.__generator_mnist = self.__generator_mnist()
      self.__discriminator_mnist = self.__discriminator_mnist()
      self.generator_optimizer = self.generator_optimizer()
      self.discriminator_optimizer = self.discriminator_optimizer()
      self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

      self.initialize_checkpoint(name)

   def initialize_checkpoint(self, name):
      self.checkpoint_dir = f'./model_{name}/training_checkpoints_{name}'
      self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
      self.checkpoint = tf.train.Checkpoint(generator_optimizer = self.generator_optimizer,
                                       discriminator_optimizer = self.discriminator_optimizer,
                                       generator = self.__generator_mnist,
                                       discriminator = self.__discriminator_mnist)

   def to_Dataset(self, images, buffer_size, batch_size):
      self.batch_size = batch_size
      dataset = tf.data.Dataset.from_tensor_slices(images).shuffle(buffer_size).batch(batch_size)
      return dataset
   
   def __generator_mnist(self):
      '''
      This generator is defined for the MNIST dataset.The generator uses a 
      series of Conv2DTranspose layers to upsample an input of shape (100,) 
      to an image of shape (128, 128, 3).
      '''
      
      dense_output = int(7*7*256)
      model = tf.keras.Sequential()
      model.add(tf.keras.layers.Dense(dense_output, use_bias=False, input_shape=(100,)))
      model.add(tf.keras.layers.BatchNormalization())
      model.add(tf.keras.layers.LeakyReLU())

      model.add(tf.keras.layers.Reshape((7, 7, 256)))
      assert model.output_shape == (None, 7, 7, 256)

      model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
      assert model.output_shape == (None, 7, 7, 128)
      model.add(tf.keras.layers.BatchNormalization())
      model.add(tf.keras.layers.LeakyReLU())

      model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
      assert model.output_shape == (None, 14, 14, 64)
      model.add(tf.keras.layers.BatchNormalization())
      model.add(tf.keras.layers.LeakyReLU())

      model.add(tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
      assert model.output_shape == (None, 28, 28, 1)

      return model
   
   def __discriminator_mnist(self):
      model = tf.keras.Sequential()
      model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                       input_shape=[28, 28, 1]))
      model.add(tf.keras.layers.LeakyReLU())
      model.add(tf.keras.layers.Dropout(0.3))

      model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
      model.add(tf.keras.layers.LeakyReLU())
      model.add(tf.keras.layers.Dropout(0.3))

      model.add(tf.keras.layers.Flatten())
      model.add(tf.keras.layers.Dense(1))

      return model
   
   def discriminator_loss(self, real_output, fake_output):
      real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
      fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
      total_loss = real_loss + fake_loss
      return total_loss
   
   def generator_loss(self, fake_output):
      return self.cross_entropy(tf.ones_like(fake_output), fake_output)
   
   def generator_optimizer(self):
      return tf.keras.optimizers.Adam(1e-4)
   
   def discriminator_optimizer(self):
      return tf.keras.optimizers.Adam(1e-4)
   
   @tf.function
   def __train_step_mnist(self, images, noise_dim):
      noise = tf.random.normal([self.batch_size, noise_dim])

      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
         generated_images = self.__generator_mnist(noise, training=True)

         real_output = self.__discriminator_mnist(images, training=True)
         fake_output = self.__discriminator_mnist(generated_images, training=True)

         gen_loss = self.generator_loss(fake_output)
         disc_loss = self.discriminator_loss(real_output, fake_output)

      gradients_of_generator = gen_tape.gradient(gen_loss, self.__generator_mnist.trainable_variables)
      gradients_of_discriminator = disc_tape.gradient(disc_loss, self.__discriminator_mnist.trainable_variables)

      self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.__generator_mnist.trainable_variables))
      self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.__discriminator_mnist.trainable_variables))
   
   def fit_mnist(self, dataset, epochs, noise_dim = 100, save_every = 15, generate_images=False, n_examples=None, batch_size = None):
      self.n_examples = n_examples
      if generate_images:
         if n_examples == None:
            raise ValueError("n_examples must be specified if generate_images is True")
         seed = tf.random.normal([n_examples, noise_dim])

      if batch_size != None:
         self.batch_size = batch_size
      
      for epoch in range(epochs):
         for image_batch in tqdm(dataset):
            self.__train_step_mnist(image_batch, noise_dim)

         if (epoch + 1) % save_every == 0:
            self.checkpoint.save(file_prefix = self.checkpoint_prefix)
            print(f"Saved Checkpoint at epoch {epoch + 1}")
         
         if generate_images:
            self.generate_and_save_images(epoch + 1, seed, n_examples)
   
   def generate_and_save_images(self, epoch, test_input, n_examples):
      dim = tf.sqrt(n_examples).astype(int())
      plt_dims_x, plt_dims_y = (dim+1, dim+1)
      predictions = self.__generator_mnist(test_input, training=False)

      fig = plt.figure(figsize=(plt_dims_x, plt_dims_y))

      for i in range(predictions.shape[0]):
         plt.subplot(plt_dims_x, plt_dims_y, i+1)
         plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
         plt.axis('off')

      os.makedirs(f'./model_{self.name}', exist_ok=True)
      os.makedirs(f'./model_{self.name}/images', exist_ok=True)
      
      plt.savefig(f'./model_{self.name}/images/image_at_epoch_{epoch}.png')
      # plt.show()
      plt.close(fig)


# Generalized DCGAN for any image dataset
class DCGAN(DCGANMNIST):
   def __init__(self, input_shape, name='DCGAN'):
      super().__init__(name)
      
      self.name = name
      self.input_height, self.input_width, self.input_depth = input_shape
      
      if self.input_height != self.input_width:
         raise ValueError("Input Height and Width must be the same")
      if is_power_of_two(self.input_height) == False:
         raise ValueError("Input Height and Width must be a power of 2")
      if is_power_of_two(self.input_width) == False:
         raise ValueError("Input Height and Width must be a power of 2")

      self.generator = self.generator()
      self.discriminator = self.discriminator()
   
   def load_and_decode_image(self, file_path):
      img = tf.io.read_file(file_path)
      img = tf.image.decode_jpeg(img, channels=self.input_depth)
      img = tf.image.resize(img, (self.input_height, self.input_width))
      img = (tf.cast(img, tf.float32) - 127.5) / 127.5
      return img
   
   def preprocess_image(self, image, normalize=True):
      image = tf.image.resize(image, (self.input_height, self.input_width))
      if normalize:
         image = (image - 127.5) / 127.5
      return image

   def process_from_path(self, path, batch_size=32, shuffle=True, buffer_size=None):
      self.batch_size = batch_size
      image_dir = path
      image_files = tf.data.Dataset.list_files(os.path.join(image_dir, '*.jpg'))
      
      if buffer_size == None:
         buffer_size = len(image_files)
      
      if shuffle:
         image_files = image_files.shuffle(buffer_size)
      
      return image_files.map(self.load_and_decode_image).batch(batch_size)
   
   def generator(self):
      '''
      The generator uses a series of Conv2DTranspose layers to upsample an input 
      of shape (100,) to an image of shape (128, 128, 3).
      
      WIP: The generator is not yet dynamic, and only works for images of shape (128, 128, 3).
      '''
      model = tf.keras.Sequential()
      model.add(tf.keras.layers.Dense(4*4*1024, use_bias=False, input_shape=(100,)))
      model.add(tf.keras.layers.BatchNormalization())
      model.add(tf.keras.layers.LeakyReLU())

      model.add(tf.keras.layers.Reshape((4, 4, 1024)))
      assert model.output_shape == (None, 4, 4, 1024)

      model.add(tf.keras.layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same', use_bias=False))
      assert model.output_shape == (None, 8, 8, 512)
      model.add(tf.keras.layers.BatchNormalization())
      model.add(tf.keras.layers.LeakyReLU())

      model.add(tf.keras.layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
      assert model.output_shape == (None, 16, 16, 256)
      model.add(tf.keras.layers.BatchNormalization())
      model.add(tf.keras.layers.LeakyReLU())

      model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
      assert model.output_shape == (None, 32, 32, 128)
      model.add(tf.keras.layers.BatchNormalization())
      model.add(tf.keras.layers.LeakyReLU())

      # model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
      # assert model.output_shape == (None, 64, 64, 64)
      # model.add(tf.keras.layers.BatchNormalization())
      # model.add(tf.keras.layers.LeakyReLU())

      model.add(tf.keras.layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
      assert model.output_shape == (None, self.input_height, self.input_width, self.input_depth)

      return model
   
   def discriminator(self):
      model = tf.keras.Sequential()
      model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                       input_shape=[self.input_height, self.input_width, self.input_depth]))
      model.add(tf.keras.layers.LeakyReLU())
      model.add(tf.keras.layers.Dropout(0.3))

      model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
      model.add(tf.keras.layers.LeakyReLU())
      model.add(tf.keras.layers.Dropout(0.3))

      model.add(tf.keras.layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
      model.add(tf.keras.layers.LeakyReLU())
      model.add(tf.keras.layers.Dropout(0.3))

      model.add(tf.keras.layers.Flatten())
      model.add(tf.keras.layers.Dense(1))

      return model
   
   @tf.function
   def train_step(self, images, noise_dim):
      noise = tf.random.normal([self.batch_size, noise_dim])

      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
         generated_images = self.generator(noise, training=True)

         real_output = self.discriminator(images, training=True)
         fake_output = self.discriminator(generated_images, training=True)

         gen_loss = self.generator_loss(fake_output)
         disc_loss = self.discriminator_loss(real_output, fake_output)

      gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
      gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

      self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
      self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
   
   def fit(self, dataset, epochs, noise_dim = 100, save_every = 15, generate_images=False, n_examples=None, batch_size = None):
      self.n_examples = n_examples
      if generate_images:
         if n_examples == None:
            raise ValueError("n_examples must be specified if generate_images is True")
         seed = tf.random.normal([n_examples, noise_dim])

      if batch_size != None:
         self.batch_size = batch_size
      
      for epoch in range(epochs):
         for image_batch in tqdm(dataset):
            self.train_step(image_batch, noise_dim)

         if (epoch + 1) % save_every == 0:
            self.checkpoint.save(file_prefix = self.checkpoint_prefix)
            print(f"Saved Checkpoint at epoch {epoch + 1}")
         
         if generate_images:
            self.generate_and_save_images(epoch + 1, seed)

   def restore_checkpoint(self, name):
      self.initialize_checkpoint(name)
      self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
      print(f"Restored Checkpoint from {self.checkpoint_dir}")
   
   def generate_image(self, seed):
      noise = tf.random.normal([1, 100])
      generated_image = self.generator(noise, training=False)
      return generated_image