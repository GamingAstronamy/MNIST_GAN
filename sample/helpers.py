'''
Holds classes for generator and discriminator
'''

import glob
import os
import pickle

import imageio
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython import display
from keras import layers
from tensorflow import keras
from tqdm import tqdm


class GAN:
    def __init__(self, noise_size : int, batch_size : int, num_layers : int, kernel_shape : tuple[int, int], image_shape : tuple[int, int, int]):        
        self.noise_size = noise_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.kernel_shape = kernel_shape
        self.image_shape = image_shape

        self._cross_entropy = keras.losses.BinaryCrossentropy(from_logits = True)

        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        
        self._generator_optimizer = keras.optimizers.Adam(3e-4)
        self._discriminator_optimizer = keras.optimizers.Adam(3e-4)
        
    @classmethod
    def load(cls, save_folder : str):
        with open(os.path.join(save_folder, 'parameters.pkl'), 'wb') as f:
            parameters = pickle.load(f)
            
        gan = GAN(*parameters)
        
        gan.generator.load_weights(os.path.join(save_folder, 'generator.h5'))
        gan.discriminator.load_weights(os.path.join(save_folder, 'discriminator.h5'))
        
        return gan
    
    def save(self, save_folder : str):
        
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        parameters = [self.noise_size,
                      self.batch_size,
                      self.num_layers,
                      self.kernel_shape,
                      self.image_shape]
        
        with open(os.path.join(save_folder, 'parameters.pkl'), 'wb') as f:
            pickle.dump(parameters, f)

        self.generator.save_weights(os.path.join(save_folder, 'generator.h5'))
        self.discriminator.save_weights(os.path.join(save_folder, 'discriminator.h5'))

    def _build_generator(self):
        model = keras.Sequential(name = 'generator')

        model.add(layers.Dense((self.image_shape[0]/(2**self.num_layers)) * (self.image_shape[1]/(2**self.num_layers)) * self.batch_size, use_bias=False, input_shape = (self.noise_size,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((int(self.image_shape[0]/(2**self.num_layers)), int(self.image_shape[1]/(2**self.num_layers)), self.batch_size)))

        model.add(layers.Conv2DTranspose(self.batch_size/2, self.kernel_shape, strides=(1, 1), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        for i in range(self.num_layers-1):
            model.add(layers.Conv2DTranspose(self.batch_size/(4 * 2**i), self.kernel_shape, strides=(2,2), padding='same', use_bias=False))
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(self.image_shape[2], self.kernel_shape, strides=(2,2), padding='same', use_bias=False, activation='tanh'))

        return model

    def _generator_loss(self, fake_output):
        return self._cross_entropy(tf.ones_like(fake_output), fake_output)

    def _build_discriminator(self):
        model = keras.Sequential(name = 'discriminator')

        model.add(layers.Conv2D(self.batch_size / (2**(self.num_layers)), self.kernel_shape, strides=(2, 2), padding='same', input_shape=self.image_shape))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        for i in range(self.num_layers-1):
            model.add(layers.Conv2D(self.batch_size / (2 ** (self.num_layers - i - 1)), self.kernel_shape, strides=(2, 2), padding='same'))
            model.add(layers.LeakyReLU())
            model.add(layers.Dropout(.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model

    def _discriminator_loss(self, real_output, fake_output):
        real_loss = self._cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self._cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
    
    def _train_step(self, images):
        noise = tf.random.normal([self.batch_size, self.noise_size])
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
                        
            gen_loss = self._generator_loss(fake_output)
            disc_loss = self._discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)  
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        
        self._generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self._discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        
        return (gen_loss, disc_loss)
        
    def train(self, dataset, epochs, make_gif = False, img_folder = None):
        images = tf.data.Dataset.from_tensor_slices(dataset).shuffle(60000).batch(self.batch_size)
        
        seed = tf.random.normal([16, self.noise_size])
        
        if make_gif:
            if img_folder is None:
                raise Exception("img_folder need to be specified when make_gif = True")
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)
        
        for epoch in range(epochs):
            
            pbar = tqdm(images, desc=f'Epoch {epoch+1}/{epochs}')
            for image_batch in pbar:
                (gen_loss, disc_loss) = self._train_step(image_batch)
                
                pbar.set_postfix_str(f'generator_loss: {round(float(gen_loss),3)} | discriminator_loss: {round(float(disc_loss),3)}')

            if make_gif:
                display.clear_output(wait=True)
                self._generate_and_save_images(epoch+1, seed, img_folder)

        if make_gif:
            display.clear_output(wait=True)
            self._generate_and_save_images(epochs, seed, img_folder)
        
            self._build_gif(img_folder)

    def _generate_and_save_images(self, epoch, test_input, img_folder):
        predictions = self.generator(test_input, training=False)
        
        fig = plt.figure(figsize=(4, 4))
        
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')
            
        plt.savefig(os.path.join(img_folder, f'image_at_epoch_{round(epoch)}.png'))
        plt.close('all')
        
    def _build_gif(self, img_folder):
        anim_file = 'dcgan.gif'

        with imageio.get_writer(anim_file, mode='I') as writer:
            filenames = glob.glob(os.path.join(img_folder, 'image*.png'))
            filenames = sorted(filenames)
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
            image = imageio.imread(filename)
            writer.append_data(image)