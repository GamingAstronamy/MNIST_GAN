'''
Holds classes for generator and discriminator
'''

import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import pickle
    
class GAN(keras.Model):
    def __init__(self, noise_size : int, batch_size : int, num_layers : int, kernel_shape : tuple[int, int], image_shape : tuple[int, int, int]):
        super(GAN, self).__init__()
        
        self.noise_size = noise_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.kernel_shape = kernel_shape
        self.image_shape = image_shape

        self._cross_entropy = keras.losses.BinaryCrossentropy(from_logits = True)

        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()

        self.built = True

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

    def call(self, x, training : bool):
        x = self.generator(x, training=training)
        return self.discriminator(x, training=training)