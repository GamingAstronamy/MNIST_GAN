    # train.py -- Creates a new GAN and trains it on the MNIST dataset
    # Copyright (C) 2022  Ethan Barnes

    # This program is free software: you can redistribute it and/or modify
    # it under the terms of the GNU Affero General Public License as
    # published by the Free Software Foundation, either version 3 of the
    # License, or (at your option) any later version.

    # This program is distributed in the hope that it will be useful,
    # but WITHOUT ANY WARRANTY; without even the implied warranty of
    # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    # GNU Affero General Public License for more details.

    # You should have received a copy of the GNU Affero General Public License
    # along with this program.  If not, see <https://www.gnu.org/licenses/>.

import cv2
import numpy as np
import tensorflow as tf
from helpers import GAN


def main():
    tf.random.set_seed(42)
    
    gan = GAN(100, 256, 2, (5,5), (28, 28, 1))

    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5
    
    gan.train(train_images, 1)
    gan.save('model/')

if __name__ == '__main__':
    main()