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