import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# from helpers import GAN

def main():
#     gan = GAN(100, 256, 2, (5,5), (28, 28, 1))
    pass
#     noise = tf.random.normal([1, 100])
#     generated_image = gan.generator(noise, training=False)

#     plt.imshow(generated_image[0, :, :, 0], cmap='gray')

if __name__ == '__main__':
    with tf.device('/CPU:0'):
        main()