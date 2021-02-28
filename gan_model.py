
import tensorflow as tf
from matplotlib import pyplot as plt

import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Lambda, LeakyReLU
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
import tensorflow.keras.backend as K
from tensorflow.keras import losses
from numpy import zeros, ones, vstack, expand_dims
from numpy.random import randn, randint
from tensorflow.keras.optimizers import Adam
import numpy as np


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(64,), mean=0., stddev=1.0)
    return z_mean + K.exp(z_log_var / 2) * epsilon


def define_generator(latent_dim):
    img = Input(shape=latent_dim)
    model = layers.Dense(latent_dim * 4, activation='relu',
                               input_shape=(latent_dim,))(img)
    model = layers.Dense(latent_dim * 8, activation='relu')(model)
    model = layers.Dense(latent_dim * 8, activation='tanh')(model)
    model = layers.Dense(latent_dim* 4, activation='sigmoid')(model)
    model = layers.Dense(latent_dim)(model)

    generator = Model(img, model)

    return generator


def define_discriminator(latent_dim):
    descriminator = Sequential()
    descriminator.add(layers.Dense(latent_dim * 2, activation='relu',
                                   input_shape=(latent_dim,)))
    descriminator.add(layers.Dense(latent_dim * 8, activation='relu'))
    descriminator.add(layers.Dense(latent_dim * 4, activation='relu'))
    descriminator.add(layers.Dense(1, activation='sigmoid'))
    encoded_repr = Input(shape=(latent_dim,))
    validity = descriminator(encoded_repr)
    descriminator = Model(encoded_repr, validity)
    descriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy',
                      metrics=['accuracy'])
    return descriminator


def define_gan(generator, discriminator):
    discriminator.trainable = False
    gan = Sequential()
    gan.add(generator)
    gan.add(discriminator)

    gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['mse'])

    return gan

# For training --------------------------------------------------------------
checkpoint_path = "model_save_gan/cp.ckptGan"

def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100,
          n_batch=256):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)

    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):

            if i % 2 == 0:
                # get randomly selected 'real' samples
                X_real, y_real = generate_real_samples(dataset, half_batch)
                # X_real = expand_dims(X_real, axis=-1)
                # generate 'fake' examples
                X_fake, y_fake = generate_fake_samples(g_model, latent_dim,
                                                       half_batch)
                # create training set for the discriminator
                X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
                # update discriminator model weights
                d_loss, _ = d_model.train_on_batch(X, y)


            # prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)

            if i % 2 == 0:
                # summarize loss on this batch
                print('>%d, %d/%d, d=%.3f, g=%.3f' % (
                i + 1, j + 1, bat_per_epo, d_loss, g_loss[0]))



def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, 1))
    return X, y


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = g_model.predict(x_input)
    # create 'fake' class labels (0)
    y = zeros((n_samples, 1))
    return X, y


def interpolation(generator, encoder, decoder, x_dataset, y_dataset, latent_dim, n):
    for i in range(10):
        # randomly select two images
        i1 = np.random.randint(0, x_dataset.shape[0], 1)
        i2 = np.random.randint(0, x_dataset.shape[0], 1)

        # make prediction by ae
        l1 = encoder.predict(x_dataset[i1, :, :])
        l2 = encoder.predict(x_dataset[i2, :, :])
        lin_ae = np.linspace(l1, l2, n).reshape(n, latent_dim)
        ae_res = decoder.predict(lin_ae)

        # make prediction by GAN
        gan = np.random.normal((2, latent_dim))
        gan_res = decoder.predict(generator.predict(np.linspace(gan[0], gan[1], n)))

        for i in range(1, n + 1):
            # Display original
            ax = plt.subplot(2, n, i)
            plt.imshow(ae_res[i].reshape(32, 32))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Display reconstruction
            ax = plt.subplot(2, n, i + n)
            plt.imshow(gan_res[i].reshape(32, 32))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

