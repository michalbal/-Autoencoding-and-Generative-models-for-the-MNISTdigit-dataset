from matplotlib import pyplot as plt
import tensorflow as tf

import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras import losses
from keras.models import load_model
import gan_model

import numpy as np
import os

latent_dim = 64
noise_sigma = 0.35
train_AE = False
sml_train_size = 50

# load train and test images, and pad & reshape them to (-1,32,32,1)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1)).astype('float32') / 255.0
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1)).astype('float32') / 255.0
x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)))
x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)))
print(x_train.shape)
print(x_test.shape)
# exit()
y_train = keras.utils.to_categorical(y_train, num_classes=10, dtype='float32')
y_test = keras.utils.to_categorical(y_test, num_classes=10, dtype='float32')

encoder = Sequential()
encoder.add(layers.Conv2D(16, (4, 4), strides=(2, 2), activation='relu', padding='same', input_shape=(32, 32, 1)))
encoder.add(layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same'))
encoder.add(layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same'))
encoder.add(layers.Conv2D(96, (3, 3), strides=(2, 2), activation='relu', padding='same'))
encoder.add(layers.Reshape((2 * 2 * 96,)))
encoder.add(layers.Dense(latent_dim))

# at this point the representation is (4, 4, 8) i.e. 128-dimensional
decoder = Sequential()
decoder.add(layers.Dense(2 * 2 * 96, activation='relu', input_shape=(latent_dim,)))
decoder.add(layers.Reshape((2, 2, 96)))
decoder.add(layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same'))
decoder.add(layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same'))
decoder.add(layers.Conv2DTranspose(16, (4, 4), strides=(2, 2), activation='relu', padding='same'))
decoder.add(layers.Conv2DTranspose(1, (4, 4), strides=(2, 2), activation='sigmoid', padding='same'))

autoencoder = keras.Model(encoder.inputs, decoder(encoder.outputs))
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

checkpoint_path = "model_save/cp.ckpt"

if train_AE:
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True)
    autoencoder.fit(x_train + noise_sigma * np.random.randn(*x_train.shape), x_train,
                    epochs=15,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    callbacks=[cp_callback])
else:
    autoencoder.load_weights(checkpoint_path)

# decoded_imgs = autoencoder.predict(x_test)
latent_codes = encoder.predict(x_train)
# decoded_imgs = decoder.predict(latent_codes)


# question 2

# # Classifer Network - currently minimal
# classifier = Sequential()
# classifier.add(layers.Dense(64, activation='sigmoid', input_shape=(latent_dim,)))
# classifier.add(layers.Dense(10,activation='softmax', input_shape=(latent_dim,)))
#
# train_codes = encoder.predict(x_train[:sml_train_size])
# test_codes = encoder.predict(x_test)
#
# classifier.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#
# classifier.fit(train_codes, y_train[:sml_train_size],
# 	                epochs=200,
# 	                batch_size=16,
# 	                shuffle=True,
# 	                validation_data=(test_codes, y_test))
#
#
# full_cls_enc = keras.models.clone_model(encoder)
# full_cls_cls = keras.models.clone_model(classifier)
# full_cls = keras.Model(full_cls_enc.inputs,full_cls_cls(full_cls_enc.outputs))
#
# full_cls.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#
# full_cls.fit(x_train[:sml_train_size], y_train[:sml_train_size],
# 	                epochs=100,
# 	                batch_size=16,
# 	                shuffle=True,
# 	                validation_data=(x_test, y_test))



# GAN

generator = gan_model.define_generator(latent_dim)
descriminator = gan_model.define_discriminator(latent_dim)
descriminator.trainable = False
gan = gan_model.define_gan(generator, descriminator)

# Train GAN
gan_model.train(generator, descriminator, gan, latent_codes, latent_dim, n_epochs=1000)

# Use gan

# generate images
# latent_points = encoder.predict(x_test)
latent_points = gan_model.generate_latent_points(latent_dim, len(x_test))
# generate images
latent_codes = generator.predict(latent_points)
decoded_imgs = decoder.predict(latent_codes)

n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(32, 32))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(32, 32))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

gan_model.interpolation(generator, encoder, decoder, x_test, y_test, latent_dim, n)
