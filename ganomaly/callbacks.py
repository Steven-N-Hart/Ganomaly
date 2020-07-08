import os

import tensorflow as tf


def tbc(log_dir='logs', hist_freq=0, write_graph=True, write_images=False, update_freq='batch'):
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=hist_freq, write_graph=write_graph,
                                          write_images=write_images, update_freq=update_freq)


def cpc(log_dir='logs', prefix='ckpt', save_weights_only=True):
    return tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(log_dir, prefix),
                                              save_weights_only=save_weights_only)


class GANMonitor(tf.keras.callbacks.Callback):
    def __init__(self, num_img=3, latent_dim=512, log_dir='logs'):
        super().__init__()
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.log_dir = log_dir

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.G(random_latent_vectors)
        latent_space = self.model.E(generated_images)
        regenerated_images = self.model.G(latent_space)

        generated_images = tf.math.multiply(tf.math.add(generated_images, 127.5), 127.5)
        generated_images.numpy()
        regenerated_images = tf.math.multiply(tf.math.add(regenerated_images, 127.5), 127.5)
        regenerated_images.numpy()

        for i in range(self.num_img):
            img = tf.keras.preprocessing.image.array_to_img(generated_images[i])
            img.save(os.path.join(self.log_dir, "generated_img_" + str(i) + "_" + str(epoch) + "_" + str(
                generated_images[i].shape[0]) + ".png"))
            img = tf.keras.preprocessing.image.array_to_img(regenerated_images[i])
            img.save(os.path.join(self.log_dir, "regenerated_img_" + str(i) + "_" + str(epoch) + "_" + str(
                generated_images[i].shape[0]) + ".png"))
        # TODO: Add image to tensorboard (https://www.tensorflow.org/tensorboard/image_summaries#visualizing_multiple_images)

class AlphaUpdate(tf.keras.callbacks.Callback):
    def __init__(self, batch_size, num_imgs_to_fade=500000, **kwargs):
        super(AlphaUpdate, self).__init__()
        self.batch_size = batch_size
        self.num_imgs_to_fade = num_imgs_to_fade

    def on_batch_begin(self, batch, **kwargs):
        num_imgs_seen = batch * self.batch_size
        a = tf.Variable(0., dtype=tf.float32)
        f = tf.Variable(True, dtype=tf.float32)
        num_imgs_seen = tf.Variable(num_imgs_seen)
        # Only update if I am on an image bigger than 4x4 (which has self.model.G_fade=None)
        if num_imgs_seen > self.num_imgs_to_fade:
            a.assign(1.0)
            f.assign(False)
        else:
            a.assign(tf.divide(tf.cast(num_imgs_seen, tf.float32), tf.cast(self.num_imgs_to_fade, tf.float32)))
            if self.model.first_round:
                f.assign(False)
        self.model.alpha.assign(a)
        self.model.num_imgs_seen.assign(num_imgs_seen)
        self.model.fade.assign(f)

    def on_epoch_end(self, epoch, logs=None):
        self.model.alpha.assign(0)
        self.model.num_imgs_seen.assign(0)


class ModelPlotter(tf.keras.callbacks.Callback):
    def __init__(self, log_dir='logs'):
        super().__init__()
        self.log_dir = log_dir

    def on_epoch_begin(self, epoch, logs=None):

        try:
            tf.keras.utils.plot_model(self.model.E, to_file=os.path.join(self.logs,
                                                                         'encoder_' + str(self.model.E.input_shape[1])
                                                                         + '.png'),
                                      show_shapes=True,
                                      expand_nested=True)

            tf.keras.utils.plot_model(self.model.G, to_file=os.path.join(self.logs,
                                                                         'generator_' + str(
                                                                             self.model.G.output_shape[1])
                                                                         + '.png'),
                                      show_shapes=True,
                                      expand_nested=True)

            tf.keras.utils.plot_model(self.model.D, to_file=os.path.join(self.logs,
                                                                         'discriminator_' + str(
                                                                             self.model.D.output_shape[1])
                                                                         + '.png'),
                                      show_shapes=True,
                                      expand_nested=True)
        except:
            print('Unable to print models. Is GraphViz installed?')

        try:
            tf.keras.utils.plot_model(self.model.E_fade, to_file='encoder_fade_' + str(in_shape[1]) + '.png',
                                      show_shapes=True,
                                      expand_nested=True)
            tf.keras.utils.plot_model(self.model.D_fade, to_file='discriminator_fade_' + str(in_shape[1]) + '.png',
                                      show_shapes=True, expand_nested=True)
            tf.keras.utils.plot_model(self.model.G_fade,
                                      to_file='generator_fade_' + str(model1.output_shape[1]) + '.png',
                                      show_shapes=True,
                                      expand_nested=True)
        except:
            # This will happen on 4x4x3 since there is no fade
            pass
