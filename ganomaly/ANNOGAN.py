import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.models import Model

from ganomaly.layers import WScaleConv2DLayer as WScaleLayer
from ganomaly.layers import WeightedSum, PixelNormalization
from ganomaly.losses import interpolate_imgs, gradient_penalty_loss, discriminator_loss, generator_loss, encoder_loss

init = tf.keras.initializers.VarianceScaling()  # weight initialization
const = tf.keras.constraints.MinMaxNorm(min_value=-1.0, max_value=1.0)  # weight constraint


class ANNOGAN(tf.keras.Model):
    def __init__(self, G, D, E, latent_dim, num_imgs_to_fade=5000):
        super(ANNOGAN, self).__init__()
        self.G = G
        self.D = D
        self.E = E
        self.G_fade = None
        self.E_fade = None
        self.D_fade = None
        self.generator = None
        self.discriminator = None
        self.encoder = None
        self.latent_dim = latent_dim
        self.generator_optimizer = None
        self.discriminator_optimizer = None
        self.encoder_optimizer = None
        self.alpha = tf.Variable(1.0, dtype=tf.float32, name='alpha')
        self.num_imgs_seen = tf.Variable(0, dtype=tf.int32, name='num_imgs_seen')
        self.num_imgs_to_fade = num_imgs_to_fade
        self.fade = tf.Variable(False, dtype=tf.float32, name='fade')
        self.first_round = tf.Variable(True, name='first_round')

    # noinspection PyMethodOverriding
    def compile(self, generator_optimizer, discriminator_optimizer, encoder_optimizer, **kwargs):
        super(ANNOGAN, self).compile()
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.encoder_optimizer = encoder_optimizer

    def train_step(self, real_images, _lambda=10):
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))


        # TODO: Use faded model or non-faded. DOES NOT WORK AS INTENDED!!!
        # determine whether or not to use the faded model
        if self.fade is False:
            return self._train(real_images, self.G, self.D, self.E, random_latent_vectors, batch_size)
        else:
            try:
                return self._train(real_images, self.G_fade, self.D_fade, self.E_fade, random_latent_vectors, batch_size)
            except:
                return self._train(real_images, self.G, self.D, self.E, random_latent_vectors, batch_size)

        #res = self._train(real_images, self.G, self.D, self.E, random_latent_vectors, batch_size)
        #return res

    def _train(self, real_images, generator, discriminator, encoder, random_latent_vectors, batch_size, _lambda=10):
        #############################################################################
        # Train Discriminator
        #############################################################################
        with tf.GradientTape() as disc_tape:
            generated_imgs = generator(random_latent_vectors, training=False)
            generated_output = discriminator(generated_imgs, training=True)
            real_output = discriminator(real_images, training=True)

            with tf.GradientTape() as interp_tape:
                interp_tape.watch(real_output)
                interp_tape.watch(generated_output)
                interp_tape.watch(generated_imgs)

                interpolated_img = interpolate_imgs(real_images, generated_imgs, batch_size)
                validity_interpolated = discriminator(interpolated_img, training=True)
                real_score, fake_score, disc_loss = discriminator_loss(real_output, generated_output)

                grads = interp_tape.gradient(validity_interpolated, interpolated_img)
            grad_penalty = gradient_penalty_loss(grads)
            disc_loss += 0.5 * _lambda * grad_penalty

        grad_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(grad_disc, discriminator.trainable_variables))

        #############################################################################
        # Train Generator
        #############################################################################
        with tf.GradientTape() as gen_tape:
            generated_imgs = generator(random_latent_vectors, training=True)
            generated_output = discriminator(generated_imgs, training=False)
            gen_loss = generator_loss(generated_output)

        grad_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(grad_gen, generator.trainable_variables))

        #############################################################################
        # Train Encoder
        #############################################################################
        with tf.GradientTape() as enc_tape:
            z_hat = encoder(real_images, training=True)
            images_reconstructed = generator(z_hat, training=False)
            enc_loss = encoder_loss(real_images, images_reconstructed)

        grad_enc = enc_tape.gradient(enc_loss, encoder.trainable_variables)
        self.encoder_optimizer.apply_gradients(zip(grad_enc, encoder.trainable_variables))

        return {'gen_loss': gen_loss, 'enc_loss': enc_loss, 'disc_loss': disc_loss, 'alpha': self.alpha,
                'num_imgs_seen': self.num_imgs_seen, 'fade': self.fade}


    #############################################################################
    # Expand the model architecture and add faded versions
    #############################################################################

    def add_generator_block(self):
        # get the end of the last block
        block_end = self.G.layers[-2].output
        # upsample, and define new block
        upsampling = UpSampling2D()(block_end)
        latent_depth = self.latent_dim
        g = WScaleLayer(latent_depth, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(
            upsampling)
        g = PixelNormalization()(g)
        g = LeakyReLU(alpha=0.2)(g)
        g = WScaleLayer(latent_depth, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
        g = PixelNormalization()(g)
        g = LeakyReLU(alpha=0.2)(g)
        # add new output layer
        out_image = Conv2D(3, (1, 1), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
        # define model
        model1 = Model(self.G.input, out_image)
        # get the output layer from old model
        out_old = self.G.layers[-1]
        # connect the upsampling to the old output layer
        out_image2 = out_old(upsampling)
        # define new output image as the weighted sum of the old and new models
        merged = WeightedSum()([out_image2, out_image])
        # define model
        model2 = Model(self.G.input, merged)
        self.G = model1
        self.G_fade = model2

    def add_discriminator_block(self, old_model, LATENT_DEPTHS, style='discriminator', n_input_layers=3):
        # get shape of existing model
        in_shape = list(old_model.input.shape)
        # define new input shape as double the size
        input_shape = (in_shape[-2] * 2, in_shape[-2] * 2, in_shape[-1])
        in_image = Input(shape=input_shape)
        latent_depth = int(LATENT_DEPTHS[str(input_shape[-2])])  # Latent depth of existing image
        latent_depth_next = int(LATENT_DEPTHS[str(in_shape[-2])])  # Latent dim of next size image
        # define new input processing layer
        # FROM RGB
        d = WScaleLayer(latent_depth, (1, 1), padding='same', kernel_initializer=init, kernel_constraint=const)(
            in_image)
        d = LeakyReLU(alpha=0.2)(d)
        # define new block
        d = WScaleLayer(latent_depth, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = WScaleLayer(latent_depth_next, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = AveragePooling2D()(d)
        block_new = d

        # skip the input, 1x1 and activation for the old model
        for i in range(n_input_layers, len(old_model.layers)):
            d = old_model.layers[i](d)

        model1 = Model(in_image, d)

        # Downsample and convert from RGB
        downsample = AveragePooling2D()(in_image)
        block_old = old_model.layers[1](downsample)
        block_old = old_model.layers[2](block_old)
        d = WeightedSum()([block_old, block_new])

        # skip the input, 1x1 and activation for the old model
        for i in range(n_input_layers, len(old_model.layers)):
            d = old_model.layers[i](d)

        model2 = Model(in_image, d)
        if style == 'discriminator':
            self.D = model1
            self.D_fade = model2

        else:
            self.E = model1
            self.E_fade = model2

    def reset_alpha(self):
        self.alpha.assign(0.)
        self.num_imgs_seen.assign(0)
        self.first_round.assign(False)
        self.fade.assign(True)
