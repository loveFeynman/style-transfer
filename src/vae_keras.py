import tensorflow as tf
from os.path import join
import os
import time
import numpy as np
import threading

from tensorflow import TensorShape
from tensorflow.python.keras.engine import Layer
from tensorflow.python.keras.saving import model_from_json, load_model

from image_utilities import ImageGenerator
from image_utilities import *
from constants import *
from model_utilities import save_keras_model, load_keras_model, get_most_recent_model_name

training_summaries = []

model_prefix = 'VAE_KERAS'

QUAD_SIDE = 8

INPUT_DIM = [-1, 224, 224, 3] # [-1, 224, 224, 3]
LATENT_DIMS = 4
use_bn = False
use_pool = True
activation = tf.nn.elu

kl_loss_divisor = float(INPUT_DIM[1]*INPUT_DIM[2])
use_mse_loss = True

LEARNING_RATE = 0.001
NUM_EPOCHS = 4000
MINI_BATCH_SIZE = 64
num_samples_per_epoch = 512

tf.enable_eager_execution()

class SampleLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x):
        mean, st_dev = x
        random_sample = tf.random_normal([tf.keras.backend.shape(mean)[0], LATENT_DIMS])
        return mean + (st_dev * random_sample)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        base_config = super().get_config()
        return base_config


def vae_encoder_keras():
    encoder_num_filters = [16, 16, 32]
    kernel_radii = [5, 5, 5]
    filter_strides = [2, 2, 2]

    pool_sizes = [2, 2, 2]
    pool_strides = [2, 2, 2]

    layer = 0

    network_input = tf.keras.layers.Input(shape=(INPUT_DIM[1], INPUT_DIM[2], INPUT_DIM[3]))

    encoder = tf.keras.layers.Conv2D(encoder_num_filters[layer], kernel_radii[layer], filter_strides[layer], activation=activation, padding='valid')(network_input)
    if use_bn:
        encoder = tf.keras.layers.BatchNormalization()(encoder)
    if use_pool:
        encoder = tf.keras.layers.MaxPool2D(pool_sizes[layer], pool_strides[layer])(encoder)

    layer += 1

    encoder = tf.keras.layers.Conv2D(encoder_num_filters[layer], kernel_radii[layer], filter_strides[layer], activation=activation, padding='valid')(encoder)
    if use_bn:
        encoder = tf.keras.layers.BatchNormalization()(encoder)
    if use_pool:
        encoder = tf.keras.layers.MaxPool2D(pool_sizes[layer], pool_strides[layer])(encoder)

    layer += 1

    encoder = tf.keras.layers.Flatten()(encoder)

    mean = tf.keras.layers.Dense(LATENT_DIMS)(encoder)
    st_dev = tf.keras.layers.Dense(LATENT_DIMS, kernel_initializer=tf.zeros_initializer())(encoder)
    # sample = tf.keras.layers.Lambda(make_sample, output_shape=[None, 4])([mean, st_dev])
    sample = SampleLayer()([mean, st_dev])
    model = tf.keras.Model(inputs=network_input, outputs=[mean, st_dev, sample])

    return model


def vae_decoder_keras():

    decoder_dims = [64, 64, 64, INPUT_DIM[3]]
    kernel_radii = [3, 3, 3, 3]
    kernel_strides = [1, 1, 1, 1]


    arbitrary_reshape_dim = [int(INPUT_DIM[1] - 4), int(INPUT_DIM[2] - 4), INPUT_DIM[3]]
    flat_reshape_dim = arbitrary_reshape_dim[0] * arbitrary_reshape_dim[1] * arbitrary_reshape_dim[2]

    network_input = tf.keras.layers.Input(shape=TensorShape([LATENT_DIMS]), dtype=tf.float32)

    decoder = tf.keras.layers.Dense(flat_reshape_dim, activation=activation)(network_input)
    decoder = tf.keras.layers.Reshape(arbitrary_reshape_dim)(decoder)
    if use_bn:
        decoder = tf.keras.layers.BatchNormalization()(decoder)

    layer = 0

    decoder = tf.keras.layers.Conv2DTranspose(decoder_dims[layer], kernel_size=kernel_radii[layer], strides=kernel_strides[layer], activation=activation, padding='valid')(decoder)
    if use_bn:
        decoder = tf.keras.layers.BatchNormalization()(decoder)

    layer += 1

    decoder = tf.keras.layers.Conv2DTranspose(decoder_dims[layer], kernel_size=kernel_radii[layer], strides=kernel_strides[layer], activation=activation, padding='valid')(decoder)
    if use_bn:
        decoder = tf.keras.layers.BatchNormalization()(decoder)

    layer += 1

    decoder = tf.keras.layers.Conv2DTranspose(decoder_dims[layer], kernel_size=kernel_radii[layer], strides=kernel_strides[layer], activation=activation, padding='same')(decoder)
    if use_bn:
        decoder = tf.keras.layers.BatchNormalization()(decoder)

    layer += 1

    decoder = tf.keras.layers.Conv2DTranspose(decoder_dims[layer], kernel_size=kernel_radii[layer], strides=kernel_strides[layer], activation=tf.nn.sigmoid, padding='same')(decoder)

    model = tf.keras.Model(inputs=network_input, outputs=decoder)

    return model


def loss(input_image, sample_image, mean, st_dev):

    loss_image = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(input_image, sample_image)) / tf.cast(tf.shape(input_image)[0], tf.float32))
    if not use_mse_loss:
        loss_image = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=sample_image, labels=input_image))
    loss_kl = tf.reduce_mean(tf.reduce_mean(-0.5 * tf.reduce_sum(1 + st_dev - tf.square(mean) - tf.exp(st_dev), 1))) / kl_loss_divisor
    return loss_image + loss_kl


def train_vae_keras():
    encoder = vae_encoder_keras()
    decoder = vae_decoder_keras()

    opt = tf.keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=.99, epsilon=1e-1)

    image_generator = ImageGenerator()
    image_generator.generate_images('train', num_samples_per_epoch, INPUT_DIM[1], INPUT_DIM[2])

    def train_step(input_image):
        with tf.GradientTape() as tape:
            [mean, st_dev, sample_latent] = encoder(input_image)
            sample_image = decoder(sample_latent)
            total_loss = loss(input_image, sample_image, mean, st_dev)

        variables = encoder.trainable_variables + decoder.trainable_variables
        grad = tape.gradient(total_loss, variables)
        opt.apply_gradients(zip(grad, variables))
        return total_loss

    train_start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print("Epoch ", epoch + 1, " out of ", NUM_EPOCHS, " epochs.")
        num_training_steps = int(num_samples_per_epoch / MINI_BATCH_SIZE) + 1

        image_generator.shuffle('train')
        train_data = image_generator.get('train')

        for step in range(num_training_steps):
            start = min(num_samples_per_epoch, step * MINI_BATCH_SIZE)
            end = min(num_samples_per_epoch, (step + 1) * MINI_BATCH_SIZE)

            batch = np.array(train_data[start:end])

            if start == end:
                continue

            total_loss = train_step(batch)
            if step == 0:

                test_img = train_data[0]
                [mean, st_dev, sample_latent] = encoder(np.array([test_img]))
                sample_image = decoder(sample_latent)
                sampled_img = sample_image[0]
                keras_samples_dir = 'keras_samples'
                save_image(test_img, os.path.join(TEST_DIR, keras_samples_dir, 'vae_keras_' + str(epoch) + '_truth.jpg'))
                save_image(sampled_img.numpy(), os.path.join(TEST_DIR, keras_samples_dir, 'vae_keras_' + str(epoch) + '_sample.jpg'))
                epoch_end = time.time()
                elapsed = epoch_end - train_start_time
                time_digits = 6
                ETA = ((epoch_end - train_start_time) / (1 + epoch)) * (NUM_EPOCHS - (epoch + 1))
                print('loss: ' + str(total_loss.numpy()) + ' | elapsed: ' + str(elapsed/60.)[:time_digits] + ' min | remaining training time: ' + str(ETA/60.)[:time_digits] + ' min')




        # print('Training concluded. Saving model...')
        # os.mkdir(join(MODELS_DIR,model_name))
        # saver.save(sess, join(MODELS_DIR, model_name, 'saved_' + model_name), global_step=0)
        # print('Model saved.')

    model_name = model_prefix + '_' + str(time.time())[:10]
    save_keras_model((encoder, decoder), MODELS_DIR, model_name)

#
# def save_vae(encoder, decoder, model_name):
#     dir_name = os.path.join(MODELS_DIR, model_prefix + '_' + model_name)
#     os.mkdir(dir_name)
#     save_keras_model(encoder, dir_name, model_prefix + '_encoder_' + model_name)
#     save_keras_model(decoder, dir_name, model_prefix + '_decoder_' + model_name)
#
#
# def load_vae(model_name):
#     dir_name = os.path.join(MODELS_DIR, model_prefix + '_' + model_name)
#     encoder = load_keras_model(dir_name, model_prefix + '_encoder_' + model_name)
#     decoder = load_keras_model(dir_name, model_prefix + '_decoder_' + model_name)
#     return encoder, decoder
#
#
# def save_keras_model(model, dir, name):
#     # model.save_weights(os.path.join(dir, name + '.h5'))
#     # with open(os.path.join(dir, name + '.json'), 'w+') as js:
#     #     js.write(model.to_json())
#     model.save(os.path.join(dir, name + '.h5'))
#
#
# def load_keras_model(dir, name):
#     model = load_model(os.path.join(dir, name + '.h5'), custom_objects={'SampleLayer': SampleLayer})
#     # with open(os.path.join(dir, name + '.json')) as js:
#     #     model = model_from_json(js.read())
#     # model.load_weights(os.path.join(dir, name + '.h5'))
#     return model
#
#
# def get_most_recent_vae_name():
#     models = os.listdir(MODELS_DIR)
#     models.sort()
#     models = [x for x in models if model_prefix in x]
#     return models[-1].split('_')[-1]


def make_quad(model_name):
    global training_summaries

    image_generator = ImageGenerator()
    image_generator.generate_images('test', 4, INPUT_DIM[1], INPUT_DIM[2])
    image_generator.shuffle('test')
    images = image_generator.get('test')

    quad_output = np.zeros([QUAD_SIDE * INPUT_DIM[1], QUAD_SIDE * INPUT_DIM[2],INPUT_DIM[3]], dtype=np.float32)

    service = VAEService(model_name)
    service.start()
    while not service.loaded:
        time.sleep(.25)

    base_vectors, base_outputs = service.get_vectors_and_samples_from_images(images)

    # 0 1
    # 2 3
    print('constructing ' + str(QUAD_SIDE) + 'x' + str(QUAD_SIDE) + ' quad')
    for x in range(QUAD_SIDE):
        for y in range(QUAD_SIDE):
            x_r = float(x)/float(QUAD_SIDE)
            y_r = float(y)/float(QUAD_SIDE)
            r0 = (1. - x_r) * (1. - y_r)
            r1 = x_r * (1. - y_r)
            r2 = (1. - x_r) * y_r
            r3 = x_r * y_r
            weights = np.array([[r0] * LATENT_DIMS, [r1] * LATENT_DIMS, [r2] * LATENT_DIMS, [r3] * LATENT_DIMS]).reshape((4,LATENT_DIMS))
            weighted_vector = np.sum(np.multiply(base_vectors,weights), axis=0)
            sample_output = service.get_image_from_vector(weighted_vector)
            quad_output[x*INPUT_DIM[1]:(x+1)*INPUT_DIM[1],y*INPUT_DIM[2]:(y+1)*INPUT_DIM[2],:] = sample_output

    save_image(quad_output, join(TEST_DIR, 'quad.jpg'))

class VAEService(threading.Thread):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.loaded = False

    def run(self):
        print('VAEService starting...')
        [self.encoder, self.decoder] = load_keras_model(MODELS_DIR, self.model_name, special_layers={'SampleLayer': SampleLayer})
        self.loaded = True
        print('VAEService running.')

    def get_image_from_vector(self, vector):
        if not self.loaded:
            return None
        sample_outputs = self.decoder(np.reshape(vector, (1, LATENT_DIMS))).numpy()
        return sample_outputs[0]

    def get_vectors_and_samples_from_images(self, images):
        if not self.loaded:
            return None
        network_inputs = images
        if isinstance(images, list):
            network_inputs = np.array(images)
        if len(network_inputs.shape) == 3:
            network_inputs = network_inputs.reshape((1, network_inputs.shape[0], network_inputs.shape[1], network_inputs.shape[2]))
        [mean, stdev, vectors] = self.encoder(network_inputs)
        samples = self.decoder(vectors).numpy()
        return vectors, samples

    def close(self):
        pass

    def wait_for_ready(self, increment = .1):
        while not self.loaded:
            time.sleep(increment)

def sample_vae(model_name):
    service = VAEService(model_name)
    service.start()
    service.wait_for_ready()

    image_generator = ImageGenerator()
    image_generator.generate_images('test', 100, INPUT_DIM[1], INPUT_DIM[2])
    image_generator.shuffle('test')
    images = image_generator.get('test')

    vectors, samples_outputs = service.get_vectors_and_samples_from_images(images)
    if not os.path.exists(VAE_KERAS_GENERATED_SAMPLES_DIR):
        os.mkdir(VAE_KERAS_GENERATED_SAMPLES_DIR)

    for x in range(len(samples_outputs)):
        save_image(samples_outputs[x],join(VAE_KERAS_GENERATED_SAMPLES_DIR, str(x) + '_sample.jpg'))
        save_image(images[x], join(VAE_KERAS_GENERATED_SAMPLES_DIR, str(x) + '_truth.jpg'))

def generate_stylizer_training_data(model_name, samples_to_generate = 1000, use_randomness = True):
    service = VAEService(model_name)
    service.start()
    service.wait_for_ready()

    image_generator = ImageGenerator()
    image_generator.generate_images('test', 100, INPUT_DIM[1], INPUT_DIM[2])
    images = image_generator.get('test')

    samples_per_class = int(len(images)/2)

    vectors_1, _ = service.get_vectors_and_samples_from_images(images[:samples_per_class])
    vectors_2, _ = service.get_vectors_and_samples_from_images(images[samples_per_class:])

    mean_vector_style_1 = np.mean(vectors_1, axis=0)
    mean_vector_style_2 = np.mean(vectors_2, axis=0)
    stdev_style_1 = np.std(vectors_1, axis=0)
    stdev_style_2 = np.std(vectors_2, axis=0)

    random_samples_1 = np.zeros((samples_to_generate, LATENT_DIMS)) if not use_randomness else np.random.randn(samples_to_generate, LATENT_DIMS)
    random_samples_2 = np.zeros((samples_to_generate, LATENT_DIMS)) if not use_randomness else np.random.randn(samples_to_generate, LATENT_DIMS)

    sampled_vectors_1 = (random_samples_1 * stdev_style_1) + mean_vector_style_1
    sampled_vectors_2 = (random_samples_2 * stdev_style_2) + mean_vector_style_2

    scalars_1 = np.arange(float(samples_to_generate))/float(samples_to_generate)
    scalars_2 = np.arange(float(samples_to_generate), 0., -1.)/float(samples_to_generate)

    scalars_1 = np.reshape(scalars_1, (samples_to_generate,1))
    scalars_2 = np.reshape(scalars_2, (samples_to_generate,1))

    vectors_to_sample = (sampled_vectors_1 * scalars_1) + (sampled_vectors_2 * scalars_2)

    print('Generating training data...')

    digits_to_include = 6

    for i in range(samples_to_generate):
        name = 'sample_' + str(scalars_1[i][0])[:digits_to_include] + '_' + str(scalars_2[i][0])[:digits_to_include]
        sampled_image = service.get_image_from_vector(vectors_to_sample[i])
        save_image(sampled_image, os.path.join(VAE_KERAS_GENERATED_TRAINING_IMAGES, name + '.jpg'))


if __name__ == '__main__':
    # train_vae_keras()

    # sample_vae(get_most_recent_vae_name())
    make_quad(get_most_recent_model_name(MODELS_DIR, model_prefix))
    # encoder, decoder = load_vae(get_most_recent_vae_name())

    # generate_stylizer_training_data(get_most_recent_vae_name(), 200, False)
