import tensorflow as tf
from os.path import join
import os
import time
import numpy as np
import threading

from tensorflow import TensorShape

from image_utilities import ImageGenerator
from image_utilities import *

training_summaries = []
LOGDIR = '../tensorboard_dir/'
TRAINING_IMAGES_DIR = '../res/training_images'
MODELS_DIR = '../models'
SAMPLES_DIR = '../res/test'
STYLES_DIR = '../res/styles'

VAE_SAMPLES_FOR_TRAINING_DIR = '../res/training/vae'

QUAD_SIDE = 8

INPUT_DIM = [-1, 128, 128, 3] # [-1, 224, 224, 3]
LATENT_DIMS = 4
use_bn = False
use_pool = True
activation = tf.nn.elu

kl_loss_divisor = float(INPUT_DIM[1]*INPUT_DIM[2])
use_mse_loss = True

LEARNING_RATE = 0.001
NUM_EPOCHS = 2000
MINI_BATCH_SIZE = 32
num_samples_per_epoch = 512

tf.enable_eager_execution()


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

    def make_sample(args):
        mean, st_dev = args
        random_sample = tf.random_normal([tf.keras.backend.shape(mean)[0], LATENT_DIMS])
        return mean + (st_dev * random_sample)

    mean = tf.keras.layers.Dense(LATENT_DIMS)(encoder)
    st_dev = tf.keras.layers.Dense(LATENT_DIMS, kernel_initializer=tf.zeros_initializer())(encoder)
    sample = tf.keras.layers.Lambda(make_sample, output_shape=[None, 4])([mean, st_dev])
    model = tf.keras.Model(inputs=network_input, outputs=[mean, st_dev, sample])

    # mean = tf.keras.layers.Dense(LATENT_DIMS)(encoder)
    # st_dev = tf.keras.layers.Dense(LATENT_DIMS, kernel_initializer=tf.zeros_initializer())(encoder)
    # random_sample = tf.random_normal([tf.shape(encoder)[0], LATENT_DIMS])
    # # sample = tf.keras.layers.Add()([mean, tf.multiply(st_dev, random_sample)])
    # model = tf.keras.Model(inputs=network_input, outputs=[mean, st_dev]) #, st_dev, mean + (st_dev * random_sample)])

    return model


def vae_decoder_keras():

    decoder_dims = [16, 16, 32]
    kernel_radii = [3, 3, 3]
    kernel_strides = [1, 1, 1]
    reshape_multi = 8

    arbitrary_reshape_dim = [int(INPUT_DIM[1] / reshape_multi), int(INPUT_DIM[2] / reshape_multi), INPUT_DIM[3]]
    flat_reshape_dim = arbitrary_reshape_dim[0] * arbitrary_reshape_dim[1] * arbitrary_reshape_dim[2]

    network_input = tf.keras.layers.Input(shape=TensorShape([LATENT_DIMS]), dtype=tf.float32)

    decoder = tf.keras.layers.Dense(flat_reshape_dim, activation=activation)(network_input)
    decoder = tf.keras.layers.Reshape(arbitrary_reshape_dim)(decoder)
    if use_bn:
        decoder = tf.keras.layers.BatchNormalization()(decoder)

    layer = 0

    decoder = tf.keras.layers.Conv2DTranspose(decoder_dims[layer], kernel_size=kernel_radii[layer], strides=kernel_strides[layer], activation=activation, padding='same')(decoder)
    if use_bn:
        decoder = tf.keras.layers.BatchNormalization()(decoder)

    layer += 1

    decoder = tf.keras.layers.Conv2DTranspose(decoder_dims[layer], kernel_size=kernel_radii[layer], strides=kernel_strides[layer], activation=activation, padding='same')(decoder)
    if use_bn:
        decoder = tf.keras.layers.BatchNormalization()(decoder)

    layer += 1

    decoder = tf.keras.layers.Flatten()(decoder)
    decoder = tf.keras.layers.Dense(INPUT_DIM[1] * INPUT_DIM[2] * INPUT_DIM[3], activation=tf.nn.sigmoid)(decoder)
    decoder = tf.keras.layers.Reshape(INPUT_DIM)(decoder)

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
            if step == 5:
                print('loss: ' + str(total_loss.numpy()))

        # print('Training concluded. Saving model...')
        # os.mkdir(join(MODELS_DIR,model_name))
        # saver.save(sess, join(MODELS_DIR, model_name, 'saved_' + model_name), global_step=0)
        # print('Model saved.')


def sample_vae(model_name):
    global training_summaries

    model_dir = join(MODELS_DIR, model_name)
    meta_name = 'saved_' + model_name + '-0.meta'

    image_generator = ImageGenerator()
    image_generator.generate_images('test', 100, INPUT_DIM[1], INPUT_DIM[2])
    image_generator.shuffle('test')
    images = image_generator.get('test')

    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph(os.path.join(model_dir, meta_name))
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))

        encoder_input = sess.graph.get_tensor_by_name('input:0')
        latent_vector = sess.graph.get_tensor_by_name('vae_encoder/sample_latent:0')
        decoder_output = sess.graph.get_tensor_by_name('vae_decoder/sample_image:0')

        for i in range(10):
            sample_vectors, sample_outputs = sess.run([latent_vector, decoder_output], feed_dict={encoder_input:[images[i]]})

            save_image(sample_outputs[0],join(SAMPLES_DIR, str(i) + '_sample.jpg'))
            save_image(images[i], join(SAMPLES_DIR, str(i) + '_truth.jpg'))
            write_vector(sample_vectors[0], join(SAMPLES_DIR, str(i) + '_vector.txt'))

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
            print(weighted_vector)
            sample_output = service.get_image_from_vector(weighted_vector)
            quad_output[x*INPUT_DIM[1]:(x+1)*INPUT_DIM[1],y*INPUT_DIM[2]:(y+1)*INPUT_DIM[2],:] = sample_output

    save_image(quad_output, join(SAMPLES_DIR, 'quad.jpg'))

class VAEService(threading.Thread):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.loaded = False

    def run(self):
        print('VAEService starting...')
        model_dir = join(MODELS_DIR, self.model_name)
        meta_name = 'saved_' + self.model_name + '-0.meta'
        self.session = tf.Session()
        saver = tf.train.import_meta_graph(os.path.join(model_dir, meta_name))
        saver.restore(self.session, tf.train.latest_checkpoint(model_dir))
        self.encoder_input = self.session.graph.get_tensor_by_name('input:0')
        self.latent_vector = self.session.graph.get_tensor_by_name('vae_encoder/sample_latent:0')
        self.decoder_output = self.session.graph.get_tensor_by_name('vae_decoder/sample_image:0')
        self.loaded = True
        print('VAEService running.')

    def get_image_from_vector(self, vector):
        if not self.loaded:
            return None
        sample_outputs = self.session.run(self.decoder_output, feed_dict={self.latent_vector:[vector]})
        return sample_outputs[0]

    def get_vectors_and_samples_from_images(self, images):
        if not self.loaded:
            return None
        vectors, samples = self.session.run([self.latent_vector, self.decoder_output], feed_dict={self.encoder_input: images})
        return vectors, samples

    def close(self):

        self.session.close()

    def wait_for_ready(self, increment = .1):
        while not self.loaded:
            time.sleep(increment)

def generate_vae_samples(model_name, num_samples = 1000):
    service = VAEService(model_name)
    service.start()
    service.wait_for_ready()

    image_1_name = 'test.jpg'
    image_2_name = 'test2.jpg'
    images = [image_1_name, image_2_name]
    images = [os.path.join(STYLES_DIR, x) for x in images]
    images = [open_image(x) for x in images]
    vectors, samples = service.get_vectors_and_samples_from_images(images)
    increment = (vectors[1] - vectors[0]) / num_samples
    for x in range(num_samples):
        image_sample = service.get_image_from_vector(vectors[0] + (x * increment))
        save_image(image_sample, os.path.join(VAE_SAMPLES_FOR_TRAINING_DIR, 'sample_' + str(x) + '.jpg'))


def get_most_recent_vae():
    models = os.listdir(MODELS_DIR)
    models.sort()
    models = [x for x in models if 'VAE' in x]
    return models[-1]

if __name__ == '__main__':
    train_vae_keras()
    # train_vae()
    # sample_vae(get_most_recent_vae())
    # make_quad(get_most_recent_vae())
