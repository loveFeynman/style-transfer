import tensorflow as tf
from os.path import join
import os
import cv2
import time
import numpy as np

from image_generator import ImageGenerator

training_summaries = []
LOGDIR = '../tensorboard_dir/'
TRAINING_IMAGES_DIR = '../res/training_images'
MODELS_DIR = '../models'
SAMPLES_DIR = '../res/test'

LATENT_DIMS = 32

INPUT_DIM = [-1,128,128,3]

LEARNING_RATE = 0.001
NUM_EPOCHS = 2000
MINI_BATCH_SIZE = 32
num_samples_per_epoch = 512

use_bn = True
use_pool = True
activation = tf.nn.relu


def vae_encoder(input):

    with tf.name_scope('vae_encoder'):
        encoder_num_filters = [16, 16, 32]
        kernel_radii = [5,5,5]
        filter_strides = [2,2,2]

        pool_sizes = [2,2,2]
        pool_strides = [2,2,2]

        encoder_hidden_0 = tf.layers.conv2d(input,encoder_num_filters[0],kernel_radii[0],filter_strides[0],activation=activation,padding='same')
        if use_bn:
            encoder_hidden_0 = tf.layers.batch_normalization(encoder_hidden_0)
        if use_pool:
            encoder_hidden_0 = tf.layers.max_pooling2d(encoder_hidden_0,pool_sizes[0],pool_strides[0],padding='same')

        encoder_hidden_1 = tf.layers.conv2d(encoder_hidden_0,encoder_num_filters[1],kernel_radii[1],filter_strides[1],activation=activation,padding='same')
        if use_bn:
            encoder_hidden_1 = tf.layers.batch_normalization(encoder_hidden_1)
        if use_pool:
            encoder_hidden_1 = tf.layers.max_pooling2d(encoder_hidden_1,pool_sizes[1],pool_strides[1],padding='same')

        # encoder_hidden_2 = tf.layers.conv2d(encoder_hidden_1, encoder_num_filters[2], kernel_radii[2], filter_strides[2],activation=activation, padding='same')
        # if use_bn:
        #     encoder_hidden_2 = tf.layers.batch_normalization(encoder_hidden_2)
        # if use_pool:
        #     encoder_hidden_2 = tf.layers.max_pooling2d(encoder_hidden_2, pool_sizes[2], pool_strides[2], padding='same')

        flat = tf.layers.flatten(encoder_hidden_1)

        mean = tf.layers.dense(flat, LATENT_DIMS, name='mean')
        st_dev = tf.layers.dense(flat, LATENT_DIMS, name='st_dev', kernel_initializer=tf.zeros_initializer())
        random_sample = tf.random_normal([tf.shape(flat)[0],LATENT_DIMS])
        sample = tf.add(mean,tf.multiply(st_dev, random_sample),name='sample_latent')

    return mean, st_dev, sample

def vae_decoder(input):
    with tf.name_scope('vae_decoder'):

        decoder_dims = [16, 16, 32]
        kernel_radii = [3, 3, 3]
        kernel_strides = [1, 1, 1]

        arbitrary_reshape_dim = [-1, 16, 16, INPUT_DIM[3]]

        decoder_leading_dense_dim = arbitrary_reshape_dim[1] * arbitrary_reshape_dim[2] * arbitrary_reshape_dim[3]
        decoder_hidden_0 = tf.layers.dense(input, 64, activation=tf.nn.elu)

        decoder_hidden_1 = tf.layers.dense(decoder_hidden_0, decoder_leading_dense_dim, activation=activation)
        decoder_hidden_1 = tf.reshape(decoder_hidden_1, arbitrary_reshape_dim)
        if use_bn:
            decoder_hidden_1 = tf.layers.batch_normalization(decoder_hidden_1)

        decoder_hidden_2 = tf.layers.conv2d_transpose(decoder_hidden_1, decoder_dims[0], kernel_size=kernel_radii[0], strides=kernel_strides[0], activation=activation, padding='same')
        if use_bn:
            decoder_hidden_2 = tf.layers.batch_normalization(decoder_hidden_2)

        decoder_hidden_3 = tf.layers.conv2d_transpose(decoder_hidden_2, decoder_dims[0], kernel_size=kernel_radii[1], strides=kernel_strides[1], activation=activation, padding='same')
        if use_bn:
            decoder_hidden_3 = tf.layers.batch_normalization(decoder_hidden_3)

        # decoder_hidden_4 = tf.layers.conv2d_transpose(decoder_hidden_3, decoder_dims[0], kernel_size=kernel_radii[2], strides=kernel_strides[2], activation=activation, padding='same')
        # if use_bn:
        #     decoder_hidden_4 = tf.layers.batch_normalization(decoder_hidden_4)

        decoder_flatten_4 = tf.layers.flatten(decoder_hidden_3)
        decoder_hidden_5 = tf.layers.dense(decoder_flatten_4, INPUT_DIM[1] * INPUT_DIM[2] * INPUT_DIM[3], activation=tf.nn.sigmoid)
        decoder_reshape_5 = tf.reshape(decoder_hidden_5, INPUT_DIM, name='sample_image')

    return decoder_reshape_5

def train_vae():
    global training_summaries

    input_image = tf.placeholder(tf.float32, [None, INPUT_DIM[1], INPUT_DIM[2], INPUT_DIM[3]], name='input')
    #labels = tf.placeholder(tf.float32, [None, LATENT_DIMS], name='input')
    mean, st_dev, sample_latent = vae_encoder(input_image)
    sample_image = vae_decoder(sample_latent)
    view_image = tf.cast(sample_image * 255., tf.uint8)


    with tf.name_scope('Loss'):
        #loss_image = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(input_image,sample_image)) / tf.cast(tf.shape(input_image)[0],tf.float32))
        loss_image = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=sample_image, labels=input_image, name='loss_image'))
        loss_kl = tf.reduce_mean(tf.reduce_mean(-0.5 * tf.reduce_sum(1 + st_dev - tf.square(mean) - tf.exp(st_dev), 1))) / 1000 # don't know why, but this 1000 seems to help A LOT
        loss = tf.identity(loss_image + loss_kl, name='loss')
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE,name='optimizer').minimize(loss)
        training_summaries.append(tf.summary.scalar('Image_Loss', loss_image))
        training_summaries.append(tf.summary.scalar('KL_Loss', loss_kl))
        training_summaries.append(tf.summary.scalar('Total_Loss', loss))

    with tf.name_scope('vae_ground_truth'):
        training_summaries.append(tf.summary.image('Input_Image', input_image, max_outputs=4))
    with tf.name_scope('vae_sample_image'):
        training_summaries.append(tf.summary.image('Sample_Image', sample_image, max_outputs=4)) #switch sample_image with view_image?

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        merged_training_summaries = tf.summary.merge(training_summaries)
        model_name = time.strftime("VAE_%Y-%m-%d-%H-%M")
        save_dir = join(LOGDIR, model_name)
        writer = tf.summary.FileWriter(save_dir)
        writer.add_graph(sess.graph)
        saver = tf.train.Saver()

        image_generator = ImageGenerator()
        image_generator.generate_images('train', num_samples_per_epoch, INPUT_DIM[1],INPUT_DIM[2])
        # print(input_shape, output_shape, np.shape(cv_data), np.shape(cv_labels))

        for epoch in range(NUM_EPOCHS):
            print("Epoch ", epoch + 1, " out of ", NUM_EPOCHS, " epochs.")
            num_training_steps = int(num_samples_per_epoch / MINI_BATCH_SIZE) + 1

            image_generator.shuffle('train')
            train_data = image_generator.get('train')

            for step in range(num_training_steps):
                start = min(num_samples_per_epoch, step * MINI_BATCH_SIZE)
                end = min(num_samples_per_epoch, (step + 1) * MINI_BATCH_SIZE)

                batch_x = train_data[start:end]

                if start == end:
                    continue

                if step % 5 == 0:
                    [s] = sess.run([merged_training_summaries], feed_dict={input_image: batch_x})
                    writer.add_summary(s, num_training_steps * epoch + step)
                    writer.flush()

                sess.run([optimizer, loss], feed_dict={input_image: batch_x})

        os.mkdir(join(MODELS_DIR,model_name))
        saver.save(sess, join(MODELS_DIR, model_name, 'saved_' + model_name), global_step=0)

def write_vector(vector,path):
    with open(path, 'w+') as file:
        file.write(str(vector))

def save_image(img,path):
    image = (img*255.).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path,image)

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

    model_dir = join(MODELS_DIR, model_name)
    meta_name = 'saved_' + model_name + '-0.meta'

    image_generator = ImageGenerator()
    image_generator.generate_images('test', 4, INPUT_DIM[1], INPUT_DIM[2])
    image_generator.shuffle('test')
    images = image_generator.get('test')

    QUAD_SIDE = 32

    quad_output = np.zeros([QUAD_SIDE * INPUT_DIM[1], QUAD_SIDE * INPUT_DIM[2],INPUT_DIM[3]], dtype=np.float32)

    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph(os.path.join(model_dir, meta_name))
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))

        encoder_input = sess.graph.get_tensor_by_name('input:0')
        latent_vector = sess.graph.get_tensor_by_name('vae_encoder/sample_latent:0')
        decoder_output = sess.graph.get_tensor_by_name('vae_decoder/sample_image:0')

        base_vectors, base_outputs = sess.run([latent_vector, decoder_output], feed_dict={encoder_input: images})
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
                weights = np.array([[r0] * LATENT_DIMS, [r1] * LATENT_DIMS, [r2] * LATENT_DIMS, [r3] * LATENT_DIMS]).reshape((4,32))
                weighted_vector = np.sum(np.multiply(base_vectors,weights), axis=0)

                sample_outputs = sess.run([decoder_output], feed_dict={latent_vector: [weighted_vector]})
                quad_output[x*INPUT_DIM[1]:(x+1)*INPUT_DIM[1],y*INPUT_DIM[2]:(y+1)*INPUT_DIM[2],:] = sample_outputs[0]

        save_image(quad_output, join(SAMPLES_DIR, 'quad.jpg'))




if __name__ == '__main__':
    # train_vae()
    # sample_vae('VAE_2019-07-11-22-55')
    make_quad('VAE_2019-07-11-22-55')
