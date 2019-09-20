import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

import threading
import time

import tensorflow as tf
import numpy as np

import constants
import model_utilities
from image_utilities import load_image, CocoDatasetManager, save_image
from model_utilities import LayerConfig, StyleTransfer

LEARNING_RATE = 1e-3
EPOCHS = 30 #160
BATCH_SIZE = 4
TOTAL_IMAGES = 1000

STYLE_IMAGE_1 = os.path.join(constants.STYLES_DIR, 'starry_night_small.jpg')
STYLE_IMAGE_1_LARGE = os.path.join(constants.STYLES_DIR, 'starry_night.jpg')
STYLE_IMAGE_2_LARGE = os.path.join(constants.STYLES_DIR, 'honeycomb_large.jpg')
STYLE_IMAGE_2_SQUEEZE = os.path.join(constants.STYLES_DIR, 'honeycomb_squeeze.jpg')

CONTENT_IMAGE_1 = os.path.join(constants.CONTENT_DIR, 'antelope_small.jpg')
CONTENT_IMAGE_2 = os.path.join(constants.CONTENT_DIR, 'sparrow_small.jpg')
CONTENT_IMAGE_2_LARGE = os.path.join(constants.CONTENT_DIR, 'sparrow.jpg')
CONTENT_IMAGE_3 = os.path.join(constants.CONTENT_DIR, 'museum.jpg')

model_prefix = 'STYLE_NET'

def build_vgg_network():
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    style_layers = [vgg.get_layer(x).output for x in StyleTransfer.VGG_STYLE_TARGET_LAYER_NAMES]
    content_layers = [vgg.get_layer(x).output for x in StyleTransfer.VGG_CONTENT_TARGET_LAYER_NAMES]

    style_layers = [StyleTransfer.gram_matrix(x) for x in style_layers]

    return tf.keras.Model(inputs=vgg.input, outputs=[style_layers, content_layers])

def build_style_network():
    conv_layer_configs = [LayerConfig(32, 9, 1),
                          LayerConfig(64, 3, 2),
                          LayerConfig(128, 3, 2)]

    residual_block_configs = [LayerConfig(128, 3, 1),
                              LayerConfig(128, 3, 1),
                              LayerConfig(128, 3, 1),
                              LayerConfig(128, 3, 1),
                              LayerConfig(128, 3, 1)]

    deconv_layer_configs = [LayerConfig(64, 3, 2),
                            LayerConfig(32, 3, 2),
                            LayerConfig(3, 9, 1, activation=tf.nn.sigmoid)]


    model_builder = model_utilities.EagerModelBuilder
    network_input = tf.keras.layers.Input(shape=(None, None, 3))
    layer = network_input

    for x in conv_layer_configs:
        layer = model_builder.conv_block(layer, x)

    for x in residual_block_configs:
        layer = model_builder.residual_block(layer, x)

    for x in deconv_layer_configs:
        layer = model_builder.deconv_block(layer, x)

    def stringify_config_list(l):
        return [x.__repr__() for x in l]

    layer_configs = {
        'conv_layers': stringify_config_list(conv_layer_configs),
        'residual_layers': stringify_config_list(residual_block_configs),
        'deconv_layers': stringify_config_list(deconv_layer_configs)
    }


    return tf.keras.Model(inputs=network_input, outputs=layer), layer_configs

def vgg_preprocess(network_input):
    if isinstance(network_input, np.ndarray):
        return tf.keras.applications.vgg19.preprocess_input(network_input* 225.)
    else:
        return tf.keras.applications.vgg19.preprocess_input(tf.multiply(network_input, 225.))


def train_style_network(style_type, test_content_image_path=CONTENT_IMAGE_1):
    style_config = StyleTransfer.STYLE_CONFIG_DICT[style_type]
    style_image = load_image(style_config.style_path)
    test_content_image = load_image(test_content_image_path)

    print('Building networks...')
    with tf.name_scope('style_network'):
        style_network, style_network_configs = build_style_network()
    with tf.name_scope('vgg_network'):
        vgg_network = build_vgg_network()
    print('Done building networks.')

    print('Loading images...')
    dataset_manager = CocoDatasetManager(target_dim=(224, 224), num_images=TOTAL_IMAGES)
    print('Done loading images.')

    style_network_input = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='style_network_input')
    vgg_network_input = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='vgg_network_input')
    [style_targets, content_targets] = vgg_network(vgg_preprocess(vgg_network_input))
    style_network_output = style_network(style_network_input)
    style_layers, content_layers = vgg_network(vgg_preprocess(style_network_output))
    style_target_placeholder = [tf.placeholder(tf.float32, shape=[None, None, None], name='style_target_placeholder_' + str(x)) for x in range(len(style_layers))]
    content_target_placeholder = [tf.placeholder(tf.float32, shape=[None, None, None, None], name='content_target_placeholder')]

    loss, style_loss, content_loss, total_var_loss = StyleTransfer.total_loss(style_network_output,
                                                                              style_target_placeholder, content_target_placeholder,
                                                                              style_layers, content_layers,
                                                                              style_weight=style_config.style_weight,
                                                                              content_weight=style_config.content_weight,
                                                                              total_variation_weight=style_config.total_variation_weight)
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE) #learning_rate=0.02, beta1=.99, epsilon=1e-1)
    optimizer_op = optimizer.minimize(loss, var_list=[style_network.trainable_variables])

    training_summaries = []
    with tf.name_scope('Images'):
        training_summaries.append(tf.summary.image('Input Image', style_network_input, max_outputs=1))
        training_summaries.append(tf.summary.image('Output Image', style_network_output, max_outputs=1))
    with tf.name_scope('Losses'):
        training_summaries.append(tf.summary.scalar('Total Loss', loss))
        training_summaries.append(tf.summary.scalar('Style Loss', style_loss))
        training_summaries.append(tf.summary.scalar('Content Loss', content_loss))
        training_summaries.append(tf.summary.scalar('Total Variation Loss', total_var_loss))
    merged_summaries = tf.summary.merge(training_summaries)
    model_name = time.strftime(model_prefix + '_%Y-%m-%d-%H-%M')
    summary_output_dir = os.path.join(constants.TENSORBOARD_DIR, model_name)
    writer = tf.summary.FileWriter(summary_output_dir)

    target_dir = os.path.join(constants.MODELS_DIR, model_name)
    os.mkdir(target_dir)
    style_network_configs['style_config'] = style_config.__repr__()
    with open(os.path.join(target_dir, 'config.json'), 'w+') as fl:
        json.dump(style_network_configs, fl, indent=4)

    with tf.keras.backend.get_session() as session:
        session.run(tf.variables_initializer(optimizer.variables() + style_network.trainable_variables))
        writer.add_graph(session.graph)
        saver = tf.train.Saver(save_relative_paths=True)

        style_targets_sample = session.run(style_targets, feed_dict={vgg_network_input: style_image})

        train_start_time = time.time()

        for epoch in range(EPOCHS):
            dataset_manager.shuffle_loaded_images()
            images = dataset_manager.get_images()
            print('Epoch ' + str(epoch + 1) + ' of ' + str(EPOCHS))
            num_training_steps = int(len(images) / BATCH_SIZE) + 1

            for step in range(num_training_steps):
                start = min(len(images), step * BATCH_SIZE)
                end = min(len(images), (step + 1) * BATCH_SIZE)
                batch = np.array(images[start:end])
                if start == end:
                    continue

                content_targets_sample = session.run(content_targets, feed_dict={vgg_network_input: batch})

                optimizer_dict = {style_network_input: batch,
                                  content_target_placeholder[0]: content_targets_sample[0]}
                for x in range(len(style_targets_sample)):
                    optimizer_dict[style_target_placeholder[x]] = style_targets_sample[x]

                run_summaries = step % int(num_training_steps/10) == 0
                ops = [optimizer_op]
                if run_summaries:
                    ops.append(merged_summaries)
                results = session.run(ops, feed_dict=optimizer_dict)

                if run_summaries:
                    writer.add_summary(results[1], (epoch*num_training_steps) + step)
                    writer.flush()

            sampled_image = session.run(style_network_output, feed_dict={style_network_input: test_content_image})
            save_image(sampled_image, os.path.join(target_dir, str(epoch) + '_style_transfer_sample_1.jpg'))
            epoch_end = time.time()
            elapsed = epoch_end - train_start_time
            time_digits = 6
            ETA = ((epoch_end - train_start_time) / max(1, (epoch+1))) * (EPOCHS - (epoch+1))
            print('elapsed: ' + str(elapsed/60.)[:time_digits] + ' min | remaining training time: ' + str(ETA/60.)[:time_digits] + ' min')
        print('Training concluded. Saving model...')



        saver.save(session, os.path.join(constants.MODELS_DIR, model_name, 'saved_' + model_name), global_step=0)
        print('Model saved.')


def normal_style_transfer(style_type):
    style_config = StyleTransfer.STYLE_CONFIG_DICT[style_type]

    style_image_1 = load_image(style_config.style_path)
    content_image_1 = load_image(CONTENT_IMAGE_1)

    var_init = content_image_1

    with tf.name_scope('vgg_network'):
        vgg_network = build_vgg_network()

    vgg_network_input = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='vgg_network_input')

    [style_targets, content_targets] = vgg_network(vgg_preprocess(vgg_network_input))

    train_target = tf.Variable(var_init, name='train_target')

    [style_layers, content_layers] = vgg_network(vgg_preprocess(train_target))

    style_target_placeholder = [tf.placeholder(tf.float32, shape=[None, None, None], name='style_target_placeholder_' + str(x)) for x in range(len(style_layers))]
    content_target_placeholder = [tf.placeholder(tf.float32, shape=[None, None, None, None], name='content_target_placeholder')]

    loss, style_loss, content_loss, total_var_loss = StyleTransfer.total_loss(train_target,
                                                                              style_target_placeholder, content_target_placeholder,
                                                                              style_layers, content_layers,
                                                                              style_weight=style_config.style_weight,
                                                                              content_weight=style_config.content_weight,
                                                                              total_variation_weight=style_config.total_variation_weight)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.02, beta1=.99, epsilon=1e-1)
    optimizer_op = optimizer.minimize(loss, var_list=[train_target])



    clipped_image = StyleTransfer.clip_0_1(train_target)

    training_summaries = []
    with tf.name_scope('Images'):
        training_summaries.append(tf.summary.image('Output Image', train_target, max_outputs=1))
    with tf.name_scope('Losses'):
        training_summaries.append(tf.summary.scalar('Total Loss', loss))
        training_summaries.append(tf.summary.scalar('Style Loss', style_loss))
        training_summaries.append(tf.summary.scalar('Content Loss', content_loss))
        training_summaries.append(tf.summary.scalar('Total Variation Loss', total_var_loss))
    merged_summaries = tf.summary.merge(training_summaries)
    model_name = time.strftime("STYLE_TRANSFER_%Y-%m-%d-%H-%M")
    summary_output_dir = os.path.join(constants.TENSORBOARD_DIR, model_name)
    writer = tf.summary.FileWriter(summary_output_dir)

    with tf.keras.backend.get_session() as session:
        session.run(tf.variables_initializer(optimizer.variables() + [train_target]))
        writer.add_graph(session.graph)
        saver = tf.train.Saver()

        style_targets_sample = session.run(style_targets, feed_dict={vgg_network_input: style_image_1})
        content_targets_sample = session.run(content_targets, feed_dict={vgg_network_input: content_image_1})

        train_start_time = time.time()

        for epoch in range(EPOCHS):
            print('Epoch ' + str(epoch + 1) + ' of ' + str(EPOCHS))
            num_training_steps = 100

            for step in range(num_training_steps):

                optimizer_dict = {content_target_placeholder[0]: content_targets_sample[0]}
                for x in range(len(style_targets_sample)):
                    optimizer_dict[style_target_placeholder[x]] = style_targets_sample[x]

                run_summaries = step % int(num_training_steps / 10) == 0
                ops = [optimizer_op]
                if run_summaries:
                    ops.append(merged_summaries)
                results = session.run(ops, feed_dict=optimizer_dict)
                clipped = session.run(clipped_image)
                train_target.assign(clipped)

                if run_summaries:
                    writer.add_summary(results[1], (epoch * num_training_steps) + step)
                    writer.flush()

            sampled_image = session.run(train_target, feed_dict={})
            # print(sampled_image)
            save_image(sampled_image, os.path.join(constants.STYLIZED_IMAGES_DIR, str(epoch) + '_style_transfer_sample_1.jpg'))
            epoch_end = time.time()
            elapsed = epoch_end - train_start_time
            time_digits = 6
            ETA = ((epoch_end - train_start_time) / max(1, (epoch + 1))) * (EPOCHS - (epoch + 1))
            print('elapsed: ' + str(elapsed / 60.)[:time_digits] + ' min | remaining training time: ' + str(ETA / 60.)[:time_digits] + ' min')
        print('Training concluded. Saving model...')
        os.mkdir(os.path.join(constants.MODELS_DIR, model_name))
        saver.save(session, os.path.join(constants.MODELS_DIR, model_name, 'saved_' + model_name), global_step=0)
        print('Model saved.')

class StyleNetService(threading.Thread):
    def __init__(self, model_name):
        # tf.logging.set_verbosity(tf.logging.ERROR)
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        super().__init__()
        self.model_name = model_name
        self.loaded = False

    def run(self):
        print('StyleNetService starting...')
        model_dir = os.path.join(constants.MODELS_DIR, self.model_name)
        meta_name = 'saved_' + self.model_name + '-0.meta'
        self.session = tf.Session()
        # saver = tf.train.import_meta_graph(os.path.join(model_dir, meta_name))
        saver = tf.compat.v1.train.import_meta_graph(os.path.join(model_dir, meta_name))
        saver.restore(self.session, tf.train.latest_checkpoint(model_dir))


        self.network_input = self.session.graph.get_tensor_by_name('style_network/input_1:0')
        self.network_output = self.session.graph.get_tensor_by_name('style_network/activation_20/Sigmoid:0')

        print('StyleNetService running.')
        self.loaded = True

    def run_on_image(self, input_image):
        output_image = self.session.run(self.network_output, feed_dict={self.network_input: input_image})
        return output_image

    def close(self):

        self.session.close()

    def wait_for_ready(self, increment = .1):
        while not self.loaded:
            time.sleep(increment)


def run_on_image(source_image_path, destination_path, model_path=None):
    target_image = load_image(source_image_path)

    real_model_path = model_path
    if real_model_path == None:
        real_model_path = model_utilities.get_most_recent_model_name(constants.MODELS_DIR, model_prefix)

    service = StyleNetService(real_model_path)
    service.start()
    service.wait_for_ready()

    output_image = service.run_on_image(target_image)

    save_image(output_image, destination_path)


if __name__=='__main__':
    if len(sys.argv) == 3:
        src = sys.argv[1]
        dest = sys.argv[2]
        if os.path.exists(src):
            run_on_image(src, dest)
    if len(sys.argv) == 4:
        model = sys.argv[1]
        src = sys.argv[2]
        dest = sys.argv[3]
        model_dir = os.path.join(constants.MODELS_DIR, model)
        if os.path.exists(src) and os.path.exists(model_dir) and os.path.isdir(model_dir):
            run_on_image(src, dest, model_path=model)
    else:
        train_style_network('starry_night_5')
        # normal_style_transfer('starry_night')
        # normal_style_transfer('heiro_alt')
        # run_on_image(CONTENT_IMAGE_2_LARGE, os.path.join(constants.TEST_DIR, 'stylized_test.jpg'))

