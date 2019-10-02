import json
import os
import sys

import scipy.io

HERE = os.path.dirname(os.path.abspath(__file__))

import threading
import time

import tensorflow as tf
import numpy as np

import constants
import model_utilities
from image_utilities import load_image, CocoDatasetManager, save_image, PreprocessedCocoDatasetManager
from model_utilities import LayerConfig, StyleTransfer

LEARNING_RATE = 1e-3
EPOCHS = 80 #160
BATCH_SIZE = 4
TOTAL_IMAGES = 1000 #2000

STYLE_IMAGE_1 = os.path.join(constants.STYLES_DIR, 'starry_night_small.jpg')
STYLE_IMAGE_1_LARGE = os.path.join(constants.STYLES_DIR, 'starry_night.jpg')
STYLE_IMAGE_2_LARGE = os.path.join(constants.STYLES_DIR, 'honeycomb_large.jpg')
STYLE_IMAGE_2_SQUEEZE = os.path.join(constants.STYLES_DIR, 'honeycomb_squeeze.jpg')

CONTENT_IMAGE_1 = os.path.join(constants.CONTENT_DIR, 'antelope_small.jpg')
CONTENT_IMAGE_1_2 = os.path.join(constants.CONTENT_DIR, 'antelope_256.jpg')
CONTENT_IMAGE_2 = os.path.join(constants.CONTENT_DIR, 'sparrow_small.jpg')
CONTENT_IMAGE_2_LARGE = os.path.join(constants.CONTENT_DIR, 'sparrow.jpg')
CONTENT_IMAGE_3 = os.path.join(constants.CONTENT_DIR, 'museum.jpg')

model_prefix = 'STYLE_NET'

TRAIN_IMG_DIMS = [None, 256, 256, 3]

vgg_preprocessing_mean = np.array([0, 0, 0])

def vgg_preprocess_input(network_input):
    normalized_value = network_input
    # if (tf.math.reduce_min(normalized_value) > 0. and tf.math.reduce_max(normalized_value) < 1.):
    normalized_value *= 255.
    normalized_value /= 2.
    normalized_value -= vgg_preprocessing_mean
    return normalized_value

def build_vgg_network(network_input):
    vgg_path = os.path.join(HERE, '../../../res/vgg19/imagenet-vgg-verydeep-19.mat')
    print('Loading VGG19 data...')
    vgg_data = scipy.io.loadmat(vgg_path)
    print('VGG19 data loaded.')

    vgg_normalization = vgg_data['normalization'][0][0][0]
    global vgg_preprocessing_mean
    vgg_preprocessing_mean = np.mean(vgg_normalization, axis=(0, 1))
    vgg_layer_constants = vgg_data['layers'][0]

    vgg_layer_names = [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4'
    ]

    vgg_network = {}
    working_layer = network_input

    for layer, layer_name in enumerate(vgg_layer_names):
        if 'conv' in layer_name:
            kernels, bias = vgg_layer_constants[layer][0][0][0][0]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            working_layer = model_utilities.GraphModelBuilder.conv_layer_from_weights(working_layer, kernels, bias)
        elif 'relu' in layer_name:
            working_layer = tf.nn.relu(working_layer)
        elif 'pool' in layer_name:
            working_layer = model_utilities.GraphModelBuilder.pool_layer(working_layer)
        vgg_network[layer_name] = working_layer

    content_layers = [vgg_network[x] for x in StyleTransfer.VGG_CONTENT_TARGET_LAYER_NAMES_ALT_ACT]
    style_layers = [vgg_network[x] for x in StyleTransfer.VGG_STYLE_TARGET_LAYER_NAMES_ALT_ACT]
    style_layers = [StyleTransfer.gram_matrix(x) for x in style_layers]

    return style_layers, content_layers

def build_vgg_network_replicate(network_inputs, replications=1):
    vgg_path = os.path.join(HERE, '../../../res/vgg19/imagenet-vgg-verydeep-19.mat')
    print('Loading VGG19 data...')
    vgg_data = scipy.io.loadmat(vgg_path)
    print('VGG19 data loaded.')

    vgg_normalization = vgg_data['normalization'][0][0][0]
    global vgg_preprocessing_mean
    vgg_preprocessing_mean = np.mean(vgg_normalization, axis=(0, 1))
    vgg_layer_constants = vgg_data['layers'][0]

    vgg_layer_names = [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2'
        # , 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4' # not used
    ]


    vgg_networks = [{} for x in range(replications)]
    working_layer = [network_inputs[x] for x in range(replications)]

    for layer, layer_name in enumerate(vgg_layer_names):
        if 'conv' in layer_name:
            kernels, bias = vgg_layer_constants[layer][0][0][0][0]
            # print(kernels.shape)
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            working_layer = [model_utilities.GraphModelBuilder.conv_layer_from_weights(working_layer[x], kernels, bias) for x in range(replications)]
        elif 'relu' in layer_name:
            working_layer = [tf.nn.relu(working_layer[x]) for x in range(replications)]
        elif 'pool' in layer_name:
            working_layer = [model_utilities.GraphModelBuilder.pool_layer(working_layer[x]) for x in range(replications)]
        for x in range(replications):
            vgg_networks[x][layer_name] = working_layer[x]

    content_layers = [[vgg_networks[y][x] for x in StyleTransfer.VGG_CONTENT_TARGET_LAYER_NAMES_ALT] for y in range(replications)]
    style_layers =   [[vgg_networks[y][x] for x in StyleTransfer.VGG_STYLE_TARGET_LAYER_NAMES_ALT] for y in range(replications)]
    style_layers = [[StyleTransfer.gram_matrix(x) for x in style_layers[y]] for y in range(replications)]

    return style_layers, content_layers

def build_style_network(network_input):
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
                            ]
    last_layer = LayerConfig(3, 9, 1, activation=tf.nn.sigmoid)


    model_builder = model_utilities.NNGraphModelBuilder
    layer = network_input

    for x in conv_layer_configs:
        layer = model_builder.conv_block(layer, x)

    for x in residual_block_configs:
        layer = model_builder.residual_block(layer, x)

    for x in deconv_layer_configs:
        layer = model_builder.deconv_block(layer, x)

    layer = model_builder.conv_block(layer, last_layer)

    def stringify_config_list(l):
        return [x.__repr__() for x in l]

    layer_configs = {
        'conv_layers': stringify_config_list(conv_layer_configs),
        'residual_layers': stringify_config_list(residual_block_configs),
        'deconv_layers': stringify_config_list(deconv_layer_configs)
    }

    return layer, layer_configs


def train_style_network(style_type, test_content_image_path=CONTENT_IMAGE_1_2):
    style_config = StyleTransfer.STYLE_CONFIG_DICT[style_type]
    style_image = load_image(style_config.style_path)
    test_content_image = load_image(test_content_image_path)

    ####################
    ### PLACEHOLDERS ###
    ####################

    style_network_input = tf.placeholder(tf.float32, shape=TRAIN_IMG_DIMS, name='style_network_input')
    vgg_network_input = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='vgg_network_input')
    style_target_placeholder = [tf.placeholder(tf.float32, shape=[None, None, None], name='style_target_placeholder_' + str(x)) for x in range(len(StyleTransfer.VGG_STYLE_TARGET_LAYER_NAMES_ALT))]
    content_target_placeholder = [tf.placeholder(tf.float32, shape=[None, None, None, None], name='content_target_placeholder')]

    ################
    ### NETWORKS ###
    ################

    print('Building networks...')
    with tf.variable_scope('style_network'):
        style_network_output, style_network_configs = build_style_network(style_network_input)
    with tf.variable_scope('vgg_network'):
        [[style_targets, style_layers], [content_targets, content_layers]] = build_vgg_network_replicate([vgg_preprocess_input(vgg_network_input),
                                                                                                          vgg_preprocess_input(style_network_output)], replications=2)
    print('Done building networks.')

    print('Loading images...')
    dataset_manager = PreprocessedCocoDatasetManager(target_dim=(256, 256), num_images=TOTAL_IMAGES)
    print('Done loading images.')

    ############################
    ### LOSS AND OPTIMIZIZER ###
    ############################

    loss, style_loss, content_loss, total_var_loss = StyleTransfer.total_loss(style_network_output,
                                                                              style_target_placeholder, content_target_placeholder,
                                                                              style_layers, content_layers,
                                                                              style_weight=style_config.style_weight,
                                                                              content_weight=style_config.content_weight,
                                                                              total_variation_weight=style_config.total_variation_weight)
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE) #learning_rate=0.02, beta1=.99, epsilon=1e-1)
    vars_to_train = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='style_network')
    # print(vars_to_train)
    # print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    optimizer_op = optimizer.minimize(loss, var_list=vars_to_train)


    all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    vars_to_init = []
    for var in all_vars:
        if 'vgg_network' in var.name:
            continue
        else:
            vars_to_init.append(var)

    ###################
    ### TENSORBOARD ###
    ###################

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
    model_name = time.strftime(model_prefix + '_%Y-%m-%d-%H-%M') + '_' + style_type
    summary_output_dir = os.path.join(constants.TENSORBOARD_DIR, model_name)
    writer = tf.summary.FileWriter(summary_output_dir)

    ##############
    ### CONFIG ###
    ##############

    target_dir = os.path.join(constants.MODELS_DIR, model_name)
    os.mkdir(target_dir)
    style_network_configs['style_config'] = style_config.__repr__()
    style_network_configs['style_layers'] = StyleTransfer.VGG_STYLE_TARGET_LAYER_NAMES
    style_network_configs['content_layers'] = StyleTransfer.VGG_CONTENT_TARGET_LAYER_NAMES
    style_network_configs['batch_size'] = BATCH_SIZE
    style_network_configs['data_size'] = TOTAL_IMAGES
    style_network_configs['epochs'] = EPOCHS
    style_network_configs['LR'] = LEARNING_RATE
    style_network_configs['config_name'] = style_type
    with open(os.path.join(target_dir, 'config.json'), 'w+') as fl:
        json.dump(style_network_configs, fl, indent=4)

    ###############
    ### SESSION ###
    ###############

    with tf.compat.v1.Session() as session:
        # session.run(tf.compat.v1.global_variables_initializer())
        session.run(tf.compat.v1.variables_initializer(vars_to_init))
        writer.add_graph(session.graph)
        saver = tf.compat.v1.train.Saver(save_relative_paths=True)

        style_targets_sample = session.run(style_targets, feed_dict={vgg_network_input: style_image})

        train_start_time = time.time()

        # sampled_image = session.run(style_network_output, feed_dict={style_network_input: test_content_image})
        # save_image(sampled_image, os.path.join(target_dir, 'init_style_transfer_sample_1.jpg'))

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

                run_summaries = step % max(int(num_training_steps/10),1) == 0
                ops = [optimizer_op]
                if run_summaries:
                    ops.append(merged_summaries)
                results = session.run(ops, feed_dict=optimizer_dict)

                if run_summaries:
                    writer.add_summary(results[1], (epoch*num_training_steps) + step)
                    writer.flush()

            sampled_image = session.run(style_network_output, feed_dict={style_network_input: test_content_image})
            save_image(sampled_image, os.path.join(target_dir, str(epoch) + '_style_transfer_sample.jpg'))
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

    gpu_config = tf.ConfigProto()
    # gpu_config.gpu_options.allow_growth = True
    # gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.7

    ####################
    ### PLACEHOLDERS ###
    ####################

    vgg_network_input = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='vgg_network_input')
    style_target_placeholder = [tf.placeholder(tf.float32, shape=[None, None, None], name='style_target_placeholder_' + str(x)) for x in range(len(StyleTransfer.VGG_STYLE_TARGET_LAYER_NAMES_ALT))]
    content_target_placeholder = [tf.placeholder(tf.float32, shape=[None, None, None, None], name='content_target_placeholder')]

    ###########################
    ### OPTIMIZATION TARGET ###
    ###########################

    train_target = tf.Variable(var_init, name='train_target')
    clipped_image = StyleTransfer.clip_0_1(train_target)

    ###################
    ### VGG NETWORK ###
    ###################

    with tf.name_scope('vgg_network'):
        [style_targets, style_layers], [content_targets, content_layers] = build_vgg_network_replicate([vgg_preprocess_input(vgg_network_input),
                                                                         vgg_preprocess_input(train_target)], replications=2)

    ############################
    ### LOSS AND OPTIMIZIZER ###
    ############################

    loss, style_loss, content_loss, total_var_loss = StyleTransfer.total_loss(train_target,
                                                                              style_target_placeholder, content_target_placeholder,
                                                                              style_layers, content_layers,
                                                                              style_weight=style_config.style_weight,
                                                                              content_weight=style_config.content_weight,
                                                                              total_variation_weight=style_config.total_variation_weight)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.02, beta1=.99, epsilon=1e-1)
    optimizer_op = optimizer.minimize(loss, var_list=[train_target])

    ###################
    ### TENSORBOARD ###
    ###################

    training_summaries = []
    with tf.name_scope('Images'):
        training_summaries.append(tf.compat.v1.summary.image('Output Image', train_target, max_outputs=1))
    with tf.name_scope('Losses'):
        training_summaries.append(tf.compat.v1.summary.scalar('Total Loss', loss))
        training_summaries.append(tf.compat.v1.summary.scalar('Style Loss', style_loss))
        training_summaries.append(tf.compat.v1.summary.scalar('Content Loss', content_loss))
        training_summaries.append(tf.compat.v1.summary.scalar('Total Variation Loss', total_var_loss))
    merged_summaries = tf.compat.v1.summary.merge(training_summaries)
    model_name = time.strftime("STYLE_TRANSFER_%Y-%m-%d-%H-%M")
    summary_output_dir = os.path.join(constants.TENSORBOARD_DIR, model_name)
    writer = tf.summary.FileWriter(summary_output_dir)

    ###############
    ### SESSION ###
    ###############

    with tf.compat.v1.Session(config=gpu_config) as session:
        # session.run(tf.variables_initializer(optimizer.variables() + [train_target]))
        session.run(tf.compat.v1.global_variables_initializer())
        writer.add_graph(session.graph)
        saver = tf.compat.v1.train.Saver()

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
    def __init__(self, model_name, model_dir = constants.MODELS_DIR):
        # tf.logging.set_verbosity(tf.logging.ERROR)
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        super().__init__()
        self.model_name = model_name
        self.model_dir = model_dir
        self.loaded = False

    def run(self):
        print('StyleNetService starting...')
        model_dir = os.path.join(self.model_dir, self.model_name)
        self.session = tf.Session()

        self.network_input = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='style_network_input')
        with tf.variable_scope('style_network'):
            self.network_output, _ = build_style_network(self.network_input)
        saver = tf.train.Saver()
        saver.restore(self.session, tf.train.latest_checkpoint(model_dir))

        # vars = [x.name for x in self.session.graph.as_graph_def().node if 'igmoid' in x.name]
        # print(vars)

        self.network_output = self.session.graph.get_tensor_by_name('style_network/Sigmoid:0')

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


def run_on_image(source_image_path, destination_path, model_path=None, model_dir=constants.MODELS_DIR):
    target_image = load_image(source_image_path)

    real_model_path = model_path
    if real_model_path == None:
        real_model_path = model_utilities.get_most_recent_model_name(model_dir, model_prefix)

    service = StyleNetService(real_model_path, model_dir)
    service.start()
    service.wait_for_ready()

    output_image = service.run_on_image(target_image)

    save_image(output_image, destination_path)

    service.close()

if __name__=='__main__':
    if len(sys.argv) == 3:
        src = sys.argv[1]
        dest = sys.argv[2]
        if os.path.exists(src):
            run_on_image(src, dest)
    elif len(sys.argv) == 4:
        model = sys.argv[1]
        src = sys.argv[2]
        dest = sys.argv[3]
        model_dir = os.path.join(constants.MODELS_DIR, model)
        if os.path.exists(src) and os.path.exists(model_dir) and os.path.isdir(model_dir):
            run_on_image(src, dest, model_path=model)
    elif len(sys.argv) == 5:
        model = sys.argv[1]
        model_base_dir = sys.argv[2]
        src = sys.argv[3]
        dest = sys.argv[4]
        model_dir = os.path.join(model_base_dir, model)
        print('--' + model_dir)

        if os.path.exists(src) and os.path.exists(model_dir) and os.path.isdir(model_dir):
            run_on_image(src, dest, model_path=model_dir, model_dir=model_base_dir)
    else:
        # normal_style_transfer('starry_night_transfer')

        train_style_network('heiro_4')



