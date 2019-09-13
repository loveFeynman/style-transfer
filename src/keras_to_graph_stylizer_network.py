import os
import time

import tensorflow as tf
import numpy as np

import constants
import model_utilities
from image_utilities import load_image, CocoDatasetManager
from model_utilities import LayerConfig, StyleTransfer

LEARNING_RATE = 1e-3

STYLE_IMAGE_1 = '../res/styles/starry_night_small.jpg'
CONTENT_IMAGE_1 = '../res/content/antelope_small.jpg'
CONTENT_IMAGE_2 = '../res/content/sparrow_small.jpg'


def build_vgg_network():
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    style_layers = [vgg.get_layer(x).output for x in StyleTransfer.VGG_STYLE_TARGET_LAYER_NAMES]
    content_layers = [vgg.get_layer(x).output for x in StyleTransfer.VGG_CONTENT_TARGET_LAYER_NAMES]

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
    network_input = tf.keras.layers.Input(shape=(224, 224, 3))
    layer = network_input

    for x in conv_layer_configs:
        layer = model_builder.conv_block(layer, x)

    for x in residual_block_configs:
        layer = model_builder.residual_block(layer, x)

    for x in deconv_layer_configs:
        layer = model_builder.deconv_block(layer, x)


    return tf.keras.Model(inputs=network_input, outputs=layer)


def train_style_network():
    style_image_1 = load_image(STYLE_IMAGE_1)
    content_image_1 = load_image(CONTENT_IMAGE_1)

    with tf.name_scope('style_network'):
        style_network = build_style_network()
    with tf.name_scope('vgg_network'):
        vgg_network = build_vgg_network()

    style_network_input = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='style_network_input')
    vgg_network_input = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='vgg_network_input')
    style_targets, content_targets = vgg_network(vgg_network_input)
    style_targets = [StyleTransfer.gram_matrix(x) for x in style_targets]
    style_network_output = style_network(style_network_input)
    style_layers, content_layers = vgg_network(style_network_output)
    style_layers = [StyleTransfer.gram_matrix(x) for x in style_layers]
    style_target_placeholder = [tf.placeholder(tf.float32, shape=[None, None, None], name='style_target_placeholder_' + str(x)) for x in range(len(style_layers))]
    content_target_placeholder = [tf.placeholder(tf.float32, shape=[None, None, None, None], name='content_target_placeholder')]

    loss = StyleTransfer.total_loss(style_network_output, style_target_placeholder, content_target_placeholder, style_layers, content_layers)
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    optimizer_op = optimizer.minimize(loss, var_list=[style_network.trainable_variables])

    training_summaries = []
    training_summaries.append(tf.summary.image('Input Image', style_network_input, max_outputs=1))
    training_summaries.append(tf.summary.image('Output Image', style_network_output, max_outputs=1))
    training_summaries.append(tf.summary.scalar('Total Loss', loss))
    merged_summaries = tf.summary.merge(training_summaries)
    summary_output_dir = os.path.join(constants.TENSORBOARD_DIR, time.strftime("STYLE_%Y-%m-%d-%H-%M"))
    writer = tf.summary.FileWriter(summary_output_dir)



    with tf.keras.backend.get_session() as session:
        session.run(tf.global_variables_initializer())

        writer.add_graph(session.graph)

        style_targets_sample = session.run(style_targets, feed_dict={vgg_network_input: style_image_1})

        content_targets_sample = session.run(content_targets, feed_dict={vgg_network_input: content_image_1})

        optimizer_dict = {style_network_input: content_image_1,
                          content_target_placeholder[0]: content_targets_sample[0]}
        for x in range(len(style_targets_sample)):
            optimizer_dict[style_target_placeholder[x]] = style_targets_sample[x]
        style_network_sample, summary, _ = session.run([style_network_output, merged_summaries, optimizer_op], feed_dict=optimizer_dict)
        writer.add_summary(summary, 0)
        writer.flush()

    # print('Loading images...')
    # dataset_manager = CocoDatasetManager(target_dim=(224,224), num_images = 1000)
    # images = dataset_manager.get_images()
    # print('Done loading images.')
    #
    #
    # train_start_time = time.time()
    #
    #
    #
    # for epoch in range(epochs):
    #     print('Epoch ' + str(epoch + 1) + ' of ' + str(epochs))
    #     num_training_steps = int(len(images) / BATCH_SIZE) + 1
    #
    #     for step in range(num_training_steps):
    #         start = min(len(images), step * BATCH_SIZE)
    #         end = min(len(images), (step + 1) * BATCH_SIZE)
    #         batch = np.array(images[start:end])
    #         if start == end:
    #             continue
    #
    #         full_loss = train_step(batch)
    #         if step == 0:
    #             trained_img = stylizer_network(content_im_1).numpy()[0]
    #             save_image(trained_img, os.path.join(output_path, str(epoch) + '_style_transfer_sample_1.jpg'))
    #             epoch_end = time.time()
    #             elapsed = epoch_end - train_start_time
    #             time_digits = 6
    #             ETA = ((epoch_end - train_start_time) / max(1, epoch)) * (epochs - epoch)
    #             print('loss: ' + str(full_loss.numpy()) + ' | elapsed: ' + str(elapsed/60.)[:time_digits] + ' min | remaining training time: ' + str(ETA/60.)[:time_digits] + ' min')
    #
    #
    # print('Saving model...')
    # model_name = model_prefix + '_' + str(time.time())[:10]
    # save_keras_model(stylizer_network, MODELS_DIR, model_name)


if __name__=='__main__':
    train_style_network()