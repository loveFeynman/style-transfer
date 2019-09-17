import os
import time

import tensorflow as tf
import numpy as np

import constants
import model_utilities
from image_utilities import load_image, CocoDatasetManager, save_image
from model_utilities import LayerConfig, StyleTransfer

LEARNING_RATE = 1e-3
EPOCHS = 160
BATCH_SIZE = 4
TOTAL_IMAGES = 1000

STYLE_IMAGE_1 = '../res/styles/starry_night_small.jpg'
CONTENT_IMAGE_1 = '../res/content/antelope_small.jpg'
CONTENT_IMAGE_2 = '../res/content/sparrow_small.jpg'


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

    # conv_layer_configs = [LayerConfig(32, 9, 1),
    #                       LayerConfig(64, 3, 1),
    #                       LayerConfig(128, 3, 1)]
    #
    # residual_block_configs = [LayerConfig(128, 3, 1),
    #                           LayerConfig(128, 3, 1),
    #                           LayerConfig(128, 3, 1),
    #                           LayerConfig(128, 3, 1),
    #                           LayerConfig(128, 3, 1)]
    #
    # deconv_layer_configs = [LayerConfig(64, 3, 1),
    #                         LayerConfig(32, 3, 1),
    #                         LayerConfig(3, 9, 1, activation=tf.nn.sigmoid)]


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


def vgg_preprocess(network_input):
    if isinstance(network_input, np.ndarray):
        return tf.keras.applications.vgg19.preprocess_input(network_input* 225.)
    else:
        return tf.keras.applications.vgg19.preprocess_input(tf.multiply(network_input, 225.))

def train_style_network():
    style_image_1 = load_image(STYLE_IMAGE_1)
    content_image_1 = load_image(CONTENT_IMAGE_1)
    # style_image_1 = vgg_preprocess(style_image_1)
    # content_image_1 = vgg_preprocess(content_image_1)

    print('Loading images...')
    dataset_manager = CocoDatasetManager(target_dim=(224, 224), num_images=TOTAL_IMAGES)
    images = dataset_manager.get_images()
    print('Done loading images.')

    with tf.name_scope('style_network'):
        style_network = build_style_network()
    with tf.name_scope('vgg_network'):
        vgg_network = build_vgg_network()

    style_network_input = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='style_network_input')
    vgg_network_input = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='vgg_network_input')
    [style_targets, content_targets] = vgg_network(vgg_preprocess(vgg_network_input))
    # style_targets = [StyleTransfer.gram_matrix(x) for x in style_targets]
    style_network_output = style_network(style_network_input)
    style_layers, content_layers = vgg_network(vgg_preprocess(style_network_output))
    # style_layers = [StyleTransfer.gram_matrix(x) for x in style_layers]
    style_target_placeholder = [tf.placeholder(tf.float32, shape=[None, None, None], name='style_target_placeholder_' + str(x)) for x in range(len(style_layers))]
    content_target_placeholder = [tf.placeholder(tf.float32, shape=[None, None, None, None], name='content_target_placeholder')]

    loss, style_loss, content_loss, total_var_loss = StyleTransfer.total_loss(style_network_output,
                                                                              style_target_placeholder, content_target_placeholder,
                                                                              style_layers, content_layers,
                                                                              style_weight=1e-3,
                                                                              content_weight=1e4,
                                                                              total_variation_weight=1e8)
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
    model_name = time.strftime("STYLE_NET_%Y-%m-%d-%H-%M")
    summary_output_dir = os.path.join(constants.TENSORBOARD_DIR, model_name)
    writer = tf.summary.FileWriter(summary_output_dir)

    with tf.keras.backend.get_session() as session:
        session.run(tf.variables_initializer(optimizer.variables() + style_network.trainable_variables))
        writer.add_graph(session.graph)
        saver = tf.train.Saver()

        style_targets_sample = session.run(style_targets, feed_dict={vgg_network_input: style_image_1})
        print(style_targets_sample[0])

        train_start_time = time.time()

        for epoch in range(EPOCHS):
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

            sampled_image = session.run(style_network_output, feed_dict={style_network_input: content_image_1})
            # print(sampled_image)
            save_image(sampled_image, os.path.join(constants.STYLIZED_IMAGES_DIR, str(epoch) + '_style_transfer_sample_1.jpg'))
            epoch_end = time.time()
            elapsed = epoch_end - train_start_time
            time_digits = 6
            ETA = ((epoch_end - train_start_time) / max(1, (epoch+1))) * (EPOCHS - (epoch+1))
            print('elapsed: ' + str(elapsed/60.)[:time_digits] + ' min | remaining training time: ' + str(ETA/60.)[:time_digits] + ' min')
        print('Training concluded. Saving model...')
        os.mkdir(os.path.join(constants.MODELS_DIR, model_name))
        saver.save(session, os.path.join(constants.MODELS_DIR, model_name, 'saved_' + model_name), global_step=0)
        print('Model saved.')


    # print('Saving model...')
    # model_name = model_prefix + '_' + str(time.time())[:10]
    # save_keras_model(stylizer_network, MODELS_DIR, model_name)


def normal_style_transfer():

    style_image_1 = load_image(STYLE_IMAGE_1)
    content_image_1 = load_image(CONTENT_IMAGE_1)

    var_init = content_image_1 #np.random.random_sample(content_image_1.shape).astype(np.float32)

    # style_image_1 = vgg_preprocess(style_image_1)
    # content_image_1 = vgg_preprocess(content_image_1)

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
                                                                              total_variation_weight=1e8)
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
        # session.run(tf.global_variables_initializer())
        # session.run(tf.initialize_variables([optimizer.variables()]))
        session.run(tf.variables_initializer(optimizer.variables() + [train_target]))
        writer.add_graph(session.graph)
        saver = tf.train.Saver()

        style_targets_sample = session.run(style_targets, feed_dict={vgg_network_input: style_image_1})
        content_targets_sample = session.run(content_targets, feed_dict={vgg_network_input: content_image_1})

        print(style_targets_sample[0])


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


def normal_style_transfer2():

    style_image_1 = load_image(STYLE_IMAGE_1)
    content_image_1 = load_image(CONTENT_IMAGE_1)

    # var_init = np.random.random_sample(content_image_1.shape).astype(np.float32)
    var_init = content_image_1

    style_image_1 = vgg_preprocess(style_image_1)
    content_image_1 = vgg_preprocess(content_image_1)

    with tf.name_scope('vgg_network'):
        vgg_network = build_vgg_network()

    vgg_network_input = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='vgg_network_input')

    [style_targets, content_targets] = vgg_network(vgg_network_input)

    train_target = tf.Variable(var_init, name='train_target')

    [style_layers, content_layers] = vgg_network(vgg_preprocess(train_target))

    style_target_placeholder = [tf.placeholder(tf.float32, shape=[None, None, None], name='style_target_placeholder_' + str(x)) for x in range(len(style_layers))]
    content_target_placeholder = [tf.placeholder(tf.float32, shape=[None, None, None, None], name='content_target_placeholder')]

    loss, style_loss, content_loss, total_var_loss = StyleTransfer.total_loss(train_target, style_target_placeholder, content_target_placeholder, style_layers, content_layers)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.02, beta1=.99, epsilon=1e-1)
    optimizer_op = optimizer.minimize(loss, var_list=[train_target])

    optimizer_get_gradients = optimizer.compute_gradients(loss, var_list=[train_target])
    optimizer_apply_gradients = optimizer.apply_gradients(optimizer_get_gradients)


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
    model_name = time.strftime("STYLE_%Y-%m-%d-%H-%M")
    summary_output_dir = os.path.join(constants.TENSORBOARD_DIR, model_name)
    writer = tf.summary.FileWriter(summary_output_dir)

    with tf.keras.backend.get_session() as session:
        # session.run(tf.global_variables_initializer())
        # session.run(tf.initialize_variables([optimizer.variables()]))
        session.run(tf.variables_initializer(optimizer.variables() + [train_target]))
        writer.add_graph(session.graph)
        saver = tf.train.Saver()

        style_targets_sample = session.run(style_targets, feed_dict={vgg_network_input: style_image_1})
        content_targets_sample = session.run(content_targets, feed_dict={vgg_network_input: content_image_1})

        # print(style_targets_sample[0])


        train_start_time = time.time()

        for epoch in range(EPOCHS):
            print('Epoch ' + str(epoch + 1) + ' of ' + str(EPOCHS))
            num_training_steps = 100

            for step in range(num_training_steps):

                optimizer_dict = {content_target_placeholder[0]: content_targets_sample[0]}
                for x in range(len(style_targets_sample)):
                    optimizer_dict[style_target_placeholder[x]] = style_targets_sample[x]

                run_summaries = step % int(num_training_steps / 10) == 0
                # ops = [optimizer_op]
                ops = [optimizer_apply_gradients]
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




if __name__=='__main__':
    train_style_network()
    #normal_style_transfer()