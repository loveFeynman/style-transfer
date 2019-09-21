import sys
import time

import cv2
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.python._pywrap_tensorflow_internal import Flatten
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout
from tensorflow.python.keras.saving import load_model

import model_utilities
from image_utilities import flip_BR, pixel_to_decimal, save_image, CocoDatasetManager, open_image, image_3d_to_4d, load_image, PreprocessedCocoDatasetManager

# using code from (https://www.tensorflow.org/beta/tutorials/generative/style_transfer) at points

from constants import *
from model_utilities import save_keras_model, load_keras_model, get_most_recent_model_name, LayerConfig

TEST_IMG = os.path.join(TEST_DIR, 'vgg_test_1.jpg')
LOGGER_PATH = '../res/test/log.txt'

#style_image_1 = '../res/styles/honeycomb_small.jpg'
style_image_1 = os.path.join(STYLES_DIR, 'starry_night.jpg')
content_image_1 = os.path.join(CONTENT_DIR, 'antelope_small.jpg')
content_image_2 = os.path.join(CONTENT_DIR, 'sparrow_small.jpg')


output_path = STYLIZED_IMAGES_DIR

model_prefix = 'STYLIZER'

style_weight = 1e-2
content_weight = 1e4
total_variation_weight = 1e8 # 1e8

learning_rate = 0.001

steps_per_epoch = 50
epochs = 80
BATCH_SIZE = 4


tf.enable_eager_execution()

class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super().__init__()

        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        outputs = [vgg.get_layer(name).output for name in (style_layers + content_layers)]

        self.vgg = tf.keras.Model([vgg.input], outputs)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}


def style_net(): # As far as I know, this is identical to StyleNet()
    filter_counts = [16 for x in range(9)]  # [16, 16, 16, 16, 16, 16, 16, 16, 16]
    kernel_sizes = [5 for x in range(9)]  # [3, 3, 3, 3, 3, 3, 3, 3, 3]
    filter_strides = [1 for x in range(9)]  # [1, 1, 1, 1, 1, 1, 1, 1, 1]
    use_batch_norm = True
    activation = tf.nn.elu

    layer = 0
    input = tf.keras.layers.Input(shape=(224, 224, 3))

    downsize_1 = tf.keras.layers.Conv2D(filter_counts[layer], kernel_sizes[layer], filter_strides[layer], activation=activation, padding='valid')(input)
    layer += 1
    downsize_2 = tf.keras.layers.Conv2D(filter_counts[layer], kernel_sizes[layer], filter_strides[layer], activation=activation, padding='valid')(downsize_1)
    layer += 1
    if use_batch_norm:
        downsize_2 = tf.keras.layers.BatchNormalization()(downsize_2)

    flat = tf.keras.layers.Conv2D(filter_counts[layer], kernel_sizes[layer], filter_strides[layer], activation=activation, padding='same')(downsize_2)
    layer += 1
    flat = tf.keras.layers.Conv2D(filter_counts[layer], kernel_sizes[layer], filter_strides[layer], activation=activation, padding='same')(flat)
    layer += 1

    if use_batch_norm:
        flat = tf.keras.layers.BatchNormalization()(flat)

    flat = tf.keras.layers.Conv2D(filter_counts[layer], kernel_sizes[layer], filter_strides[layer], activation=activation, padding='same')(flat)
    layer += 1
    flat = tf.keras.layers.Conv2D(filter_counts[layer], kernel_sizes[layer], filter_strides[layer], activation=activation, padding='same')(flat)
    layer += 1

    if use_batch_norm:
        flat = tf.keras.layers.BatchNormalization()(flat)

    upsample_add_1 = tf.keras.layers.Add()([flat, downsize_2])
    upsample_layers = tf.keras.layers.Conv2DTranspose(filter_counts[layer], kernel_sizes[layer], filter_strides[layer], activation=activation, padding='valid')(upsample_add_1)
    layer += 1
    upsample_add_2 = tf.keras.layers.Add()([upsample_layers, downsize_1])
    upsample_layers = tf.keras.layers.Conv2DTranspose(3, kernel_sizes[layer], filter_strides[layer], activation=tf.nn.sigmoid, padding='valid')(upsample_add_2)
    layer += 1

    if use_batch_norm:
        upsample_layers = tf.keras.layers.BatchNormalization()(upsample_layers)

    model = tf.keras.Model(inputs=input, outputs=upsample_layers)

    return model

def thicc_style_net():
    '''
    set up like:
    012345678901234567890
    |||===============|||
    ||||||=========||||||
    |||||||||===|||||||||
    |||||||||||||||||||||
    |||||||||===|||||||||
    ||||||=========||||||
    |||===============|||


    '''

    '''
    224-218:1
    218-212:2
    212-206:3
    206-200:4
    
    200-200
    200-200
    200-200    
    200-200    
    
    200:4-206
    206:3-212
    212:2-218
    218:1-224
    
    
    
    
    '''

    num_down_layers = 3
    num_flat_layers = 3
    total_layers = (num_down_layers * 2) + num_flat_layers
    filter_counts = [16 for x in range(total_layers)]
    kernel_sizes = [5 for x in range(total_layers)]
    filter_strides = [1 for x in range(total_layers)]
    use_batch_norm = True
    activation = tf.nn.elu

    input = tf.keras.layers.Input(shape=(224, 224, 3))

    down_layers = []
    flat_layers = []
    up_layers = []

    def down_layer(layer_input, layer_num, layer_activation=activation):
        layer = tf.keras.layers.Conv2D(filter_counts[layer_num], kernel_sizes[layer_num], filter_strides[layer_num], activation=layer_activation, padding='valid')(layer_input)
        if use_batch_norm:
            layer = tf.keras.layers.BatchNormalization()(layer)
        return layer

    def flat_layer(layer_input, layer_num, layer_activation=activation):
        layer = tf.keras.layers.Conv2D(filter_counts[layer_num], kernel_sizes[layer_num], filter_strides[layer_num], activation=layer_activation, padding='same')(layer_input)
        if use_batch_norm:
            layer = tf.keras.layers.BatchNormalization()(layer)
        return layer

    def up_layer(layer_input, layer_num, layer_activation=activation, num_filters = None):
        filter_count = filter_counts[layer_num]
        if num_filters != None:
            filter_count = num_filters
        layer = tf.keras.layers.Conv2DTranspose(filter_count, kernel_sizes[layer_num], filter_strides[layer_num], activation=layer_activation, padding='valid')(layer_input)
        if use_batch_norm:
            layer = tf.keras.layers.BatchNormalization()(layer)
        return layer

    prev_input = input
    for i in range(num_down_layers):
        layer = down_layer(prev_input, i)
        prev_input = layer
        down_layers.append(layer)

    for i in range(num_flat_layers):
        layer = flat_layer(prev_input, i + num_down_layers)
        prev_input = layer
        flat_layers.append(layer)

    for i in range(num_down_layers - 1):
        reference_down_layer = down_layers[-(i+1)]
        layer = up_layer(prev_input + reference_down_layer, i + num_down_layers + num_flat_layers)
        prev_input = layer
        up_layers.append(layer)

    layer = up_layer(prev_input + down_layers[-num_down_layers], num_flat_layers + (num_down_layers * 2) - 2, layer_activation=tf.nn.sigmoid, num_filters=3)

    model = tf.keras.Model(inputs=input, outputs=layer)

    return model

def red_net():

    use_batch_norm = True
    activation = tf.nn.elu

    filter_count = 16
    kernel_size = 5
    kernel_strides = 1

    input = tf.keras.layers.Input(shape=(224, 224, 3))

    num_conv_layers = 5
    layers_per_segment = 2

    def conv_layer(layer_input, layer_activation=activation, num_filters=None):
        kernel_count = filter_count
        if num_filters != None:
            kernel_count = num_filters
        layer = tf.keras.layers.Conv2D(kernel_count, kernel_size, kernel_strides, activation=layer_activation, padding='same')(layer_input)
        if use_batch_norm:
            layer = tf.keras.layers.BatchNormalization()(layer)
        return layer

    def deconv_layer(layer_input, layer_activation=activation, num_filters=None):
        kernel_count = filter_count
        if num_filters != None:
            kernel_count = num_filters
        layer = tf.keras.layers.Conv2DTranspose(kernel_count, kernel_size, kernel_strides, activation=layer_activation, padding='same')(layer_input)
        if use_batch_norm:
            layer = tf.keras.layers.BatchNormalization()(layer)
        return layer

    def n_layers(layer_func, layer_input, num_layers, layer_activation=activation, num_filters=None):
        prev_layer = layer_input
        for i in range(num_layers):
            layer = layer_func(prev_layer, layer_activation=layer_activation, num_filters=num_filters)
            prev_layer = layer
        return prev_layer

    def n_conv_layers(layer_input, num_layers, layer_activation=activation, num_filters=None):
        return n_layers(conv_layer, layer_input, num_layers, layer_activation=layer_activation, num_filters=num_filters)

    def n_deconv_layers(layer_input, num_layers, layer_activation=activation, num_filters=None):
        return n_layers(deconv_layer, layer_input, num_layers, layer_activation=layer_activation, num_filters=num_filters)

    conv_layers = []
    deconv_layers = []

    prev_input = input
    for i in range(num_conv_layers):
        layer = n_conv_layers(prev_input, layers_per_segment)
        prev_input = layer
        conv_layers.append(layer)

    starting_deconv = n_deconv_layers(prev_input, 2)
    deconv_layers.append(starting_deconv)
    prev_input = starting_deconv

    for i in range(1, num_conv_layers - 1):
        reference_conv_layer = conv_layers[-(i + 1)]
        reference_deconv_layer = deconv_layers[i-1]
        layer = n_deconv_layers(tf.concat([prev_input, reference_conv_layer, reference_deconv_layer], axis=3), 2)
        prev_input = layer
        deconv_layers.append(layer)

    layer = n_deconv_layers(tf.concat([prev_input, conv_layers[0], deconv_layers[-1]], axis=3), 2, layer_activation=tf.nn.sigmoid, num_filters=3)

    model = tf.keras.Model(inputs=input, outputs=layer)

    return model

def residual_blocks():
    activation = tf.nn.elu

    filter_count = 16
    kernel_size = 5
    kernel_strides = 1

    merge_layer_type = tf.keras.layers.Add()
    # merge_layer_type = tf.keras.layers.Concatenate(axis=3)
    use_hourglass = False

    num_conv_layers = 5
    num_deconv_layers = num_conv_layers
    num_residual_blocks = 5

    def conv_layer(layer_input, layer_activation=activation, layer_stride=kernel_strides, use_batch_norm=True):
        layer = tf.keras.layers.Conv2D(filter_count, kernel_size, layer_stride, padding='same')(layer_input)
        if use_batch_norm:
            layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.Activation(layer_activation)(layer)
        return layer

    def deconv_layer(layer_input, layer_activation=activation, num_filters=filter_count, use_batch_norm=True):
        layer = tf.keras.layers.Conv2DTranspose(num_filters, kernel_size, kernel_strides, padding='same')(layer_input)
        if use_batch_norm:
            layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.Activation(layer_activation)(layer)
        return layer

    def residual_block(layer_input, layer_activation=activation):
        '''
        uses BN after add vs before
        reference http://torch.ch/blog/2016/02/04/resnets.html
        for other options
        '''
        layer = tf.keras.layers.Conv2D(filter_count, kernel_size, 1, padding='same')(layer_input)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.Activation(layer_activation)(layer)
        layer = tf.keras.layers.Conv2D(filter_count, kernel_size, 1, padding='same')(layer)
        layer = merge_layer_type([layer, layer_input])
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.Activation(layer_activation)(layer)
        return layer

    network_input = tf.keras.layers.Input(shape=(224, 224, 3))

    conv_layers = []
    deconv_layers = []

    layer = network_input
    for i in range(num_conv_layers):
        layer = conv_layer(layer)
        conv_layers.append(layer)

    for i in range(num_residual_blocks):
        layer = residual_block(layer)

    if use_hourglass:
        layer = merge_layer_type([layer, conv_layers[-1]])

    layer = deconv_layer(layer)
    deconv_layers.append(layer)

    for i in range(1, num_conv_layers - 1):
        reference_conv_layer = conv_layers[-(i + 1)]
        if use_hourglass:
            layer = merge_layer_type([layer, reference_conv_layer])
        layer = deconv_layer(layer)
        deconv_layers.append(layer)

    if use_hourglass:
        layer = merge_layer_type([layer, conv_layers[0]])
    layer = deconv_layer(layer, layer_activation=tf.nn.sigmoid, num_filters=3)

    return tf.keras.Model(inputs=network_input, outputs=layer)

def residual_blocks_2():
    '''
    32, 9x9, stride 1
    64, 3x3, s 2
    128, 3x3, s2

    residual block x 5

    64, 3x3, s2
    32, 3x3, s2
    3, 9x9, s1



    :return:
    '''
    activation = tf.nn.relu

    filter_count = 128
    kernel_size = 5
    kernel_strides = 1

    merge_layer_type = tf.keras.layers.Add()

    num_residual_blocks = 3

    def conv_layer(layer_input, layer_activation=activation, num_filters=filter_count, filter_size=kernel_size, layer_stride=kernel_strides, use_batch_norm=True):
        layer = tf.keras.layers.Conv2D(num_filters, filter_size, layer_stride, padding='same')(layer_input)
        if use_batch_norm:
            layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.Activation(layer_activation)(layer)
        return layer

    def deconv_layer(layer_input, layer_activation=activation, num_filters=filter_count, filter_size=kernel_size, layer_stride=kernel_strides, use_batch_norm=True):
        layer = tf.keras.layers.Conv2DTranspose(num_filters, filter_size, layer_stride, padding='same')(layer_input)
        if use_batch_norm:
            layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.Activation(layer_activation)(layer)
        return layer

    def residual_block(layer_input, layer_activation=activation):
        '''
        uses BN after add vs before
        reference http://torch.ch/blog/2016/02/04/resnets.html
        for other options
        '''
        layer = tf.keras.layers.Conv2D(filter_count, kernel_size, 1, padding='same')(layer_input)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.Activation(layer_activation)(layer)
        layer = tf.keras.layers.Conv2D(filter_count, kernel_size, 1, padding='same')(layer)
        layer = merge_layer_type([layer, layer_input])
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.Activation(layer_activation)(layer)
        return layer

    network_input = tf.keras.layers.Input(shape=(224, 224, 3))

    layer = network_input

    layer = conv_layer(layer, num_filters=32, filter_size=5, layer_stride=1)
    layer = conv_layer(layer, num_filters=64, filter_size=3, layer_stride=1)
    layer = conv_layer(layer, num_filters=128, filter_size=3, layer_stride=1)

    for i in range(num_residual_blocks):
        layer = residual_block(layer)

    layer = deconv_layer(layer, num_filters=64, filter_size=3, layer_stride=1)
    layer = deconv_layer(layer, num_filters=32, filter_size=3, layer_stride=1)
    layer = deconv_layer(layer, num_filters=3, filter_size=5, layer_stride=1, layer_activation=tf.nn.sigmoid)

    return tf.keras.Model(inputs=network_input, outputs=layer)

def residual_blocks_3():
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

    layer = tf.identity(layer, name='network_output')
    return tf.keras.Model(inputs=network_input, outputs=layer)

def train_stylizer():
    stylizer_network = style_net()

    style_image = load_image(style_image_1)
    content_image = load_image(content_image_1)
    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']
    extractor = StyleContentModel(style_layers, content_layers)
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    opt = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=.99, epsilon=1e-1)

    def train_step(input_image):
        with tf.GradientTape() as tape:
            stylized_image = stylizer_network(input_image)
            outputs = extractor(stylized_image)
            loss = total_loss(outputs, style_targets, content_targets, len(style_layers), len(content_layers), stylized_image)

        variables = stylizer_network.trainable_variables
        grad = tape.gradient(loss, variables)
        opt.apply_gradients(zip(grad, variables))

    for epoch in range(epochs):
        print('Epoch ' + str(epoch + 1) + ' of ' + str(epochs))
        for steps in range(steps_per_epoch):
            train_step(content_image)
        trained_img = stylizer_network(content_image).numpy()[0]
        save_image(trained_img, os.path.join(output_path, str(epoch) + '_style_transfer_sample.jpg'))

    print('Saving model...')
    model_name = model_prefix + '_' + str(time.time())[:10]
    save_keras_model(stylizer_network, MODELS_DIR, model_name)

def train_stylizer_on_dataset():
    stylizer_network = residual_blocks_3()
    style_image = load_image(style_image_1)
    content_im_1 = load_image(content_image_1)
    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']
    extractor = StyleContentModel(style_layers, content_layers)
    opt = tf.keras.optimizers.Adam(lr=learning_rate) #, beta_1=.99, epsilon=1e-1)
    style_targets = extractor(style_image)['style']

    def train_step(input_image):

        content_targets = extractor(input_image)['content']
        with tf.GradientTape() as tape:
            stylized_image = stylizer_network(input_image)
            outputs = extractor(stylized_image)
            loss = total_loss(outputs, style_targets, content_targets, len(style_layers), len(content_layers), stylized_image)

        variables = stylizer_network.trainable_variables
        grad = tape.gradient(loss, variables)
        opt.apply_gradients(zip(grad,variables))
        return loss

    print('Loading images...')
    dataset_manager = PreprocessedCocoDatasetManager(num_images = 1000)
    images = dataset_manager.get_images()
    print('Done loading images.')


    train_start_time = time.time()



    for epoch in range(epochs):
        print('Epoch ' + str(epoch + 1) + ' of ' + str(epochs))
        num_training_steps = int(len(images) / BATCH_SIZE) + 1

        for step in range(num_training_steps):
            start = min(len(images), step * BATCH_SIZE)
            end = min(len(images), (step + 1) * BATCH_SIZE)
            batch = np.array(images[start:end])
            if start == end:
                continue

            full_loss = train_step(batch)
            if step == 0:
                trained_img = stylizer_network(content_im_1).numpy()[0]
                save_image(trained_img, os.path.join(output_path, str(epoch) + '_style_transfer_sample_1.jpg'))
                epoch_end = time.time()
                elapsed = epoch_end - train_start_time
                time_digits = 6
                ETA = ((epoch_end - train_start_time) / max(1, epoch)) * (epochs - epoch)
                print('loss: ' + str(full_loss.numpy()) + ' | elapsed: ' + str(elapsed/60.)[:time_digits] + ' min | remaining training time: ' + str(ETA/60.)[:time_digits] + ' min')


    print('Saving model...')
    model_name = model_prefix + '_' + str(time.time())[:10]
    save_keras_model(stylizer_network, MODELS_DIR, model_name)

def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def high_pass_x_y(image):
    x_var = image[:,:,1:,:] - image[:,:,:-1,:]
    y_var = image[:,1:,:,:] - image[:,:-1,:,:]
    return x_var, y_var
def total_variation_loss(image):
    x_deltas, y_deltas = high_pass_x_y(image)
    return tf.reduce_mean(x_deltas**2) + tf.reduce_mean(y_deltas**2)
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result / (num_locations)
def style_content_loss(outputs, style_targets, content_targets, num_style_layers, num_content_layers):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss
def total_loss(style_network_outputs, style_targets, content_targets, num_style_layers, num_content_layers, style_network_input):
    return style_content_loss(style_network_outputs, style_targets, content_targets, num_style_layers, num_content_layers) + (total_variation_weight * total_variation_loss(style_network_input))

# def save(model, name):
#     dirname = os.path.join(STYLIZER_NETWORK_MODELS_DIR, name)
#     if not os.path.exists(dirname):
#         os.makedirs(dirname)
#     save_keras_model(model, dirname, name)
#
# def load(name):
#     dirname = os.path.join(STYLIZER_NETWORK_MODELS_DIR, name)
#     return load_keras_model(dirname, name)
#
# def save_keras_model(model, dir, name):
#     # model.save_weights(os.path.join(dir, name + '.h5'))
#     # with open(os.path.join(dir, name + '.json'), 'w+') as js:
#     #     js.write(model.to_json())
#     model.save(os.path.join(dir, name + '.h5'))
#
# def load_keras_model(dir, name):
#     model = load_model(os.path.join(dir, name + '.h5'))
#     # with open(os.path.join(dir, name + '.json')) as js:
#     #     model = model_from_json(js.read())
#     # model.load_weights(os.path.join(dir, name + '.h5'))
#     return model
#
# def get_most_recent_stylizer_name():
#     models = os.listdir(STYLIZER_NETWORK_MODELS_DIR)
#     models.sort()
#     models = [x for x in models if model_prefix in x]
#     return models[-1]

def test_model(model_name):
    model = load_keras_model(MODELS_DIR, model_name)
    content_image = load_image(content_image_2)
    output = model(content_image)
    save_image(output.numpy(), os.path.join(TEST_DIR, 'stylized_test.jpg'))

if __name__ == '__main__':
    # train_stylizer()
    train_stylizer_on_dataset()
    # test_model(get_most_recent_model_name(MODELS_DIR, model_prefix))