import cv2
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.python._pywrap_tensorflow_internal import Flatten
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout

from image_utilities import flip_BR, pixel_to_decimal, save_image, CocoDatasetManager

# using code from (https://www.tensorflow.org/beta/tutorials/generative/style_transfer) at points

from constants import *

TEST_IMG = os.path.join(TEST_DIR, 'vgg_test_1.jpg')
LOGGER_PATH = '../res/test/log.txt'

style_image_1 = '../res/styles/honeycomb_small.jpg'
# style_image_1 = '../res/styles/starry_night_small.jpg'
content_image_1 = '../res/content/antelope_small.jpg'
output_path = STYLIZED_IMAGES_DIR



style_weight = 1e-2
content_weight = 1e4
total_variation_weight = 1e8

learning_rate = 0.001

steps_per_epoch = 50
epochs = 500

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

class StyleNet(tf.keras.models.Model):
    def __init__(self):
        super().__init__()

        filter_counts = [32 for x in range(9)] #[16, 16, 16, 16, 16, 16, 16, 16, 16]
        kernel_sizes = [5 for x in range(9)] #[3, 3, 3, 3, 3, 3, 3, 3, 3]
        filter_strides = [1 for x in range(9)] #[1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.use_batch_norm = True
        activation = tf.nn.elu
        layer = 0

        self.downsize_1 = tf.keras.layers.Conv2D(filter_counts[layer], kernel_sizes[layer], filter_strides[layer], activation=activation, padding='valid')
        layer += 1
        self.downsize_2 = tf.keras.layers.Conv2D(filter_counts[layer], kernel_sizes[layer], filter_strides[layer], activation=activation, padding='valid')
        layer += 1
        self.downsize_2_bn = tf.keras.layers.BatchNormalization()

        self.flat_1 = tf.keras.layers.Conv2D(filter_counts[layer], kernel_sizes[layer], filter_strides[layer], activation=activation, padding='same')
        layer += 1
        self.flat_2 = tf.keras.layers.Conv2D(filter_counts[layer], kernel_sizes[layer], filter_strides[layer], activation=activation, padding='same')
        layer += 1
        self.flat_2_bn = tf.keras.layers.BatchNormalization()

        self.flat_3 = tf.keras.layers.Conv2D(filter_counts[layer], kernel_sizes[layer], filter_strides[layer], activation=activation, padding='same')
        layer += 1
        self.flat_4 = tf.keras.layers.Conv2D(filter_counts[layer], kernel_sizes[layer], filter_strides[layer], activation=activation, padding='same')
        layer += 1
        self.flat_4_bn = tf.keras.layers.BatchNormalization()

        self.upsample_layers_1 = tf.keras.layers.Conv2DTranspose(filter_counts[layer], kernel_sizes[layer], filter_strides[layer], activation=activation, padding='valid')
        layer += 1
        self.upsample_layers_2 = tf.keras.layers.Conv2DTranspose(3, kernel_sizes[layer], filter_strides[layer], activation=activation, padding='valid')
        layer += 1
        self.upsample_layers_2_bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        value_1 = self.downsize_1(inputs)
        value_2 = self.downsize_2(value_1)
        if self.use_batch_norm:
            value_2 = self.downsize_2_bn(value_2)
        value = self.flat_1(value_2)
        value = self.flat_2(value)
        if self.use_batch_norm:
            value = self.flat_2_bn(value)
        value = self.flat_3(value)
        value = self.flat_4(value)
        if self.use_batch_norm:
            value = self.flat_4_bn(value)
        value = self.upsample_layers_1(value + value_2)
        value = self.upsample_layers_2(value + value_1)
        if self.use_batch_norm:
            value = self.upsample_layers_2_bn(value)
        return value

def style_net(): # As far as I know, this is identical to StyleNet()
    filter_counts = [16, 16, 16, 16, 16, 16, 16, 16, 16]
    kernel_sizes = [3, 3, 3, 3, 3, 3, 3, 3, 3]
    filter_strides = [1, 1, 1, 1, 1, 1, 1, 1, 1]
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

    upsample_layers = tf.keras.layers.Conv2DTranspose(filter_counts[layer], kernel_sizes[layer], filter_strides[layer], activation=activation, padding='valid')(flat + downsize_2)
    layer += 1
    upsample_layers = tf.keras.layers.Conv2DTranspose(3, kernel_sizes[layer], filter_strides[layer], activation=activation, padding='valid')(upsample_layers + downsize_1)
    layer += 1

    if use_batch_norm:
        upsample_layers = tf.keras.layers.BatchNormalization()(upsample_layers)

    model = tf.keras.Model(inputs=input, outputs=upsample_layers)

    return model

def train_stylizer():
    stylizer_network = style_net()
    #stylizer_network = StyleNet() # WORKS EQUALLY AS WELL AS PREVIOUS LINE(?)
    style_image = pixel_to_decimal(flip_BR(cv2.imread(style_image_1))).reshape((1, 224, 224, 3))
    content_image = pixel_to_decimal(flip_BR(cv2.imread(content_image_1))).reshape((1, 224, 224, 3))
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
        save_image(trained_img, os.path.join(output_path, 'style_transfer_sample_' + str(epoch) + '.jpg'))

def train_stylizer_on_dataset():
    stylizer_network = style_net()
    style_image = pixel_to_decimal(flip_BR(cv2.imread(style_image_1))).reshape((1, 224, 224, 3))
    content_image = pixel_to_decimal(flip_BR(cv2.imread(content_image_1))).reshape((1, 224, 224, 3))
    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']
    extractor = StyleContentModel(style_layers, content_layers)
    opt = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=.99, epsilon=1e-1)

    def train_step(input_image):
        style_targets = extractor(style_image)['style']
        content_targets = extractor(input_image)['content']
        with tf.GradientTape() as tape:
            stylized_image = stylizer_network(input_image)
            outputs = extractor(stylized_image)
            loss = total_loss(outputs, style_targets, content_targets, len(style_layers), len(content_layers), stylized_image)

        variables = stylizer_network.trainable_variables
        grad = tape.gradient(loss, variables)
        opt.apply_gradients(zip(grad,variables))

    print('Loading images...')
    dataset_manager = CocoDatasetManager(target_dim=(224,224), num_images = 1000)
    print('Done loading images.')
    images = dataset_manager.get_images()
    BATCH_SIZE = 20
    batch_position = 0

    for epoch in range(epochs):
        print('Epoch ' + str(epoch + 1) + ' of ' + str(epochs))
        if batch_position == 0:
            dataset_manager.shuffle_loaded_images()
        for image in images[batch_position*BATCH_SIZE: (batch_position+1)*BATCH_SIZE]:
            train_step(image.reshape((1,224,224,3)))
        batch_position += 1
        batch_position %= int(len(images)/BATCH_SIZE)
        trained_img = stylizer_network(content_image).numpy()[0]
        save_image(trained_img, os.path.join(output_path, str(epoch) + '_style_transfer_sample.jpg'))

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

if __name__ == '__main__':
    #test()
    train_stylizer_on_dataset()