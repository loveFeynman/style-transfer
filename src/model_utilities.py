import os
from typing import List, Tuple
import tensorflow as tf
from tensorflow.python.keras.saving import load_model

import constants
from image_utilities import sort_numerical


class LayerConfig:
    def __init__(self, num_filters, filter_size, stride, padding='same', activation=tf.nn.relu, use_batch_norm=True):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.use_batch_norm = use_batch_norm

    def copy(self):
        return LayerConfig(self.num_filters,
                           self.filter_size,
                           self.stride,
                           self.padding,
                           self.activation,
                           self.use_batch_norm)


def load_keras_model(dir, name, special_layers=None):
    target_dir = os.path.join(dir, name)
    model_paths = sort_numerical([os.path.join(target_dir, x) for x in os.listdir(target_dir) if '.h5' in x])
    models = [load_single_keras_model(x, special_layers=special_layers) for x in model_paths]
    if len(models) == 1:
        return models[0]
    else:
        return models


def save_keras_model(model, dir, name):
    target_dir = os.path.join(dir, name)
    create_dir_if_not_exists(target_dir)

    if isinstance(model, (list, tuple)):
        for x in range(len(model)):
            save_single_keras_model(model[x], target_dir, name + '_' + str(x))
    else:
        save_single_keras_model(model, target_dir, name)
    # model.save_weights(os.path.join(dir, name + '.h5'))
    # with open(os.path.join(dir, name + '.json'), 'w+') as js:
    #     js.write(model.to_json())


def save_single_keras_model(model, dir, name):
    # model.save_weights(os.path.join(dir, name + '.h5'))
    # with open(os.path.join(dir, name + '.json'), 'w+') as js:
    #     js.write(model.to_json())
    model.save(os.path.join(dir, name + '.h5'))


def load_single_keras_model(path, special_layers=None):
    # with open(os.path.join(dir, name + '.json')) as js:
    #     model = model_from_json(js.read())
    # model.load_weights(os.path.join(dir, name + '.h5'))
    custom_objects = {}
    if isinstance(special_layers, (dict)):
        custom_objects = special_layers
    model = load_model(path, custom_objects=custom_objects)
    return model


def get_most_recent_model_name(dir, prefix):
    models = os.listdir(dir)
    models = [x for x in models if prefix in x]
    # models.sort()
    sort_numerical(models)
    return models[-1]


def create_dir_if_not_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

#theoretically should be using ABC
class ModelBuilder:
    @staticmethod
    def conv_block(layer_input, config: LayerConfig):
        return

    @staticmethod
    def deconv_block(layer_input, config: LayerConfig):
        return

    @staticmethod
    def residual_block(layer_input, config: LayerConfig):
        return


class GraphModelBuilder(ModelBuilder):
    @staticmethod
    def conv_block(layer_input, config: LayerConfig):
        conv_layer = tf.layers.conv2d(layer_input, config.num_filters, config.filter_size, config.stride, config.padding)
        if config.use_batch_norm:
            conv_layer = tf.layers.batch_normalization(conv_layer)
        if config.activation is not None:
            conv_layer = config.activation(conv_layer)
        return conv_layer
    @staticmethod
    def deconv_block(layer_input, config: LayerConfig):
        deconv_layer = tf.layers.conv2d_transpose(layer_input, config.num_filters, config.filter_size, config.stride, config.padding)
        if config.use_batch_norm:
            deconv_layer = tf.layers.batch_normalization(deconv_layer)
        if config.activation is not None:
            deconv_layer = config.activation(deconv_layer)
        return deconv_layer
    @staticmethod
    def residual_block(layer_input, config: LayerConfig):
        conv_config = config.copy()
        conv_config.use_batch_norm = False
        conv_config.activation = None
        residual_layer = GraphModelBuilder.conv_block(layer_input, config)
        residual_layer = GraphModelBuilder.conv_block(residual_layer, conv_config)
        residual_layer = layer_input + residual_layer
        if config.use_batch_norm:
            residual_layer = tf.layers.batch_normalization(residual_layer)
        if config.activation is not None:
            residual_layer = config.activation(residual_layer)
        return residual_layer


class EagerModelBuilder(ModelBuilder):
    @staticmethod
    def conv_block(layer_input, config: LayerConfig):
        conv_layer = tf.keras.layers.Conv2D(config.num_filters, config.filter_size, config.stride, config.padding)(layer_input)
        if config.use_batch_norm:
            conv_layer = tf.keras.layers.BatchNormalization()(conv_layer)
        conv_layer = tf.keras.layers.Activation(config.activation)(conv_layer)
        return conv_layer

    @staticmethod
    def deconv_block(layer_input, config: LayerConfig):
        deconv_layer = tf.keras.layers.Conv2DTranspose(config.num_filters, config.filter_size, config.stride, config.padding)(layer_input)
        if config.use_batch_norm:
            deconv_layer = tf.keras.layers.BatchNormalization()(deconv_layer)
        deconv_layer = tf.keras.layers.Activation(config.activation)(deconv_layer)
        return deconv_layer

    @staticmethod
    def residual_block(layer_input, config: LayerConfig):
        conv_config = config.copy()
        conv_config.use_batch_norm = False
        conv_config.activation = None

        residual_layer = EagerModelBuilder.conv_block(layer_input, config)
        residual_layer = EagerModelBuilder.conv_block(residual_layer, conv_config)
        residual_layer = tf.keras.layers.Add()([layer_input, residual_layer])
        if config.use_batch_norm:
            residual_layer = tf.keras.layers.BatchNormalization()(residual_layer)
        if config.activation != None:
            residual_layer = tf.keras.layers.Activation(config.activation)(residual_layer)
        return residual_layer


class StyleTransfer:
    STYLE_WEIGHT = 1e-2
    TOTAL_VARIATION_WEIGHT = 1e8
    CONTENT_WEIGHT = 1e4
    VGG_STYLE_TARGET_LAYER_NAMES = ['block1_conv2',
                                    'block2_conv2',
                                    'block3_conv4',
                                    'block4_conv4',
                                    'block5_conv4']
    VGG_CONTENT_TARGET_LAYER_NAMES = ['block3_conv4']

    @staticmethod
    def clip_0_1(image):
        return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    @staticmethod
    def high_pass_x_y(image):
        x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
        y_var = image[:, 1:, :, :] - image[:, :-1, :, :]
        return x_var, y_var

    @staticmethod
    def total_variation_loss(image):
        x_deltas, y_deltas = StyleTransfer.high_pass_x_y(image)
        return tf.reduce_mean(x_deltas ** 2) + tf.reduce_mean(y_deltas ** 2)

    @staticmethod
    def gram_matrix(input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return result / (num_locations)

    @staticmethod
    def sum_mse(array_1, array_2):
        return tf.add_n([tf.reduce_mean((array_1[x] - array_2[x]) ** 2) for x in range(len(array_1))])

    @staticmethod
    def total_loss(style_network_outputs, style_targets, content_targets, vgg_style_outputs, vgg_content_outputs,
                   style_weight=STYLE_WEIGHT,
                   content_weight=CONTENT_WEIGHT,
                   total_variation_weight=TOTAL_VARIATION_WEIGHT
                   ):
        style_loss_weighted = StyleTransfer.sum_mse(vgg_style_outputs, style_targets) * style_weight / len(vgg_content_outputs)
        content_loss_weighted = StyleTransfer.sum_mse(vgg_content_outputs, content_targets) * content_weight
        total_variation_loss_weighted = StyleTransfer.total_variation_loss(style_network_outputs) * total_variation_weight
        return style_loss_weighted + content_loss_weighted + total_variation_loss_weighted

