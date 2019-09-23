import os
from typing import List, Tuple
import tensorflow as tf
from tensorflow.python.keras.engine import Layer
from tensorflow.python.keras.saving import load_model

import constants
from image_utilities import sort_numerical


class LayerConfig:
    def __init__(self, num_filters, filter_size, stride, padding='same', activation=tf.nn.relu, use_batch_norm=False, use_instance_norm=True):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.use_instance_norm = use_instance_norm

    def copy(self):
        return LayerConfig(self.num_filters,
                           self.filter_size,
                           self.stride,
                           self.padding,
                           self.activation,
                           self.use_batch_norm,
                           self.use_instance_norm)
    def __repr__(self):
        return '<LayerConfig %d, %dx%d, %d stride, %s pad, %s activation, %r bn, %r in>' % \
               (self.num_filters, self.filter_size, self.filter_size, self.stride, self.padding, str(self.activation), self.use_batch_norm, self.use_instance_norm)


# used for configuring loss/style-image pairings
class StyleConfig:
    def __init__(self, style_path, style_weight, content_weight, var_weight):
        self.style_path = style_path
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.total_variation_weight = var_weight
    def __repr__(self):
        return '<StyleConfig ' + str({
            'style_weight': self.style_weight,
            'content_weight': self.content_weight,
            'var_weight': self.total_variation_weight,
            'style_path' : self.style_path
        }) + '>'


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

    @staticmethod
    def instance_norm(layer_input):
        return tf.contrib.layers.instance_norm(layer_input)


class GraphModelBuilder(ModelBuilder):
    @staticmethod
    def conv_block(layer_input, config: LayerConfig):
        conv_layer = tf.layers.conv2d(layer_input, config.num_filters, config.filter_size, config.stride, config.padding) #, kernel_initializer=tf.zeros_initializer())
        if config.activation is not None:
            conv_layer = config.activation(conv_layer)
        if config.use_instance_norm:
            conv_layer = ModelBuilder.instance_norm(conv_layer)
        if config.use_batch_norm and not config.use_instance_norm:
            conv_layer = tf.layers.batch_normalization(conv_layer)
        return conv_layer
    @staticmethod
    def deconv_block(layer_input, config: LayerConfig):
        deconv_layer = tf.layers.conv2d_transpose(layer_input, config.num_filters, config.filter_size, config.stride, config.padding) #, kernel_initializer=tf.zeros_initializer())
        if config.activation is not None:
            deconv_layer = config.activation(deconv_layer)
        if config.use_instance_norm:
            deconv_layer = ModelBuilder.instance_norm(deconv_layer)
        if config.use_batch_norm and not config.use_instance_norm:
            deconv_layer = tf.layers.batch_normalization(deconv_layer)
        return deconv_layer
    @staticmethod
    def residual_block(layer_input, config: LayerConfig):
        conv_config = config.copy()
        conv_config.use_batch_norm = False
        conv_config.use_instance_norm = False
        conv_config.activation = None
        residual_layer = GraphModelBuilder.conv_block(layer_input, config)
        residual_layer = GraphModelBuilder.conv_block(residual_layer, conv_config)
        residual_layer = layer_input + residual_layer
        if config.activation is not None:
            residual_layer = config.activation(residual_layer)
        if config.use_instance_norm:
            residual_layer = ModelBuilder.instance_norm(residual_layer)
        if config.use_batch_norm and not config.use_instance_norm:
            residual_layer = tf.layers.batch_normalization(residual_layer)

        return residual_layer

    @staticmethod
    def conv_layer_from_weights(layer_input, weights, bias):
        return tf.nn.bias_add(tf.nn.conv2d(layer_input, tf.constant(weights), strides=(1,1,1,1), padding='SAME'), bias)
    @staticmethod
    def pool_layer(layer_input):
        return tf.nn.max_pool2d(layer_input, ksize=(1,2,2,1), strides=(1,2,2,1), padding='SAME')


class InstanceNormLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, x):
        return ModelBuilder.instance_norm(x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        base_config = super().get_config()
        return base_config


class EagerModelBuilder(ModelBuilder):
    @staticmethod
    def conv_block(layer_input, config: LayerConfig):
        conv_layer = tf.keras.layers.Conv2D(config.num_filters, config.filter_size, config.stride, padding=config.padding)(layer_input)
        if config.activation is not None:
            conv_layer = tf.keras.layers.Activation(config.activation)(conv_layer)
        if config.use_instance_norm:
            conv_layer = InstanceNormLayer()(conv_layer)
        if config.use_batch_norm and not config.use_instance_norm:
            conv_layer = tf.keras.layers.BatchNormalization()(conv_layer)

        return conv_layer

    @staticmethod
    def deconv_block(layer_input, config: LayerConfig):
        deconv_layer = tf.keras.layers.Conv2DTranspose(config.num_filters, config.filter_size, config.stride, padding=config.padding)(layer_input)
        if config.activation is not None:
            deconv_layer = tf.keras.layers.Activation(config.activation)(deconv_layer)
        if config.use_instance_norm:
            deconv_layer = InstanceNormLayer()(deconv_layer)
        if config.use_batch_norm and not config.use_instance_norm:
            deconv_layer = tf.keras.layers.BatchNormalization()(deconv_layer)

        return deconv_layer

    @staticmethod
    def residual_block(layer_input, config: LayerConfig):
        conv_config = config.copy()
        conv_config.use_batch_norm = False
        conv_config.use_instance_norm = False
        conv_config.activation = None

        residual_layer = EagerModelBuilder.conv_block(layer_input, config)
        residual_layer = EagerModelBuilder.conv_block(residual_layer, conv_config)
        residual_layer = tf.keras.layers.Add()([layer_input, residual_layer])
        if config.activation is not None:
            residual_layer = tf.keras.layers.Activation(config.activation)(residual_layer)
        if config.use_instance_norm:
            residual_layer = InstanceNormLayer()(residual_layer)
        if config.use_batch_norm and not config.use_instance_norm:
            residual_layer = tf.keras.layers.BatchNormalization()(residual_layer)

        return residual_layer


class NNGraphModelBuilder(ModelBuilder):
    @staticmethod
    def var_init(shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=.1, seed=1), dtype=tf.float32)

    @staticmethod
    def conv_block(layer_input, config: LayerConfig):
        weights = NNGraphModelBuilder.var_init([config.filter_size, config.filter_size, layer_input.get_shape()[3].value, config.num_filters])
        conv_layer = tf.nn.conv2d(layer_input, weights, [1, config.stride, config.stride, 1], padding=config.padding.upper())
        if config.use_instance_norm:
            conv_layer = NNGraphModelBuilder.instance_norm(conv_layer)
        if config.activation is not None:
            conv_layer = config.activation(conv_layer)
        return conv_layer

    @staticmethod
    def deconv_block(layer_input, config: LayerConfig):
        weights = NNGraphModelBuilder.var_init([config.filter_size, config.filter_size, config.num_filters, layer_input.get_shape()[3].value])
        layer_shape = tf.stack([tf.shape(layer_input)[0], tf.shape(layer_input)[1] * config.stride, tf.shape(layer_input)[2] * config.stride, config.num_filters])
        deconv_layer = tf.nn.conv2d_transpose(layer_input, weights, layer_shape, [1, config.stride, config.stride, 1], padding=config.padding.upper())
        if config.use_instance_norm:
            deconv_layer = NNGraphModelBuilder.instance_norm(deconv_layer)
        if config.activation is not None:
            deconv_layer = config.activation(deconv_layer)
        return deconv_layer

    @staticmethod
    def residual_block_alt(layer_input, config: LayerConfig):
        conv_config = config.copy()
        conv_config.use_batch_norm = False
        conv_config.use_instance_norm = False
        conv_config.activation = None
        residual_layer = NNGraphModelBuilder.conv_block(layer_input, config)
        residual_layer = NNGraphModelBuilder.conv_block(residual_layer, conv_config)
        residual_layer = layer_input + residual_layer
        if config.use_instance_norm:
            residual_layer = NNGraphModelBuilder.instance_norm(residual_layer)
        if config.activation is not None:
            residual_layer = config.activation(residual_layer)
        return residual_layer

    @staticmethod
    def residual_block(layer_input, config: LayerConfig):
        conv_config = config.copy()
        conv_config.use_batch_norm = False
        conv_config.use_instance_norm = True
        conv_config.activation = None
        residual_layer = NNGraphModelBuilder.conv_block(layer_input, config)
        residual_layer = NNGraphModelBuilder.conv_block(residual_layer, conv_config)
        residual_layer = layer_input + residual_layer
        return residual_layer

    @staticmethod
    def instance_norm(layer_input):
        batch, rows, cols, channels = [i.value for i in layer_input.get_shape()]
        shift = tf.Variable(tf.zeros([channels]))
        scale = tf.Variable(tf.ones([channels]))
        mu, sigma_sq = tf.nn.moments(layer_input, [1, 2], keep_dims=True)
        normalized = (layer_input - mu) / (sigma_sq + 1e-3) ** (.5)
        return scale * normalized + shift


class StyleTransfer:
    STYLE_WEIGHT = 1e-2
    TOTAL_VARIATION_WEIGHT = 1e-4#use 1e8 for normal style transfer
    CONTENT_WEIGHT = 1e8
    # VGG_STYLE_TARGET_LAYER_NAMES = ['block1_conv2',
    #                                 'block2_conv2',
    #                                 'block3_conv4',
    #                                 'block4_conv4',
    #                                 'block5_conv4']
    # VGG_CONTENT_TARGET_LAYER_NAMES = ['block5_conv4']
    VGG_CONTENT_TARGET_LAYER_NAMES = ['block5_conv2']
    VGG_STYLE_TARGET_LAYER_NAMES = [
                                    'block1_conv1',
                                    'block2_conv1',
                                    'block3_conv1',
                                    'block4_conv1',
                                    'block5_conv1'
                                    ]
    VGG_CONTENT_TARGET_LAYER_NAMES_ALT = ['conv5_2']
    VGG_STYLE_TARGET_LAYER_NAMES_ALT = [
        'conv1_1',
        'conv2_1',
        'conv3_1',
        'conv4_1',
        'conv5_1'
    ]
    VGG_CONTENT_TARGET_LAYER_NAMES_ALT_ACT = ['relu5_2']
    VGG_STYLE_TARGET_LAYER_NAMES_ALT_ACT = [
        'relu1_1',
        'relu2_1',
        'relu3_1',
        'relu4_1',
        'relu5_1'
    ]

    STYLE_CONFIG_DICT = {
        'starry_night_transfer': StyleConfig(os.path.join(constants.STYLES_DIR, 'starry_night.jpg'), 1e-2, 1e4, 1e8),
        'starry_night_style': StyleConfig(os.path.join(constants.STYLES_DIR, 'starry_night.jpg'), 1, 0, 0),
        'starry_night_content': StyleConfig(os.path.join(constants.STYLES_DIR, 'starry_night.jpg'), 0, 1, 0),
        # 'starry_night_net': StyleConfig(os.path.join(constants.STYLES_DIR, 'starry_night.jpg'), 1e-3, 4e4, 1e8),

         'honeycomb_transfer': StyleConfig(os.path.join(constants.STYLES_DIR, 'honeycomb_squeeze.jpg'), 1e-2, 1e4, 1e8),
         'honeycomb_style': StyleConfig(os.path.join(constants.STYLES_DIR, 'honeycomb_squeeze.jpg'), 1e-2, 1e4, 1e8),

        'heiro_transfer': StyleConfig(os.path.join(constants.STYLES_DIR, 'heiro.jpg'), 1e-2, 1e4, 1e8),
        'heiro_style': StyleConfig(os.path.join(constants.STYLES_DIR, 'heiro.jpg'), 1, 0, 0),

        'heiro_alt': StyleConfig(os.path.join(constants.STYLES_DIR, 'heiro_alt.jpg'), 1e-2, 1e4, 1e8),

    }

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
        return style_loss_weighted + content_loss_weighted + total_variation_loss_weighted, style_loss_weighted, content_loss_weighted, total_variation_loss_weighted

