import os
import scipy.io
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
from model_utilities import StyleTransfer

VGG_PATH = os.path.join(HERE, '../res/vgg19/imagenet-vgg-verydeep-19.mat')

vgg_preprocessing_mean = np.array([0, 0, 0])

def vgg_preprocess_input(network_input):
    normalized_value = network_input
    # if (tf.math.reduce_min(normalized_value) > 0. and tf.math.reduce_max(normalized_value) < 1.):
    normalized_value *= 255.
    normalized_value /= 2.
    normalized_value -= vgg_preprocessing_mean
    return normalized_value

def build_vgg_network_from_mat(network_input):

    print('Loading VGG19 data...')
    vgg_data = scipy.io.loadmat(VGG_PATH)
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


def build_vgg_network_replicate_from_mat(network_inputs, replications=1):
    '''
    builds multiple vgg networks from the same weights
    :param network_inputs:
    :param replications:
    :return:
    '''
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