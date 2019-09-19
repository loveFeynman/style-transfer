import os
HERE = os.path.dirname(os.path.abspath(__file__))

import types

TENSORBOARD_DIR = os.path.join(HERE, '../tensorboard_dir/')
TRAINING_IMAGES_DIR = os.path.join(HERE, '../res/training_images')
MODELS_DIR = os.path.join(HERE, '../models')
TEST_DIR = os.path.join(HERE, '../res/test')
STYLES_DIR = os.path.join(HERE, '../res/styles')
CONTENT_DIR = os.path.join(HERE, '../res/content')

VAE_KERAS_GENERATED_SAMPLES_DIR = os.path.join(TEST_DIR, 'keras_generated_samples')

STYLIZED_IMAGES_DIR = os.path.join(TEST_DIR, 'stylized_images')

STYLE_TRANSFER_IMAGES_DIR = os.path.join(TEST_DIR, 'style_transfer')

VAE_KERAS_GENERATED_TRAINING_IMAGES_DIR = os.path.join(TRAINING_IMAGES_DIR, 'vae_keras_generated')


STYLIZER_NETWORK_INPUT_DIR = os.path.join(TEST_DIR, 'style_network_input')
STYLIZER_NETWORK_OUTPUT_DIR = os.path.join(TEST_DIR, 'style_network_output')


def make_dirs_if_not_exist(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

def init_dirs():
    global_vars = {k: v for (k, v) in globals().items() if
                   not isinstance(v, types.ModuleType)
                   and not isinstance(v, types.FunctionType)
                   and isinstance(v, type('string'))
                   and '__' not in k
                   and len(k) > 4
                   and k[-4:] == '_DIR'}
    make_dirs_if_not_exist(list(global_vars.values()))

init_dirs()

if __name__ == '__main__':
    global_vars = {k:v for (k,v) in globals().items() if '__' not in k and not isinstance(v, types.ModuleType) and not isinstance(v, types.FunctionType)}
    print(global_vars.keys())
    print(globals())
    # print(locals())