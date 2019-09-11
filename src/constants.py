import os
import types

LOG_DIR = '../tensorboard_dir/'
TRAINING_IMAGES_DIR = '../res/training_images'
MODELS_DIR = '../models'
TEST_DIR = '../res/test'
STYLES_DIR = '../res/styles'

VAE_KERAS_GENERATED_SAMPLES_DIR = os.path.join(TEST_DIR, 'keras_generated_samples')

STYLIZED_IMAGES_DIR = os.path.join(TEST_DIR, 'stylized_images')

STYLE_TRANSFER_IMAGES_DIR = os.path.join(TEST_DIR, 'style_transfer')

VAE_KERAS_GENERATED_TRAINING_IMAGES_DIR = os.path.join(TRAINING_IMAGES_DIR, 'vae_keras_generated')

STYLIZER_NETWORK_MODELS_DIR = os.path.join(MODELS_DIR, 'stylizer_network')

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