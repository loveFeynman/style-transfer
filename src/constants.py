import os

LOGDIR = '../tensorboard_dir/'
TRAINING_IMAGES_DIR = '../res/training_images'
MODELS_DIR = '../models'
TEST_DIR = '../res/test'
STYLES_DIR = '../res/styles'

VAE_KERAS_GENERATED_SAMPLES_DIR = os.path.join(TEST_DIR, 'keras_generated_samples')

STYLIZED_IMAGES_DIR = os.path.join(TEST_DIR, 'stylized_images')

STYLE_TRANSFER_IMAGES_DIR = os.path.join(TEST_DIR, 'style_transfer')

VAE_KERAS_GENERATED_TRAINING_IMAGES = os.path.join(TRAINING_IMAGES_DIR, 'vae_keras_generated')