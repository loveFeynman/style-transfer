import os

LOGDIR = '../tensorboard_dir/'
TRAINING_IMAGES_DIR = '../res/training_images'
MODELS_DIR = '../models'
SAMPLES_DIR = '../res/test'
STYLES_DIR = '../res/styles'

VAE_KERAS_GENERATED_SAMPLES_DIR = os.path.join(SAMPLES_DIR, 'keras_generated_samples')


VAE_KERAS_GENERATED_TRAINING_IMAGES = os.path.join(TRAINING_IMAGES_DIR, 'vae_keras_generated')