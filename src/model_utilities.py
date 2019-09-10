import os
from typing import List, Tuple

from tensorflow.python.keras.saving import load_model

import constants
from image_utilities import sort_numerical


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