import json

#mostly just brainstorming for ways to manage hyperparameters

def read_hyperparameters_from_file(path):
    params = {}
    with open(path, 'r') as file:
        params = json.load(file)
    if 'hyper_parameters' in globals():
        global hyper_parameters
        hyper_parameters = params


def load_hyperparameters(): #maybe only try to load parameters with names starting with ##, the rest can be accessed via dict or p('key')
    if 'hyper_parameters' in globals():
        global hyper_parameters
        for key in hyper_parameters:
            value = hyper_parameters[key]
            exec('global %s' % key)
            if type(value) == type('str') and value[:2] == '##':
                exec('%s = %s' % (key, value[2:]))
            else:
                exec('%s = %s' % (key, 'value'))


def save_hyperparameters(path, obj=None):
    params = obj
    if obj is None and 'hyper_parameters' in globals():
        global hyper_parameters
        params = hyper_parameters
    with open(path, 'w+') as file:
        json.dump(params, file)


def p(key):
    if 'hyper_parameters' in globals():
        global hyper_parameters
        return hyper_parameters[key]
    else:
        return None