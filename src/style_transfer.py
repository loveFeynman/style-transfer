import cv2
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

from image_utilities import flip_BR, pixel_to_decimal, save_image

# mostly just messing with tensorflow-provided code for style transfer (https://www.tensorflow.org/beta/tutorials/generative/style_transfer)

TEST_DIR = '../res/test'
TEST_IMG = os.path.join(TEST_DIR, 'vgg_test_1.jpg')

style_image_1 = '../res/styles/honeycomb_small.jpg'
# style_image_1 = '../res/styles/starry_night_small.jpg'
content_image_1 = '../res/content/antelope_small.jpg'
output_path = '../res/test'

style_weight = 1e-2
content_weight = 1e4
total_variation_weight = 1e8

steps_per_epoch = 50
epochs = 20

tf.enable_eager_execution()

def net():
    style_image = pixel_to_decimal(flip_BR(cv2.imread(style_image_1))).reshape((1, 224, 224, 3))
    content_image = pixel_to_decimal(flip_BR(cv2.imread(content_image_1))).reshape((1, 224, 224, 3))
    image = tf.Variable(content_image)

    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    extractor = StyleContentModel(style_layers, content_layers)

    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']
    opt = tf.keras.optimizers.Adam(lr=0.02, beta_1=0.99, epsilon=1e-1)
    opt = tf.train.AdamOptimizer(learning_rate=0.02, beta1=.99, epsilon=1e-1)

    def style_content_loss(outputs):
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                               for name in style_outputs.keys()])
        style_loss *= style_weight / len(style_layers)

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                                 for name in content_outputs.keys()])
        content_loss *= content_weight / len(content_layers)
        loss = style_loss + content_loss
        return loss

    def train_step(image):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = style_content_loss(outputs) + (total_variation_weight * total_variation_loss(image))

        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))

    for epoch in range(epochs):
        print('Epoch ' + str(epoch + 1) + ' of ' + str(epochs))
        for steps in range(steps_per_epoch):
            train_step(image)
        trained_img = image.read_value()[0].numpy()
        save_image(trained_img, os.path.join(output_path,'style_transfer_sample_' + str(epoch) + '.jpg'))

class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
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

def vgg_layers(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result / (num_locations)

def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def high_pass_x_y(image):
    x_var = image[:,:,1:,:] - image[:,:,:-1,:]
    y_var = image[:,1:,:,:] - image[:,:-1,:,:]
    return x_var, y_var

def total_variation_loss(image):
    x_deltas, y_deltas = high_pass_x_y(image)
    return tf.reduce_mean(x_deltas**2) + tf.reduce_mean(y_deltas**2)

def test():
    content_image = cv2.imread(TEST_IMG)

    x = tf.keras.applications.vgg19.preprocess_input(content_image)
    x = cv2.resize(x, (224, 224)).reshape((1, 224, 224, 3))
    vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
    predict = vgg.predict(x)
    labels = tf.keras.applications.vgg19.decode_predictions(predict)
    print(labels)
    for layer in vgg.layers:
        print(layer.name)

if __name__ == '__main__':
    #test()
    net()