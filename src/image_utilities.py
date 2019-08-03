import cv2
from os.path import join
import os
import numpy as np
from typing import overload

STYLES_DIR = '../res/styles'
TEST_DIR = '../res/test'

COCO_DATASET = '../../../res/coco/train2014/train2014/'

DEBUG_MODE = False


class ImageGenerator:
    def __init__(self, styles = ('starry_night','honeycomb')):
        self.styles = []
        for style in styles:
            img = open_image(join(STYLES_DIR,style + '.jpg'))
            self.styles.append(img)
        print('ImageGenerator loaded  ' + str(len(self.styles)) + ' styles')
        self.images = {}
    def generate_images(self, category, count, sample_width, sample_height):
        # print('ImageGenerator generating images...')
        self.images[category] = []
        counts_per_style = [int(count/len(self.styles)) for x in self.styles]
        for i in range(count%len(self.styles)):
            counts_per_style[i] += 1

        for i in range(len(self.styles)):
            img = self.styles[i]
            height = img.shape[0]
            width = img.shape[1]

            random_heights = np.random.randint(height-sample_height, size=counts_per_style[i])
            random_widths = np.random.randint(width-sample_width, size=counts_per_style[i])

            sampled_images = [img[random_heights[j]:random_heights[j]+sample_height,random_widths[j]:random_widths[j]+sample_width] for j in range(counts_per_style[i])]

            self.images[category].extend(sampled_images)

        #print('ImageGenerator generated ' + str(len(self.images[category])) + ' images for category \"' + category + '\"')

    def shuffle(self, category):
        for x in range(len(self.images[category])):
            random_indexes = np.random.randint(len(self.images[category]), size=2)
            hold = self.images[category][random_indexes[0]]
            self.images[category][random_indexes[0]] = self.images[category][random_indexes[1]]
            self.images[category][random_indexes[1]] = hold

    def get(self, category):
        return self.images[category]

    def save(self, category, directory):
        for i in range(len(self.images[category])):
            img = self.images[category][i]
            save_image(img,join(directory, category + '_' + str(i) + '.jpg'))

class TrainingImageManager: # TODO: complete this
    def generate_style_images(self):
        pass
    def load_style_images(self):
        pass
    def load_test_images(self):
        self.test_images = []

class DatasetManager:
    def __init__(self, directory, target_dim = (224, 224), num_images = None):
        self.images = []
        self.target_dim = target_dim
        self.load_images(directory, num_images)

    def load_images(self, directory, num_images = None):
        files = os.listdir(directory)
        debug('Found ' + str(len(files)) + ' images in \"' + directory + '\"')

        for file in files:
            if len(self.images) > num_images:
                continue
            if '.jpg' not in file:
                continue
            path = os.path.join(directory, file)

            img = cv2.imread(path)
            if can_clip_image_to_dims(img, dim=self.target_dim):

                img = random_clip_image_to_dim(img, dim=self.target_dim)
                img = pixel_to_decimal(flip_BR(img))

                self.images.append(img)
                debug('Loaded ' + str(len(self.images)) + ' out of ' + str(num_images) + ' (' + path + ')')



    def shuffle_loaded_images(self):
        for x in range(len(self.images)):
            random_indexes = np.random.randint(len(self.images), size=2)
            hold = self.images[random_indexes[0]]
            self.images[random_indexes[0]] = self.images[random_indexes[1]]
            self.images[random_indexes[1]] = hold
    def get_images(self):
        return self.images

class CocoDatasetManager(DatasetManager):
    def __init__(self, target_dim = (224, 224), num_images = None):
        super().__init__(COCO_DATASET, target_dim = target_dim, num_images = num_images)


def clip_image_to_dims(image, width, height, start_x = 0, start_y = 0):
    if image.shape[0] < height - start_y or image.shape[1] < width - start_x:
        return image
    return image[start_y:height+start_y,start_x:width+start_x]


def can_clip_image_to_dims(image, width = None, height = None, dim = None):
    w = 0
    h = 0
    if width != None and height != None:
        w = width
        h = height
    if dim != None:
        w = dim[0]
        h = dim[1]

    return image.shape[0] > h and image.shape[1] > w

def random_clip_image_to_dim(image, width = None, height = None, dim = None):
    w = 0
    h = 0
    if width != None and height != None:
        w = width
        h = height
    if dim != None:
        w = dim[0]
        h = dim[1]

    random_start_x = np.random.randint(image.shape[1] - w, size=1)[0]
    random_start_y = np.random.randint(image.shape[0] - h, size=1)[0]
    return clip_image_to_dims(image, w, h, random_start_x, random_start_y)



def flip_RB(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def flip_BR(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def decimal_to_pixel(img):
    return (img*255.).astype(np.uint8)

def pixel_to_decimal(img):
    image = img
    if type(image) == 'list':
        image = np.array(image)
    return image.astype(np.float32)/255.

def save_image(img, path):
    image = decimal_to_pixel(img)
    image = flip_RB(image)
    cv2.imwrite(path, image)

def open_image(path):
    image = cv2.imread(path)
    image = flip_BR(image)
    image = pixel_to_decimal(image)
    return image

def write_vector(vector, path):
    with open(path, 'w+') as file:
        file.write(str(vector))

def test():
    generator = ImageGenerator()
    generator.generate_images('test', 12, 256, 256)
    generator.save('test', TEST_DIR)

def debug(str):
    if DEBUG_MODE:
        print(str)


if __name__ == '__main__':
    test()
