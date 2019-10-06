import time

import cv2
from os.path import join
import os
import numpy as np
import imageio
from typing import overload
import constants


HERE = os.path.dirname(os.path.abspath(__file__))



COCO_DATASET = os.path.join(HERE, '../res/coco/coco/train2014/train2014/')
COCO_DATASET_CLIPPED = os.path.join(HERE, '../res/coco/coco_clipped')

DEBUG_MODE = False


class ImageGenerator:
    def __init__(self, styles = ('starry_night','honeycomb')):
        self.styles = []
        for style in styles:
            img = open_image(join(constants.STYLES_DIR,style + '.jpg'))
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
        debug('Beginning image load')
        self.load_images(directory, num_images)

    def load_images(self, directory, num_images = None):
        start = time.time()
        files = os.listdir(directory)
        debug('Found ' + str(len(files)) + ' images in \"' + directory + '\"')
        debug('duration = ' + str(time.time() - start))

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
    def __init__(self, target_dim = (256, 256), num_images = None):
        super().__init__(COCO_DATASET, target_dim = target_dim, num_images = num_images)


class PreprocessedCocoDatasetManager:
    def __init__(self, target_dim=(256, 256), num_images=None):
        self.images = []
        self.target_dim = target_dim
        debug('Beginning image load')
        self.load_images(COCO_DATASET_CLIPPED, num_images)

    def load_images(self, directory, num_images=None):
        start = time.time()
        files = os.listdir(directory)


        for file in files:
            if len(self.images) > num_images:
                continue
            if '.jpg' not in file:
                continue
            path = os.path.join(directory, file)
            img = cv2.imread(path)
            img = pixel_to_decimal(flip_BR(img))
            self.images.append(img)


    def shuffle_loaded_images(self):
        for x in range(len(self.images)):
            random_indexes = np.random.randint(len(self.images), size=2)
            hold = self.images[random_indexes[0]]
            self.images[random_indexes[0]] = self.images[random_indexes[1]]
            self.images[random_indexes[1]] = hold

    def get_images(self):
        return self.images




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

def image_3d_to_4d(img):
    image = None
    if not isinstance(img, np.ndarray):
        image = np.array(img)
    else:
        image = img
    if image.shape[0] != 1 and len(image.shape) == 3:
        image = np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))

    return image

def image_4d_to_3d(img):
    image = None
    if isinstance(img, np.ndarray) and img.shape[0] == 1 and len(img.shape) == 4:
        image = np.reshape(img, (img.shape[1], img.shape[2], img.shape[3]))
    else:
        image = img
    return image

def save_image(img, path):
    image = image_4d_to_3d(img)
    image = decimal_to_pixel(image)
    image = flip_RB(image)
    cv2.imwrite(path, image)

def open_image(path):
    image = cv2.imread(path)
    image = flip_BR(image)
    image = pixel_to_decimal(image)
    return image

def load_image(path):
    return image_3d_to_4d(open_image(path))

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


def generate_gif_from_images_in_dir(dir_path):
    output_path = os.path.join(dir_path, os.path.basename(os.path.normpath(dir_path) + '.gif'))
    files = os.listdir(dir_path)
    files = [os.path.join(dir_path, x) for x in files if '.jpg' in x]
    files = sort_numerical(files)
    print('Generating gif of ' + str(len(files)) + ' files')
    with imageio.get_writer(output_path, mode='I') as stream:
        for file in files:
            image = imageio.imread(file)
            stream.append_data(image)
    # images = [imageio.imread(x) for x in files]
    # images = images + list(reversed(images))
    # imageio.mimsave(output_path, images)


def sort_numerical(items):
    items.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    return items


def clip_COCO_to_dims(num_images=2000, start=0, target_dim=(256,256)):
    directory = COCO_DATASET
    files = os.listdir(directory)

    if len(files) > start+num_images:
        files = files[start:start+num_images]


    count = 0
    for file in files:
        if '.jpg' not in file:
            continue
        path = os.path.join(directory, file)
        out_path = os.path.join(COCO_DATASET_CLIPPED, file)

        img = cv2.imread(path)
        if can_clip_image_to_dims(img, dim=target_dim):
            img = random_clip_image_to_dim(img, dim=target_dim)
            cv2.imwrite(out_path, img)
        print(count)
        count += 1



if __name__ == '__main__':
    #test()
    # generate_gif_from_images_in_dir(os.path.join(constants.STYLE_TRANSFER_IMAGES_DIR, 'starry_night'))
    # generate_gif_from_images_in_dir(os.path.join(constants.STYLE_TRANSFER_IMAGES_DIR, 'honeycomb'))
    clip_COCO_to_dims(start=11000, num_images=5000)
    # pass

