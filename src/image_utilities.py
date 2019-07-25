import cv2
from os.path import join
import numpy as np

STYLES_DIR = '../res/styles'
TEST_DIR = '../res/test'

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

class TrainingImageManager(): # TODO: complete this
    def generate_style_images(self):
        pass
    def load_style_images(self):
        pass
    def load_test_images(self):
        self.test_images = []


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

if __name__ == '__main__':
    test()
