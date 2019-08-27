import threading
import http.client
import PyQt5
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QMainWindow, QApplication, QAction, qApp, QLabel, QLineEdit, QFileDialog, QPushButton, QComboBox, QSlider

from image_utilities import save_image, flip_RB, flip_BR, pixel_to_decimal
from vae_keras import VAEService, get_most_recent_vae_name, decimal_to_pixel
import time
import cv2
import numpy as np
from constants import *

a_initial_image = os.path.join(VAE_KERAS_GENERATED_SAMPLES_DIR,'2_truth.jpg')
b_initial_image = os.path.join(VAE_KERAS_GENERATED_SAMPLES_DIR,'34_truth.jpg')
tick_interval = 100

IMAGE_DIM = (224, 224)
IMAGE_WIDTH  = IMAGE_DIM[0]
IMAGE_HEIGHT  = IMAGE_DIM[1]
HALF_WIDTH  = int(IMAGE_WIDTH/2)
HALF_HEIGHT  = int(IMAGE_HEIGHT/2)
MARGIN = 10
BUTTON_HEIGHT = 30
SLIDER_HEIGHT = 30

class ImageInfo:
    def __init__(self, path, vae_service: VAEService):
        self.path = path
        self.pix = QPixmap(self.path)
        while not vae_service.loaded:
            time.sleep(.1)
        self.image = pixel_to_decimal(flip_BR(cv2.imread(self.path)))

        vectors, _ = vae_service.get_vectors_and_samples_from_images([self.image])
        self.vector = vectors[0]


class ControlsWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.vae_service = VAEService(get_most_recent_vae_name())
        self.vae_service.start()

        self.initialize_components()

        self.requesting_image = False


    def initialize_components(self):
        load_image_a_button = QPushButton('Load Image A', self)
        load_image_a_button.clicked.connect(self.load_image_a_handler)
        load_image_a_button.setGeometry(MARGIN, MARGIN, IMAGE_WIDTH, BUTTON_HEIGHT)

        load_image_b_button = QPushButton('Load Image B', self)
        load_image_b_button.clicked.connect(self.load_image_b_handler)
        load_image_b_button.setGeometry((IMAGE_WIDTH * 2) + (MARGIN * 3), MARGIN, IMAGE_WIDTH, BUTTON_HEIGHT)

        self.image_a_info = ImageInfo(a_initial_image, self.vae_service)
        self.image_b_info = ImageInfo(b_initial_image, self.vae_service)

        self.image_a_label = QLabel(self)
        self.image_a_label.setPixmap(self.image_a_info.pix)
        self.image_a_label.setGeometry(MARGIN, BUTTON_HEIGHT + (MARGIN * 2), IMAGE_WIDTH, IMAGE_HEIGHT)

        self.image_b_label = QLabel(self)
        self.image_b_label.setPixmap(self.image_b_info.pix)
        self.image_b_label.setGeometry((IMAGE_WIDTH * 2) + (MARGIN * 3), BUTTON_HEIGHT + (MARGIN * 2), IMAGE_WIDTH, IMAGE_HEIGHT)

        self.image_c_label = QLabel(self)
        self.image_c_label.setPixmap(self.image_b_info.pix)
        self.image_c_label.setGeometry(IMAGE_WIDTH + (MARGIN * 2), BUTTON_HEIGHT + (MARGIN * 2), IMAGE_WIDTH, IMAGE_HEIGHT)

        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(tick_interval)
        self.slider.setSingleStep(1)
        # self.slider.sliderReleased.connect(self.slider_handler)
        self.slider.sliderMoved.connect(self.slider_handler)
        self.slider.setGeometry(MARGIN + HALF_WIDTH, IMAGE_HEIGHT + (MARGIN * 3) + BUTTON_HEIGHT, (IMAGE_WIDTH * 2) + (MARGIN * 2), SLIDER_HEIGHT)

        self.setGeometry(100, 100, (IMAGE_WIDTH * 3) + (MARGIN * 4), IMAGE_HEIGHT + (MARGIN * 4) + BUTTON_HEIGHT + SLIDER_HEIGHT)
        self.setWindowTitle('Interpolater')
        self.show()

    def set_image(self, image_pos, image_path):
        image_info = ImageInfo(image_path, self.vae_service)
        image = None

        if image_pos == 'A':
            image = self.image_a_label
            self.image_a_info = image_info
        if image_pos == 'B':
            image = self.image_b_label
            self.image_b_info = image_info
        if image_pos == 'C':
            image = self.image_c_label

        image.setPixmap(image_info.pix)

    def load_image_a_handler(self):
        self.load_image_handler('A')
    def load_image_b_handler(self):
        self.load_image_handler('B')
    def load_image_handler(self, image_pos):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        full_path, _ = QFileDialog.getOpenFileName(self, 'Select File to Load', '', 'Images (*.jpg)', options=options)
        if full_path:
            self.set_image(image_pos, full_path)


    def slider_handler(self):
        if self.requesting_image:
            return
        self.requesting_image = True

        sample_increment = (self.image_b_info.vector - self.image_a_info.vector) / float(tick_interval)
        sample_vector = (sample_increment * float(self.slider.value())) + self.image_a_info.vector
        img = self.vae_service.get_image_from_vector(sample_vector)
        img_pix = decimal_to_pixel(img)
        qimg = QImage(img_pix.data, img_pix.shape[1], img_pix.shape[0], 3 * img_pix.shape[1], QImage.Format_RGB888)
        self.image_c_label.setPixmap(QPixmap(qimg))

        self.requesting_image = False



if __name__ == '__main__':





    app = QApplication([])
    window = ControlsWindow()
    app.exec_()
