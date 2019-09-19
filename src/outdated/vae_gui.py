from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QFileDialog, QPushButton, QSlider

from image_utilities import flip_BR, pixel_to_decimal
from outdated.vae import VAEService, get_most_recent_vae, decimal_to_pixel
import time
import cv2

# class Renderer(threading.Thread):
#     def __init__(self):
#         super().__init__()
#     def run(self):
#         pass


url = 'localhost:8000'
a_initial_image = '../res/test/6_truth.jpg'
b_initial_image = '../res/test/2_truth.jpg'
tick_interval = 100

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

        self.vae_service = VAEService(get_most_recent_vae())
        self.vae_service.start()

        self.initialize_components()

        self.requesting_image = False


    def initialize_components(self):
        load_image_a_button = QPushButton('Load Image A', self)
        load_image_a_button.clicked.connect(self.load_image_a_handler)
        load_image_a_button.setGeometry(10, 10, 128, 30)

        load_image_b_button = QPushButton('Load Image B', self)
        load_image_b_button.clicked.connect(self.load_image_b_handler)
        load_image_b_button.setGeometry(286, 10, 128, 30)

        self.image_a_info = ImageInfo(a_initial_image, self.vae_service)
        self.image_b_info = ImageInfo(b_initial_image, self.vae_service)

        self.image_a_label = QLabel(self)
        self.image_a_label.setPixmap(self.image_a_info.pix)
        self.image_a_label.setGeometry(10, 50, 128, 128)

        self.image_b_label = QLabel(self)
        self.image_b_label.setPixmap(self.image_b_info.pix)
        self.image_b_label.setGeometry(286, 50, 128, 128)

        self.image_c_label = QLabel(self)
        self.image_c_label.setPixmap(self.image_b_info.pix)
        self.image_c_label.setGeometry(148, 50, 128, 128)

        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(tick_interval)
        self.slider.setSingleStep(1)
        # self.slider.sliderReleased.connect(self.slider_handler)
        self.slider.sliderMoved.connect(self.slider_handler)
        self.slider.setGeometry(74, 188, 296, 20)

        self.setGeometry(100, 100, 424, 300)
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
        # img_pix = flip_RB(img_pix)
        # cv2.imwrite('../res/test/gui_sample.jpg', img_pix)
        # img_pix = np.transpose(img_pix, (1, 0, 2)).copy()
        #qimg = QImage(img_pix, img_pix.shape[1], img_pix.shape[0], QImage.Format_RGB888)
        qimg = QImage(img_pix.data, img_pix.shape[1], img_pix.shape[0], 3 * img_pix.shape[1], QImage.Format_RGB888)
        self.image_c_label.setPixmap(QPixmap(qimg))

        self.requesting_image = False

    # def slider_handler(self): # initial ideas for when service is running on cloud etc
    #     if self.requesting_image:
    #         return
    #     self.requesting_image = True
    #     connection = http.client.HTTPConnection(url)
    #     connection.request('GET', '/vae/')
    #     response = connection.getresponse()
    #     if response.status == 200:
    #         data = response.read()
    #         print(data)
    #
    #     print('slider handler hit')
    #     self.requesting_image = False

    # def send_image(self,image_path): # initial ideas for when service is running on cloud etc
    #     print('posting image')
    #     connection = http.client.HTTPConnection(url)
    #     connection.request('POST', '/vae/', 'test post body')
    #     response = connection.getresponse()
    #     if response.status == 200:
    #         data = response.read()
    #         print(data)



if __name__ == '__main__':





    app = QApplication([])
    window = ControlsWindow()
    app.exec_()
