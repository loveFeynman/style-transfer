import threading
import http.client
import PyQt5
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QAction, qApp, QLabel, QLineEdit, QFileDialog, QPushButton, QComboBox, QSlider


# class Renderer(threading.Thread):
#     def __init__(self):
#         super().__init__()
#     def run(self):
#         pass


url = 'localhost:8000'

class ControlsWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initialize_components()

        self.requesting_image = False

    def initialize_components(self):
        load_image_a_button = QPushButton('Load Image A', self)
        load_image_a_button.clicked.connect(self.load_image_a_handler)
        load_image_a_button.setGeometry(10,10,128,30)

        load_image_b_button = QPushButton('Load Image B', self)
        load_image_b_button.clicked.connect(self.load_image_b_handler)
        load_image_b_button.setGeometry(286,10,128,30)

        self.image_a_path = '../res/test/4_truth.jpg'
        self.image_b_path = '../res/test/2_truth.jpg'

        self.image_a_label = QLabel(self)
        self.image_a_label.setPixmap(QPixmap(self.image_a_path))
        self.image_a_label.setGeometry(10, 50, 128, 128)

        self.image_b_label = QLabel(self)
        self.image_b_label.setPixmap(QPixmap(self.image_b_path))
        self.image_b_label.setGeometry(286, 50, 128, 128)

        self.image_c_label = QLabel(self)
        self.image_c_label.setPixmap(QPixmap(self.image_b_path))
        self.image_c_label.setGeometry(148, 50, 128, 128)

        slider = QSlider(Qt.Horizontal,self)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(100)
        slider.setSingleStep(1)
        slider.sliderReleased.connect(self.slider_handler)
        #slider.sliderMoved.connect(self.slider_handler)
        slider.setGeometry(74,188,296,20)

        self.setGeometry(100, 100, 424, 300)
        self.setWindowTitle('Interpolater')
        self.show()

    def set_image(self,image_pos, image_path):
        image = None
        img_path = None
        if image_pos == 'A':
            image = self.image_a_label
            img_path = self.image_a_path
        if image_pos == 'B':
            image = self.image_b_label
            img_path = self.image_b_path
        if image_pos == 'C':
            image = self.image_c_label
        image.setPixmap(QPixmap(image_path))
        img_path = image_path

    def load_image_a_handler(self):
        self.load_image_handler('A')
    def load_image_b_handler(self):
        self.load_image_handler('B')
    def load_image_handler(self, image_pos):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        full_path, _ = QFileDialog.getOpenFileName(self, 'Select File to Load', '', 'Images (*.jpg)', options=options)
        if full_path:
            self.set_image(image_pos,full_path)

    def slider_handler(self):
        if self.requesting_image:
            return
        self.requesting_image = True
        connection = http.client.HTTPConnection(url)
        connection.request('GET', '/vae/')
        response = connection.getresponse()
        if response.status == 200:
            data = response.read()
            print(data)

        print('slider handler hit')
        self.requesting_image = False

    def send_image(self,image_path):
        print('posting image')
        connection = http.client.HTTPConnection(url)
        connection.request('POST', '/vae/','test post body')
        response = connection.getresponse()
        if response.status == 200:
            data = response.read()
            print(data)



if __name__ == '__main__':





    app = QApplication([])
    window = ControlsWindow()
    app.exec_()
