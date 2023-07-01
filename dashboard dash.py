import cv2
import sys
import PyQt5
from PyQt5.QtWidgets import  QWidget, QLabel, QApplication
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from ultralytics import YOLO
import os

class Thread(QThread):
    changePixmap = pyqtSignal(QImage)

    def run(self):
        cap = cv2.VideoCapture(0)
        model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

        while True:
            ret, image = cap.read()
            if ret:
                # https://stackoverflow.com/a/55468544/6622587

                # model(image,save=True)
                # folders = os.listdir("C:/Users/20171949/Documents/Programming/runs/detect")[-1]
                # image2= os.listdir(f"C:/Users/20171949/Documents/Programming/runs/detect/{folders}")[-1]
                # image2 = cv2.imread(f"C:/Users/20171949/Documents/Programming/runs/detect/{folders}/{image2}")
                result = model.predict(image)  # default
                image2 = result[0].plot()
                rgbImage = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'YOLO Tool'
        self.left = 100
        self.top = 100
        self.width = 640
        self.height = 480
        self.initUI()

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.resize(1800, 1200)
        # create a label
        self.label = QLabel(self)
        self.label.move(280, 120)
        self.label.resize(640, 480)
        th = Thread(self)
        th.changePixmap.connect(self.setImage)
        th.start()
        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())