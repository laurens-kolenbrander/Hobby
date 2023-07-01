import dash
from dash import dcc
from dash import html
from flask import Flask, Response
import cv2
from ultralytics import YOLO
import os
import time

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.video.set(cv2.CAP_PROP_FPS, 0.1)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        time.sleep(2)
        model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
        model(image,save=True)
        folders = os.listdir("../runs/detect")[-1]
        image2= os.listdir(f"../runs/detect/{folders}")[-1]
        image2 = cv2.imread(f"../runs/detect/{folders}/{image2}")
        ret, jpeg = cv2.imencode('.jpg', image2)
        return jpeg.tobytes()


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

server = Flask(__name__)
app = dash.Dash(__name__, server=server)

@server.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

app.layout = html.Div([
    html.H1("Webcam Test"),
    html.Img(src="/video_feed")
])

if __name__ == '__main__':
    app.run_server(debug=True)