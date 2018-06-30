import cv2
import sys
#from mail import sendEmail
from flask import Flask, render_template, Response
from camera import VideoCamera
from mobilenet import MobileNet
import numpy as np
import time
import threading

email_update_interval = 600 # sends an email only once in this time interval
video_camera = VideoCamera(flip=True) # creates a camera object, flip vertically
object_classifier = cv2.CascadeClassifier("models/facial_recognition_model.xml") # an opencv classifier
face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt","res10_300x300_ssd_iter_140000.caffemodel")
mobile_net = MobileNet()

# App Globals (do not edit)
app = Flask(__name__)
last_epoch = 0
pc = []

def check_for_objects():
    global last_epoch
    global pc
    while True:
        frame = video_camera.get_raw_frame()

        if frame is not None:
            fpf = 0
            ppf = 0

            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))
            face_net.setInput(blob)
            detections = face_net.forward()
            # loop over the detections
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with the
                # prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if confidence > 0.5:
                    fpf += 1

            detections = mobile_net.process(frame)
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    idx = int(detections[0, 0, i, 1])
                    if mobile_net.CLASSES[idx] == "person":
                        ppf += 1

            pc.append({"people": ppf, "faces": fpf})



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/info')
def info():
    return "{}".format(pc)

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(video_camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    t = threading.Thread(target=check_for_objects, args=())
    t.daemon = True
    t.start()
    app.run(host='0.0.0.0', debug=False)