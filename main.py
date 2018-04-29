import cv2
import sys
#from mail import sendEmail
from flask import Flask, render_template, Response
from camera import VideoCamera
from mobilenet import MobileNet
from age_gender import FaceCV
import numpy as np
import time
import threading

email_update_interval = 600 # sends an email only once in this time interval
video_camera = VideoCamera(flip=True) # creates a camera object, flip vertically
object_classifier = cv2.CascadeClassifier("models/facial_recognition_model.xml") # an opencv classifier
mobile_net = MobileNet()
faceCV = FaceCV

# App Globals (do not edit)
app = Flask(__name__)
last_epoch = 0
pc = []

def check_for_objects():
    global last_epoch
    global pc
    while True:
        try:
            frame, found_obj, count_objects = video_camera.get_object(object_classifier)

            detections = mobile_net.process(frame)
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    idx = int(detections[0, 0, i, 1])

                    if mobile_net.CLASSES[idx] == "person":
                        (h, w) = frame.shape[:2]
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        (startX, startY, endX, endY) = (startX.item(), startY.item(), endX.item(), endY.item())
                        fc = faceCV.detect_face(frame[startY: endY, startX:endX])

                        if len(fc) > 0:
                            pc.append("0 Faces")
                        else:
                            pc.append("{} Faces".format(len(fc)))

            if found_obj == True:
                print("Found {} people".format(count_objects))
        except:
            print("Error")

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