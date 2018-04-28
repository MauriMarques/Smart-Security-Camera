import imutils
import pkg_resources
import cv2
import numpy as np
import os


class MobileNet:

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]

    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    def __init__(self):
        # load our serialized model from disk
        print("[INFO] loading model...")
        self.net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

    def process(self, image):
        image = imutils.resize(image, width=400)
        # grab the frame dimensions and convert it to a blob
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)),
            0.007843, (300, 300), 127.5)

        self.net.setInput(blob)
        return self.net.forward()
