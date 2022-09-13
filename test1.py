from deepface import DeepFace
import os
import cv2
import numpy

img_org = "F:/profile5.jpg"
models = [
    "VGG-Face",
    "Facenet",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "ArcFace",
    "SFace",
    "Facenet512",
    "Dlib",
    "Human-beings"
]
flag = True
threashold = numpy.float(0.5)

directory = 'DATA/'
for filename in os.listdir(directory):
    try:
        result = DeepFace.verify(img_org, directory + filename)
        if result['verified'] == flag and result['distance'] > threashold:
            img = cv2.imread(directory + filename, cv2.IMREAD_COLOR)
            cv2.imshow(filename, img)
            cv2.waitKey(5000)
    except ValueError:
        continue
