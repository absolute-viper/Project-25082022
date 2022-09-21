from deepface import DeepFace
import os
import cv2
import numpy
import tensorflow as tf

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
embedding = DeepFace.represent(img_org)
directory = 'DATA/'
with tf.device('/GPU:0'):
    for filename in os.listdir(directory):
        try:
            result = DeepFace.verify(embedding, directory + filename)

            if result['verified'] == flag:
                img = cv2.imread([directory + filename, cv2.IMREAD_COLOR])
                cv2.imshow(filename, img)
                cv2.waitKey(5000)
        except ValueError:
            continue
