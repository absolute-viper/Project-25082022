import os
import time
import numpy as np
import cv2
import face_recognition
import tensorflow as tf
from deepface import DeepFace

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
start_time = time.time()
with tf.device('/GPU:0'):
    img_org = 'F:/TEST/profile6.jpg'
    profile_1 = cv2.imread('F:/TEST/profile6.jpg')
    profile_1 = cv2.cvtColor(profile_1, cv2.COLOR_BGR2RGB)
    encode_1 = face_recognition.face_encodings(profile_1)[0]
    directory = 'E:/Codes/Application/Project25082022/DATA/'
    count = 0
    a = np.array([])
    for filename in os.listdir(directory):
        try:
            profile_2 = cv2.imread(directory + filename)
            profile_2 = cv2.cvtColor(profile_2, cv2.COLOR_BGR2RGB)
            encode_2 = face_recognition.face_encodings(profile_2)[0]
            compare = face_recognition.compare_faces([encode_1], encode_2)
            result = DeepFace.verify(img_org, directory + filename, enforce_detection=False, prog_bar=False)
        except IndexError or ValueError:
            continue

        if str(compare[0]) == 'True' and result['verified'] is True:
            accuracy = face_recognition.face_distance([encode_1], encode_2)
            if accuracy < 0.5:
                print("Match:", filename)
                count += 1
                a = np.append(a, filename)
    print("Match Count:", count)
    print("Matched accounts:", a)
    print("Time Elapsed:", time.time() - start_time)

