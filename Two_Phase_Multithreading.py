import csv
import cv2
import os
import face_recognition
from deepface import DeepFace
import tensorflow as tf
import numpy as np
import time
import asyncio

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
    img_org = 'F:/TEST/profile0.jpg'
    profile_1 = cv2.imread('F:/TEST/profile0.jpg')
    profile_1 = cv2.cvtColor(profile_1, cv2.COLOR_BGR2RGB)
    encode_1 = face_recognition.face_encodings(profile_1)[0]
    directory = 'E:/Codes/Application/Project25082022/DATA/'


    async def face_recog():
        for filename in os.listdir(directory):
            try:
                profile_2 = cv2.imread(directory + filename)
                profile_2 = cv2.cvtColor(profile_2, cv2.COLOR_BGR2RGB)
                encode_2 = face_recognition.face_encodings(profile_2)[0]
                compare = face_recognition.compare_faces([encode_1], encode_2)

                if str(compare[0]) == 'True':
                    accuracy = face_recognition.face_distance([encode_1], encode_2)
                    if accuracy < 0.5:
                        print("Face_Recognition_Match:", filename)

            except IndexError:
                continue


    flag = True
    f = open('trial_run1.csv', 'w', encoding='UTF8')
    writer = csv.writer(f, lineterminator='\n')
    header = ['Name']
    writer.writerow(header)


    async def deep_face():
        a = np.array([])
        for filename in os.listdir(directory):
            try:
                result = DeepFace.verify(img_org, directory + filename, prog_bar=False)
                if result['verified'] == flag:
                    a = np.append(a, filename)
                    print("Deepface_Match:", filename)
                    writer.writerow([filename])
            except ValueError:
                continue

    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.gather(deep_face(), face_recog()))
    loop.close()
    print("Time Elapsed:", time.time() - start_time)
