import csv

import cv2
import os
import face_recognition
from deepface import DeepFace
import numpy as np

img_org = "F:/profile2.jpg"
profile_1 = cv2.imread('F:/profile2.jpg')
profile_1 = cv2.cvtColor(profile_1, cv2.COLOR_BGR2RGB)
encode_1 = face_recognition.face_encodings(profile_1)[0]
directory = 'DATA/'


def face_recog(refined_arr):
    count = 0
    for filename in refined_arr:
        try:
            profile_2 = cv2.imread(directory + filename)
            profile_2 = cv2.cvtColor(profile_2, cv2.COLOR_BGR2RGB)
            encode_2 = face_recognition.face_encodings(profile_2)[0]
            compare = face_recognition.compare_faces([encode_1], encode_2)

            if str(compare[0]) == 'True':
                accuracy = face_recognition.face_distance([encode_1], encode_2)
                if accuracy < 0.5:
                    print(filename + " Possible Match, Accuracy: " + str(accuracy))
                    count += 1
                    cv2.imshow("Match_" + filename, profile_2)
                    cv2.waitKey(5000)

        except IndexError:
            print(filename + " Face not Visible")
    print("Faces matched:", count)


flag = True

f = open('trial_run1.csv', 'w', encoding='UTF8')
writer = csv.writer(f, lineterminator='\n')
header = ['Name']
writer.writerow(header)

def deep_face():
    a = np.array([])
    for filename in os.listdir(directory):
        try:
            result = DeepFace.verify(img_org, directory + filename, distance_metric="euclidean")
            if result['verified'] == flag:
                a = np.append(a, filename)
                print(filename)
                writer.writerow(a[-1])
        except ValueError:
            continue
    print("Size of array:", len(a))
    face_recog(a)


deep_face()
