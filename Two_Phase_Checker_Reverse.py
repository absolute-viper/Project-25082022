import csv

import cv2
import os
import face_recognition
import numpy as np

img_org = "F:/profile6.jpg"
profile_1 = cv2.imread('F:/profile6.jpg')
profile_1 = cv2.cvtColor(profile_1, cv2.COLOR_BGR2RGB)
encode_1 = face_recognition.face_encodings(profile_1)[0]
directory = 'DATA/'

f = open('trial_run1.csv', 'w', encoding='UTF8')
writer = csv.writer(f, lineterminator='\n')
header = ['Name']
writer.writerow(header)


def face_recog():
    a = np.array([])
    for filename in os.listdir(directory):
        try:
            profile_2 = cv2.imread(directory + filename)
            profile_2 = cv2.cvtColor(profile_2, cv2.COLOR_BGR2RGB)
            encode_2 = face_recognition.face_encodings(profile_2)[0]
            compare = face_recognition.compare_faces([encode_1], encode_2)

            if str(compare[0]) == 'True':
                accuracy = face_recognition.face_distance([encode_1], encode_2)
                if accuracy < 0.5:
                    a = np.append(a, filename)
                    val = [filename]
                    print(filename)
                    writer.writerow(val)

        except IndexError:
            print(filename + " Face not Visible")
    print("Size of the array:", len(a))
#     deep_face(a)
#
#
# flag = True
#
#
# def deep_face(refined_arr):
#     count = 0
#     for filename in refined_arr:
#         try:
#             result = DeepFace.verify(img_org, directory + filename, distance_metric="euclidean")
#             if result['verified'] == flag:
#                 img_profile = cv2.imread(directory + filename)
#                 count += 1
#                 cv2.imshow("Match_" + filename, img_profile)
#                 cv2.waitKey(5000)
#
#         except ValueError:
#             continue
#     print("Faces matched:", count)


face_recog()
