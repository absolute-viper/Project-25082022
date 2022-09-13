import os
import cv2
import face_recognition

profile_1 = cv2.imread('DATA/rupeshtailor209.jpg')
profile_1 = cv2.cvtColor(profile_1, cv2.COLOR_BGR2RGB)
encode_1 = face_recognition.face_encodings(profile_1)[0]

flag = True
directory = 'DATA/'
for filename in os.listdir(directory):
    try:
        profile_2 = cv2.imread(directory+filename)
        profile_2 = cv2.cvtColor(profile_2, cv2.COLOR_BGR2RGB)
        encode_2 = face_recognition.face_encodings(profile_2)[0]
        compare = face_recognition.compare_faces([encode_1], encode_2)

        if str(compare[0]) == 'True':
            accuracy = face_recognition.face_distance([encode_1], encode_2)
            print(filename + " Possible Match, Accuracy: " + str(accuracy))
            cv2.imshow("Match_" + filename, profile_2)
            cv2.waitKey(5000)

    except IndexError:
        print(filename + " Face not Visible")
