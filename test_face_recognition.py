import face_recognition
import cv2

img_7 = "E:/Codes/Application/Project25082022/DATA/test_2.jpg"
img_8 = "E:/Codes/Application/Project25082022/DATA/test_photu.jpg"
profile_1 = cv2.imread('F:/TEST/profile0.jpg')
profile_1 = cv2.cvtColor(profile_1, cv2.COLOR_BGR2RGB)
encode_1 = face_recognition.face_encodings(profile_1)[0] 

try:
    profile_2 = cv2.imread(img_7)
    profile_2 = cv2.cvtColor(profile_2, cv2.COLOR_BGR2RGB)
    encode_2 = face_recognition.face_encodings(profile_2)[0]
    compare = face_recognition.compare_faces([encode_1], encode_2)

    if str(compare[0]) == 'True':
        distance = face_recognition.face_distance([encode_1], encode_2)
        print(distance)

except IndexError:
    print("Face not Visible")
