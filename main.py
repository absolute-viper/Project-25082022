import cv2
import os
filename = 'pic2.jpg'
img = cv2.imread(filename)  # Path of an image
faceCascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
faces = faceCascade.detectMultiScale(img, 1.1, 4)

directory = os.getcwd() + r''

try:
    os.mkdir(directory)
except FileExistsError as fee:
    print('Exception Occurred', fee)
os.chdir(directory)
i = 1
for (x, y, w, h) in faces:
    FaceImg = img[y:y + h, x:x + w]
    # To save an image on disk
    filename = 'Face' + str(i) + '.jpg'
    cv2.imwrite(filename, FaceImg)
    i += 1
