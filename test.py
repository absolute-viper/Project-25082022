from deepface import DeepFace
import os
import cv2
import numpy
profile = 'DATA/rupeshtailor209.jpg'
img_org1 = "F:/profile1.jpg"
img_org2 = "F:/profile4.jpg"
img_org3 = "F:/profile5.jpg"
img_org4 = "F:/profile6.jpg"
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

result = DeepFace.verify(img_org2, img_org3)
print(result)
description = DeepFace.analyze(profile, actions=('age', 'gender', 'race'))
print(description)
description1 = DeepFace.analyze(img_org1, actions=('age', 'gender', 'race'))
print(description1)
description2 = DeepFace.analyze(img_org2, actions=('age', 'gender', 'race'))
print(description2)
description3 = DeepFace.analyze(img_org3, actions=('age', 'gender', 'race'))
print(description3)
description4 = DeepFace.analyze(img_org4, actions=('age', 'gender', 'race'))
print(description4)

