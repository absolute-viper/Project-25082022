from deepface import DeepFace
import tensorflow as tf
img_0 = "F:/profile0.jpg"
img_1 = "F:/profile1.jpg"
img_2 = "F:/profile2.jpg"
img_3 = "F:/profile3.jpg"
img_4 = "F:/profile6.jpg"
img_5 = "F:/profile7.jpg"
img_6 = "F:/profile8.jpg"
img_7 = "E:/Codes/Application/Project25082022/DATA/test_2.jpg"
img_8 = "E:/Codes/Application/Project25082022/DATA/test_photu.jpg"

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
with(tf.device('/GPU:0')):
    try:
        result = DeepFace.verify(img_7, img_8, model_name=models[8], detector_backend='mtcnn', distance_metric='euclidean')
        print(result)

    except ValueError:
        print("No Face found")
