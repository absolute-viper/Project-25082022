import csv
import os
import deepface.DeepFace

f = open('result_1.csv', 'w', encoding='UTF8')
writer = csv.writer(f)
header = ['Name', 'Age', 'Gender', 'Race', 'Emotion']
writer.writerow(header)

directory = 'DATA/'
for filename in os.listdir(directory):
    try:
        profile = directory + filename
        obj = deepface.DeepFace.analyze(profile, actions=('age', 'gender', 'race', 'emotion'), models="Dlib")
        row = [filename, obj['age'], obj['gender'], obj['dominant_race'], obj['dominant_emotion']]
        writer.writerow(row)
        # print('Age:', obj['age'], ' Gender:', obj['gender'], ' Race:', obj['dominant_race'], ' Emotion:',
        #       obj['dominant_emotion'])
    except ValueError:
        row = [filename, " Face not found"]
        writer.writerow(row)
f.close()
