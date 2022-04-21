import cv2 as cv
import face_recognition
import numpy as np
import os

from numpy import true_divide

path = 'faces'
images = []
names = []
my_list = os.listdir(path)

for instance in my_list:
    current_image = cv.imread(f'{path}/{instance}')
    images.append(current_image)
    names.append(os.path.splitext(instance)[0])


def encoding(images):
    encodings = []
    for img in images:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        encoded = face_recognition.face_encodings(img)[0]
        encodings.append(encoded)
    return encodings

finished_encoding = encoding(images)
print(len(finished_encoding))

capture = cv.VideoCapture(0)

while True:
    success, web_cam = capture.read()

    img_down = cv.resize(web_cam, (0,0), None, 0.25, 0.25)
    img_down = cv.cvtColor(img_down, cv.COLOR_BGR2RGB)

    faces_current = face_recognition.face_locations(img_down)
    encode_current = face_recognition.face_encodings(img_down, faces_current)

    for encode_face, face_location in zip(encode_current, faces_current):
        matches = face_recognition.compare_faces(finished_encoding, encode_face)

        face_distance = face_recognition.face_distance(finished_encoding, encode_face)
        print(face_distance)

        match_id = np.argmin(face_distance)

        if matches[match_id]:
            name = names[match_id].upper()
            print(name)
            y1,x2,y2,x1 = face_location 
            y1,x2,y2,x1 = y1*4, x2*4, y2*4, x1*4
            cv.rectangle(web_cam, (x1,y1), (x2,y2), (0,255,0), 2)
            cv.putText(web_cam, name, (x1+6, y2-6), cv.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1.5)


        cv.imshow('webcam', web_cam)
        cv.waitKey(1)