# importing libraries
import cv2
import numpy as np
import face_recognition as face_rec
import os
import pyttsx3 as textSpeech
from datetime import datetime

engine = textSpeech.init()

# function
def resize(img, size) :
    width = int(img.shape[1]*size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation= cv2.INTER_AREA)

#depicting path of politicians image
path = 'politicians image'
politiciansImg = []
politiciansName = []
myList = os.listdir(path)
print(myList)
for cl in myList :
    curimg = cv2.imread(f'{path}/{cl}') # politicians image/Amit_Shah.jpg
    politiciansImg.append(curimg)
    politiciansName.append(os.path.splitext(cl)[0]) # Here 0 depicts the first part of image name
print(politiciansName)

def findEncoding(images) :
    imgEncodings = []
    for img in images :
        img = resize(img, 0.50)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeimg = face_rec.face_encodings(img)[0]
        imgEncodings.append(encodeimg)
    return imgEncodings
def MarkAttendence(name):
    with open('attendence.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList :
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            timestr = now.strftime('%H:%M')
            f.writelines(f'\n{name}, {timestr}')
            statment = str('Welcome to Lok Sabha' + name)
            engine.say(statment)
            engine.runAndWait()

EncodeList = findEncoding(politiciansImg)

vid = cv2.VideoCapture(0) # 0 is to tell the source of web cam
while True :
    success, frame = vid.read()
    Smaller_frames = cv2.resize(frame, (0,0), None, 0.25, 0.25)

    facesInframe = face_rec.face_locations(Smaller_frames)
    encodeFacesInframe = face_rec.face_encodings(Smaller_frames, facesInframe)

    for encodeFace, faceloc in zip(encodeFacesInframe, facesInframe) :
        matches = face_rec.compare_faces(EncodeList, encodeFace)
        facedis = face_rec.face_distance(EncodeList, encodeFace)
        print(facedis)
        matchIndex = np.argmin(facedis)

        if matches[matchIndex] :
            name = politiciansName[matchIndex].upper()
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.rectangle(frame, (x1, y2-25), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            MarkAttendence(name)
    cv2.imshow('video',frame)
    cv2.waitKey(1)