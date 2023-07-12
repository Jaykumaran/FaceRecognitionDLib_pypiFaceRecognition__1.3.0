import csv

import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

from PIL import ImageGrab



path = 'Training_images'
images=[]
classNames=[]
myList= os.listdir(path)

print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []


    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    now = datetime.now()
    dtString = now.strftime('%Y-%m-%d %H:%M:%S')

    hour = now.hour

    with open(f'Attendance_{hour}.csv', 'a', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow([name, dtString])

### FOR CAPTURING SCREEN RATHER THAN WEBCAM
def captureScreen(bbox=(300,300,690+300,530+300)):
    capScr = np.array(ImageGrab.grab(bbox))
    capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
    return capScr


encodeListKnown = findEncodings(images)
print('Encoding completed')

cap = cv2.VideoCapture('C:/Users/jaikr/PycharmProjects/CVFaceKrish/Video_klhi.mp4') #inbuilt webcam 0

while True:
    success, img = cap.read()
    imgS = captureScreen()
    # imgS = cv2.resize(img,(0,0), None, 0.25, 0.25)
    # imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    facesCurFrame = face_recognition.face_locations(imgS)  #stores the face encodings of the faces in the current frame
    encodesCurFrame = face_recognition.face_encodings(imgS) #stores the face encodings of the faces in the current frame

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()

            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0),2)
            cv2.rectangle(img, (x1,y2 - 35), (x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255),2)
            markAttendance(name)

        cv2.imshow('Webcam', img)
        # cv2.waitKey(1)
            