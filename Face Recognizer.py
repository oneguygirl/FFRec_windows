import numpy as np
import cv2
import sqlite3

fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (255, 0, 0)
# for record not found case
fontColor1 = (0, 0, 255)

cap = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# img = cv2.imread('images/download.jpeg', cv2.IMREAD_COLOR)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
# path= 'dataSet'

def getProfile(Id):
    conn=sqlite3.connect("Faces1.0.db")
    cmd="SELECT * FROM People WHERE ID="+str(Id)
    cursor=conn.execute(cmd)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile


#font = cv2.InitFont(cv2.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1)
while True:
    ret, img = cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:

        Idu, conf = recognizer.predict(gray[y:y+h, x:x+w])
        cv2.putText(img,str(conf), (x, y + h + 120 ), fontFace, fontScale, fontColor1)
        if (conf<80):
            cv2.rectangle(img, (x, y), (x + w, y + h), (225, 0, 0), 2)
            profile = getProfile(Idu)
            if(profile!=None):
                cv2.putText(img, str(profile[1]), (x,y+h+30),fontFace, fontScale, fontColor)
                cv2.putText(img, str(profile[2]), (x, y + h + 60), fontFace, fontScale, fontColor)
                cv2.putText(img, str(profile[3]), (x, y + h + 90), fontFace, fontScale, fontColor)
                #cv2.putText(img, str(profile[4]), (x, y + h + 120), fontFace, fontScale, fontColor)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 225), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(img,"Not Found", (x, y + h + 30 ), fontFace, fontScale, fontColor1)
    cv2.imshow('frame', img)
    if ord('q') == cv2.waitKey(1):
        break
cv2.destroyAllWindows()