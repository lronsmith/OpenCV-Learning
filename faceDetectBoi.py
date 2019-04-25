import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.rectangle(gray,(x,y),(x+w,y+h),(0,0,0),1)
        roi_gray = gray[y-50:y+h+50, x-50:x+w+50]
        roi_color = img[y-50:y+h+50, x-50:x+w+50]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)
            cv2.rectangle(roi_gray, (ex, ey), (ex+ew, ey+eh), (255, 255, 255), 1)
            cv2.imshow('Face Color', cv2.flip( roi_color, 1 ))         
            cv2.imshow('Face Gray', cv2.flip( roi_gray, 1 ))

    cv2.imshow('Full Image Color', cv2.flip( img, 1 ))
    cv2.imshow('Full Image Gray', cv2.flip( gray, 1 ))
    k = cv2.waitKey(30 & 0xff)
    if k == 27:
        break

cv2.destroyAllWindows()
