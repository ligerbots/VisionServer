# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 16:12:21 2017

@author: Mark Nazzaro
"""

import cv2
import numpy as np
import time

def nothing(q):
    pass

cv2.namedWindow("Window")
cv2.createTrackbar("LowerH", "Window", 110, 179, nothing)
cv2.createTrackbar("LowerS", "Window", 50, 255, nothing)
cv2.createTrackbar("LowerV", "Window", 50, 255, nothing)
cv2.createTrackbar("UpperH", "Window", 130, 179, nothing)
cv2.createTrackbar("UpperS", "Window", 255, 255, nothing)
cv2.createTrackbar("UpperV", "Window", 255, 255, nothing)
cv2.createTrackbar("ColorSpace", "Window", 0, 1, nothing)
cv2.createTrackbar("Erode", "Window", 0, 1, nothing)
cv2.createTrackbar("Dilate", "Window", 0, 1, nothing)
vid = cv2.VideoCapture(0)

lowerBlue = np.array([110,50,50])
upperBlue = np.array([130,255,255])

kernel = np.ones((5,5), np.uint8)

while True:
    
    a = cv2.getTrackbarPos("LowerH", "Window")
    b = cv2.getTrackbarPos("LowerS", "Window")
    c = cv2.getTrackbarPos("LowerV", "Window")
    d = cv2.getTrackbarPos("UpperH", "Window")
    e = cv2.getTrackbarPos("UpperS", "Window")
    f = cv2.getTrackbarPos("UpperV", "Window")
    lowerBlue = np.array([a, b, c])
    upperBlue = np.array([d, e, f])
    _, frame = vid.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    mask = cv2.inRange(hsv, lowerBlue, upperBlue)
    if cv2.getTrackbarPos("Erode", "Window"):
        mask = cv2.erode(mask, kernel, iterations=1)
    if cv2.getTrackbarPos("Dilate", "Window"):
        mask = cv2.dilate(mask, kernel, iterations=1)
    ret, thresh = cv2.threshold(mask, 127, 255, 0)
    contours,hierarchy = cv2.findContours(thresh, 1, 2)
    if len(contours) > 0:
        areas = [cv2.contourArea(g) for g in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]
        x,y,w,h = cv2.boundingRect(cnt)
        if cv2.getTrackbarPos("ColorSpace", "Window"):
            cv2.rectangle(hsv,(x,y),(x+w,y+h),(100,255,255), 5)
        else:
            cv2.rectangle(mask,(x,y),(x+w,y+h),(100,255,255), 5)
    if cv2.getTrackbarPos("ColorSpace", "Window"):
            cv2.imshow("Window", hsv)
    else:
        cv2.imshow("Window", mask)
    q = cv2.waitKey(5) & 0xFF
    if q == ord('q'):
        break
    time.sleep(0.05)
    
vid.release()
cv2.destroyAllWindows()  

