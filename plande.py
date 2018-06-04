# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 16:34:30 2017

@author: zpy
"""

import numpy as np
import cv2
import math



lower_H=0
lower_S=0
lower_V=0
upper_H=179
upper_S=255
upper_V=255
lower_color=np.array([61,69,184])
upper_color=np.array([114,255,255])
cap=cv2.VideoCapture(0)
def barReactLH(value):
    global lower_H,lower_S,lower_V,lower_color
    global upper_H,upper_S,upper_V,upper_color
    lower_H=value
    lower_color=np.array([lower_H,lower_S,lower_V])
def barReactLS(value):
    global lower_H,lower_S,lower_V,lower_color
    global upper_H,upper_S,upper_V,upper_color
    lower_S=value
    lower_color=np.array([lower_H,lower_S,lower_V])
def barReactLV(value):
    global lower_H,lower_S,lower_V,lower_color
    global upper_H,upper_S,upper_V,upper_color
    lower_V=value
    lower_color=np.array([lower_H,lower_S,lower_V])


def barReactUH(value):
    global lower_H,lower_S,lower_V,lower_color
    global upper_H,upper_S,upper_V,upper_color
    upper_H=value
    upper_color=np.array([upper_H,upper_S,upper_V])


def barReactUS(value):
    global lower_H,lower_S,lower_V,lower_color
    global upper_H,upper_S,upper_V,upper_color
    upper_S=value
    upper_color=np.array([upper_H,upper_S,upper_V])

def barReactUV(value):
    global lower_H,lower_S,lower_V,lower_color
    global upper_H,upper_S,upper_V,upper_color
    upper_V=value
    upper_color=np.array([upper_H,upper_S,upper_V])

cv2.namedWindow('MyWindow4',cv2.WINDOW_NORMAL)
cv2.createTrackbar('Lower_H','MyWindow4',0,179,barReactLH)
cv2.createTrackbar('Lower_S','MyWindow4',50,255,barReactLS)
cv2.createTrackbar('Lower_V','MyWindow4',50,255,barReactLV)
cv2.createTrackbar('Upper_H','MyWindow4',179,179,barReactUH)
cv2.createTrackbar('Upper_S','MyWindow4',255,255,barReactUS)
cv2.createTrackbar('Upper_V','MyWindow4',255,255,barReactUV)





while True:
    img_ori = cv2.imread('E:/tempdict/0.jpg')

    frame = img_ori
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,lower_color,upper_color)
    print mask.shape
    cv2.rectangle(mask, (0,0), (720,60), 0, -1)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, 2, (255, 0, 255), -1)
    print len(contours)
    aimImg=cv2.bitwise_and(frame,frame,mask=mask)
    cv2.imshow('MyWindow',frame)
    cv2.imshow('MyWindow2',mask)
    cv2.imshow('MyWindow3',aimImg)
    cv2.imshow('MyWindow4',aimImg)
    inputKey=cv2.waitKey(0)
    if inputKey==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()