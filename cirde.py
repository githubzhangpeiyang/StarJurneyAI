# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 16:34:30 2017

@author: zpy
"""

import numpy as np
import cv2
import math
import copy
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from PIL import ImageGrab

lower_color=np.array([76,56,130])
upper_color=np.array([115,106,180])
lower_H=lower_color[0]
lower_S=lower_color[1]
lower_V=lower_color[2]
upper_H=upper_color[0]
upper_S=upper_color[1]
upper_V=upper_color[2]
circle_thresh=30

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
def barReactCT(value):
    global circle_thresh
    circle_thresh=value
    # upper_color=np.array([upper_H,upper_S,upper_V])
cv2.namedWindow('MyWindow4',cv2.WINDOW_NORMAL)
cv2.namedWindow('MyWindown',cv2.WINDOW_NORMAL)
cv2.namedWindow('MyWindowf2',cv2.WINDOW_NORMAL)
cv2.createTrackbar('Lower_H','MyWindow4',lower_H,179,barReactLH)
cv2.createTrackbar('Lower_S','MyWindow4',lower_S,255,barReactLS)
cv2.createTrackbar('Lower_V','MyWindow4',lower_V,255,barReactLV)
cv2.createTrackbar('Upper_H','MyWindow4',upper_H,179,barReactUH)
cv2.createTrackbar('Upper_S','MyWindow4',upper_S,255,barReactUS)
cv2.createTrackbar('Upper_V','MyWindow4',upper_V,255,barReactUV)
cv2.createTrackbar('CircleThreshold','MyWindow4',circle_thresh,60,barReactCT)

bbox = (0, 47, 887, 544)
im = ImageGrab.grab(bbox)
img_convert_ndarray = np.array(im)
img_ori=np.zeros(img_convert_ndarray.shape,np.uint8)
cv2.imshow('Window11', img_ori)
while(True):
    bbox = (0, 47, 887, 544)
    im = ImageGrab.grab(bbox)
    img_convert_ndarray = np.array(im)
    img_ori[:,:,0]=img_convert_ndarray[:,:,2]
    img_ori[:,:,1]=img_convert_ndarray[:,:,1]
    img_ori[:,:,2]=img_convert_ndarray[:,:,0]

    img = img_ori

    frame = img_ori
    frame2 = copy.deepcopy(frame)
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,lower_color,upper_color)
    kernel = np.ones((3, 3), np.uint8)
    kernel1=np.array([[0,1,0],[1,1,1],[0,1,0]],np.uint8)
    kernelx=np.array([[1,0,1],[0,1,0],[1,0,1]],np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    mask = cv2.erode(mask, kernel, iterations=1)
    rec_roi=np.zeros(mask.shape,np.uint8)
    cv2.rectangle(rec_roi, (0, 260), (720, 480), 255, -1)
    # mask=cv2.bitwise_and(rec_roi,mask, mask=mask)
    # print mask.shape

    # cv2.rectangle(mask, (0,0), (720,60), 0, -1)
    circles = cv2.HoughCircles(mask, 3, 1, 20,circles=None,
                               param1=50, param2=circle_thresh, minRadius=0, maxRadius=0)
    circle_centers=[]
    if circles is not None:
        # print circles
        # circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(frame2, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(frame2, (i[0], i[1]), 2, (0, 0, 255), 5)
            circle_centers.append([i[0], i[1]])
    if circle_centers!=[]:
        circle_centers=np.array(circle_centers,np.double)
        print circle_centers.shape
        y_pred = DBSCAN(eps=40, min_samples=1).fit_predict(circle_centers)
        plt.scatter(circle_centers[:, 0], circle_centers[:, 1], c=y_pred)
        center_sta=[[] for i in range(max(y_pred)+1)]

        for indxe,class_type in enumerate(y_pred):
            # print circle_centers[indxe]
            center_sta[class_type].append(circle_centers[indxe])
        # center_sta = np.array(center_sta, np.double)
        for i in range(max(y_pred)+1):
            cx,cy=np.uint16(np.around(np.mean(center_sta[i],axis=0)))
            cv2.circle(frame, (cx,cy), 2, (0, 0, 255), 5)
            cv2.circle(mask, (cx,cy), 2, (122), 5)
        print max(y_pred)+1

    # kernel = np.ones((3, 3), np.uint8)
    # mask = cv2.erode(mask, kernel, iterations=1)
    # mask = cv2.dilate(mask, kernel, iterations=1)
    # contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(frame, contours, 2, (255, 0, 255), -1)
    # print len(contours)
    aimImg=cv2.bitwise_and(frame,frame,mask=mask)
    cv2.imshow('MyWindow',frame)
    cv2.imshow('MyWindow2',mask)
    # cv2.imshow('MyWindow3',aimImg)
    # cv2.imshow('MyWindow5', rec_roi)
    # cv2.imshow('MyWindow4',aimImg)
    cv2.imshow('MyWindown',frame)
    cv2.imshow('MyWindowf2qwe',frame2)
    plt.show()
    inputKey=cv2.waitKey(1)
    if inputKey==ord('q'):
        break
cv2.destroyAllWindows()