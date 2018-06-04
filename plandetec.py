import numpy as np
import math
import cv2
img_ori=cv2.imread('E:/tempdict/2.jpg')
img_ori_hsv = cv2.cvtColor(img_ori,cv2.COLOR_BGR2HSV)
lower_color=np.array([61,69,184])
upper_color=np.array([114,255,255])
mask=cv2.inRange(img_ori_hsv,lower_color,upper_color)
cv2.rectangle(mask, (0, 0), (60, 720), 0, -1)
kernel = np.ones((3, 3), np.uint8)
mask = cv2.erode(mask, kernel, iterations=1)
mask = cv2.dilate(mask, kernel, iterations=1)
contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_use=[]
contours_inf=[]
for cnt in contours:
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    # if (area>=20 and area<=50 or area>=500 and area<=560 or area>=60 and area<=90)\
    #         and (perimeter>=15.8 and perimeter<=32.7 or perimeter>=100.2 and perimeter<=120.3 or perimeter>=25.4 and perimeter<=40.4):
    if True:
        contours_use.append([(cx,cy),area,perimeter])
    print cx,cy,area,perimeter

contours_use=sorted(contours_use,key=lambda x:x[-2])
print contours_use
cv2.drawContours(img_ori, contours, 0, (255, 0, 0), -1)
cv2.drawContours(img_ori, contours, 1, (0,255, 0), -1)
cv2.drawContours(img_ori, contours, 2, (0, 0, 255), -1)
cv2.line(img_ori,contours_use[0][0],contours_use[1][0],(255,255,255),2)
cv2.namedWindow('myWindow1',cv2.WINDOW_NORMAL)
cv2.imshow('myWindow1',img_ori)
cv2.namedWindow('myWindow11',cv2.WINDOW_AUTOSIZE)
cv2.imshow('myWindow11',img_ori)
# cv2.namedWindow('myWindow2',cv2.WINDOW_NORMAL)
# cv2.imshow('myWindow2',img_ori)
# cv2.namedWindow('myWindow22',cv2.WINDOW_AUTOSIZE)
# cv2.imshow('myWindow22',img_ori)
# cv2.namedWindow('myWindow3',cv2.WINDOW_NORMAL)
# cv2.imshow('myWindow3',img_ori)
# cv2.namedWindow('myWindow33',cv2.WINDOW_AUTOSIZE)
# cv2.imshow('myWindow33',img_ori)
cv2.waitKey(0)
cv2.destroyAllWindows()