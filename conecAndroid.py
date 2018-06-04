# -*- coding: utf-8 -*-
import os
import cv2
from egine import decision
# cv2.namedWindow('Window11',cv2.WINDOW_NORMAL)
# os.system("adb shell /system/bin/screencap -p /sdcard/screenshot.png")
os.system("adb connect 127.0.0.1")
while(True):
    os.system("adb shell /system/bin/screencap -p /sdcard/screenshot.png")
    os.system("adb pull /sdcard/screenshot.png E:/tempdic/screenshot.png")
    cv2.waitKey(2)
    img=cv2.imread("E:/tempdic/screenshot.png")
    # cv2.imshow('Window11',img)
    # cv2.waitKey(10)
    if decision(img,thresh=1):
        # break
        os.system("adb shell input tap 500 500")
# for i in range(100):
#     os.system("adb shell /system/bin/screencap -p /sdcard/screenshot.png")
#     os.system("adb pull /sdcard/screenshot.png E:/tempdic/"+str(i)+".png")
#     print i
