from PIL import ImageGrab
import numpy as np
import time
import cv2
import os
from egine import decision
import win32api
import win32con
import win32gui
import ctypes

def click1(x,y):
    ctypes.windll.user32.SetCursorPos(x, y)

    ctypes.windll.user32.mouse_event(2, 0, 0, 0, 0)

    ctypes.windll.user32.mouse_event(4, 0, 0, 0, 0)


bbox = (0, 47, 887, 544)
im = ImageGrab.grab(bbox)
img_convert_ndarray = np.array(im)
print img_convert_ndarray.shape
img_ori=np.zeros(img_convert_ndarray.shape,np.uint8)
# cv2.imshow('Window11', img_ori)
# os.system("adb connect 127.0.0.1")
# time.sleep(2)
# print img_ori.shape
th=13
just_click=False
a=0.2
while True:
    bbox = (0, 47, 887, 544)
    im = ImageGrab.grab(bbox)
    img_convert_ndarray = np.array(im)
    img_ori[:,:,0]=img_convert_ndarray[:,:,2]
    img_ori[:,:,1]=img_convert_ndarray[:,:,1]
    img_ori[:,:,2]=img_convert_ndarray[:,:,0]

    img = img_ori
    # cv2.imwrite('E:\\tempdict\\'+str(i)+'.jpg',img)
    # cv2.imwrite('E:/tempdict/2.jpg', img)
    # img_ori = cv2.imread('E:/tempdict/2.jpg')
    if decision(img_ori, thresh=th,alph=a):
        # break
        # pass
        print 'c'
        click1(650, 400)
        just_click=True
        time.sleep(0.3)
        # time.sleep(0.05)




    # if just_click:
    cv2.imshow('Window211',img_ori)
    cv2.waitKey(1)

    #     just_click=False

    # im.save('as.png')