import numpy as np
import math
import cv2
import copy
from sklearn.cluster import DBSCAN
def get_plane_loc(img,alph=0.3):
    img_use=copy.deepcopy(img)
    img_ori_hsv = cv2.cvtColor(img_use, cv2.COLOR_BGR2HSV)
    lower_color = np.array([61, 69, 184])
    upper_color = np.array([114, 255, 255])
    mask = cv2.inRange(img_ori_hsv, lower_color, upper_color)
    cv2.rectangle(mask, (0, 0), (60, 720), 0, -1)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_use = []
    for cnt in contours:
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        # if (area >= 20 and area <= 50 or area >= 500 and area <= 560 or area >= 60 and area <= 90) \
        #         and (
        #                         perimeter >= 15.8 and perimeter <= 32.7 or perimeter >= 100.2 and perimeter <= 120.3 or perimeter >= 25.4 and perimeter <= 40.4):
        if True:
            contours_use.append([(cx, cy), area, perimeter])
        # print cx, cy, area, perimeter
    # print len(contours_use)
    contours_use = sorted(contours_use, key=lambda x: x[-2])
    if len(contours_use)<3:
        return False
    # cv2.line(img_ori, contours_use[1][0], contours_use[2][0], (255, 255, 255), 2)
    x=contours_use[1][0][0]+(contours_use[0][0][0]-contours_use[1][0][0])*alph
    y = contours_use[1][0][1]+(contours_use[0][0][1] - contours_use[1][0][1])*alph
    # print x,y
    return int(round(x)),int(round(y))
def get_net_loc(img_in):
    lower_color = np.array([76,56,130])
    upper_color = np.array([115,106,180])
    circle_thresh = 30

    # lower_color = np.array([99, 52, 161])
    # upper_color = np.array([117, 95, 196])

    frame = copy.deepcopy(img_in)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    kernel = np.ones((3, 3), np.uint8)
    kernel1 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    kernelx = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    mask = cv2.erode(mask, kernel, iterations=1)
    # mask = cv2.dilate(mask, kernel, iterations=1)
    cv2.imshow('WindowCmask',mask)
    # mask = cv2.erode(mask, kernel, iterations=1)
    # mask=cv2.bitwise_and(rec_roi,mask, mask=mask)
    # print mask.shape

    # cv2.rectangle(mask, (0,0), (720,60), 0, -1)
    circles = cv2.HoughCircles(mask, 3, 1, 20, circles=None,
                               param1=50, param2=circle_thresh, minRadius=0, maxRadius=0)
    circle_centers = []
    # print 'num:-;',circles
    if circles is not None:
        # print circles
        # circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            circle_centers.append([i[0], i[1]])
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 5)
        cv2.imshow('Window1388', frame)
    # cv2.imshow('Window138',frame)
    if circle_centers!=[]:
        circle_centers = np.array(circle_centers, np.double)
        # print 'num:-2;',circle_centers
        y_pred = DBSCAN(eps=40, min_samples=1).fit_predict(circle_centers)
        # plt.scatter(circle_centers[:, 0], circle_centers[:, 1], c=y_pred)
        center_sta = [[] for i in range(max(y_pred) + 1)]

        for indxe, class_type in enumerate(y_pred):
            # print circle_centers[indxe]
            center_sta[class_type].append(circle_centers[indxe])

        # center_sta = np.array(center_sta, np.double)
        result=[]
        for i in range(max(y_pred) + 1):
            cx, cy = np.uint16(np.around(np.mean(center_sta[i], axis=0)))
            result.append([cx, cy])
        result=sorted(result,key=lambda x:x[0],reverse=True)
        # print 'circle_num:',max(y_pred) + 1
        return result
    else:
        return False
def decision(img,thresh,alph):
    img_ori = copy.deepcopy(img)
    plan_center=get_plane_loc(img_ori, alph=alph)
    if plan_center==False:
        return False
    x, y =plan_center
    x = long(x)
    y = long(y)
    # cv2.circle(img_ori,(x,y),1,(0,0,255),2)
    # cv2.imshow('Window12',img_ori)
    centers = get_net_loc(img_ori)
    if centers==False:
        return False
    cv2.circle(img_ori, (x, y), 2, (0, 0, 255), 2)
    candi_index = None
    for index in range(len(centers) - 1):
        if x < centers[index][0] and x > centers[index + 1][0]:
            candi_index = index
            break
    if candi_index is None:
        return False
    for index, center in enumerate(centers):
        cx, cy = center
        # if index == candi_index:
        #     cv2.circle(img_ori, (cx, cy), 2, (0, 0, 255), 5)
        # else:
        #     cv2.circle(img_ori, (cx, cy), 2, (0, 255, 255), 5)
    x1, y1 = centers[candi_index]
    x2, y2 = centers[candi_index + 1]
    x1 = long(x1)
    y1 = long(y1)
    x2 = long(x2)
    y2 = long(y2)
    distence = 10000
    if x1 == x2:
        distence = math.fabs(x1 - x)
    elif y1 == y2:
        distence = math.fabs(y1 - y)
    else:
        A = y2 - y1
        B = (x1 - x2)
        C = x2 * y1 - x1 * y2
        distence = math.fabs(A * x + B * y + C) * 1.0 / math.sqrt(A ** 2 + B ** 2)
    # cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 5)
    cv2.line(img_ori,(x1,y1),(x2,y2),(255,255,255),1)
    cv2.imshow('Window138',img_ori)
    print 'dist:',distence
    if distence<=thresh:
        return True
    else:
        return False
# img_ori=cv2.imread('E:/tempdic/screenshot.png')
# # print decision(img_ori,6)
# #
# #
# #
# x, y = get_plane_loc(img_ori, alph=0.2)
# x = long(x)
# y = long(y)
# centers = get_net_loc(img_ori)
# cv2.circle(img_ori, (x, y), 2, (0, 0, 255), 2)
# candi_index = 0
# for index in range(len(centers) - 1):
#     if y < centers[index][1] and y > centers[index + 1][1]:
#         candi_index = index
#         break
# for index, center in enumerate(centers):
#     cx, cy = center
#     if index == candi_index:
#         cv2.circle(img_ori, (cx, cy), 2, (0, 0, 255), 5)
#     else:
#         cv2.circle(img_ori, (cx, cy), 2, (0, 255, 255), 5)
# cv2.circle(img_ori, (x, y), 2, (255, 0, 255), 5)
# cv2.line(img_ori,(centers[candi_index][0],centers[candi_index][1]),(centers[candi_index+1][0],centers[candi_index+1][1]),(255,255,255),2)
# cv2.namedWindow('myWindow1',cv2.WINDOW_NORMAL)
# cv2.imshow('myWindow1',img_ori)
# cv2.namedWindow('myWindow11',cv2.WINDOW_AUTOSIZE)
# cv2.imshow('myWindow11',img_ori)



# cv2.namedWindow('myWindow2',cv2.WINDOW_NORMAL)
# cv2.imshow('myWindow2',img_ori)
# cv2.namedWindow('myWindow22',cv2.WINDOW_AUTOSIZE)
# cv2.imshow('myWindow22',img_ori)
# cv2.namedWindow('myWindow3',cv2.WINDOW_NORMAL)
# cv2.imshow('myWindow3',img_ori)
# cv2.namedWindow('myWindow33',cv2.WINDOW_AUTOSIZE)
# cv2.imshow('myWindow33',img_ori)
cv2.waitKey(1)
# cv2.destroyAllWindows()