import os
import cv2
import numpy as np

HEIGHT, WIDTH = 28, 28

def sortContours(dected_list):
    connect_obj = []
    space = 13
    while True:
        if not dected_list:
            break
        cnt = 0
        [x, y, w, h] = dected_list.pop()
        for i, chk in enumerate(dected_list):
            [x_, y_, w_, h_] = chk  
            #front
            if (x+w)<x_ and (x+w+space)>x_ and y-5<=y_ and y_<=y+5:
                #connect_obj.append([x, y, (x_-x+w_), h])
                dected_list.pop(i)
                dected_list.append([x, y, (x_-x+w_), h])
                cnt += 1
                break
            #back
            if (x_+w_)<x and (x_+w_+space)>x and y_-5<=y and y<=y_+5:
                #connect_obj.append([x_, y_, (x-x_+w), h_])
                dected_list.pop(i)
                dected_list.append([x_, y_, (x-x_+w), h_])
                cnt += 1
                break
        if cnt == 0 :
            connect_obj.append([x, y, w, h])
    return connect_obj

def make_image_list(src, point_list, imgPath):
    img_arr = []
    for point in point_list:
        [x, y, w, h] = point
        pad = 4 
        crop = src[y-pad:y+h+pad, x-pad:x+w+pad]
        padding = cv2.resize(crop, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
        padding = np.reshape(padding, [HEIGHT,WIDTH,1])
        img_arr.append(padding)
    return img_arr

def readIMG(imgPath):
    img = cv2.imread(imgPath,cv2.IMREAD_GRAYSCALE)#.astype(np.float32) / 255.
    _, thr = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    opening = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel)
    _, contours, hierachy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_obj = []
    for cnt in contours:
        [x, y, w, h] = cv2.boundingRect(cnt)
        if h>19 and h<30 and w>8 and w<30:
                detected_obj.append([x, y, w, h])
    #connect nearest digit
    connect_obj = sortContours(detected_obj)
    #make image array list
    img_list =  make_image_list(thr, connect_obj,imgPath.split('/')[-1].split('.')[0])
    img_list = np.asarray(img_list)
    return (connect_obj, img_list)