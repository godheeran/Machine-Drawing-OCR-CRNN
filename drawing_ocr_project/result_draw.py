import os
import cv2

IMG_PATH ='/home/god/Desktop/daedong/combine_img/'
CTC_PATH = '/home/god/Desktop/daedong/combine_CNN_Csharp/resultTxt/'
RESULT_PATH = '/home/god/Desktop/daedong/combine_CNN_Csharp_drawImg/'

img_list = os.listdir(IMG_PATH)
font = cv2.FONT_HERSHEY_SIMPLEX
for img in img_list:
    copy = cv2.imread(IMG_PATH+img,1)
    name = img.split('.')[0]
    result = open(os.path.join(CTC_PATH+name+'.txt'))
    for l in result.readlines():
        arr = l.split(',')
        list(map(lambda x:x.strip(),arr))
        cv2.rectangle(copy, (int(arr[1]),int(arr[2])), (int(arr[1])+int(arr[3]),int(arr[2])+int(arr[4])), (0, 0, 255))
        cv2.putText(copy,arr[0],(int(arr[1]), int(arr[2])), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite(RESULT_PATH+img, copy)
