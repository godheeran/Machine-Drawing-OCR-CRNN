import cv2
import os

path = './chk/'
new = './chk/imgs/'
list_dir = os.listdir(path)

for i, f in enumerate(list_dir):
    img = cv2.imread(os.path.join(path, f), 0)
    cv2.imwrite(new+'%05d'%i+'.png', img)