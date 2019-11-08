import numpy as np
import os

GT_PATH = '/home/god/Desktop/daedong/combine_gt/'
CTC_PATH = '/home/god/Desktop/daedong/combine_pool_2_result/'
RESULT_TXT ='/home/god/Desktop/daedong/combin_pool_2.txt'
chk = 5

files_list = os.listdir(CTC_PATH)
result = open(RESULT_TXT, 'w')
for file in files_list:
    ctc_path = os.path.join(CTC_PATH, file)
    gt_path = os.path.join(GT_PATH, file)
    if  os.path.exists(gt_path):
        ctc = open(ctc_path, 'r')
        cnt, ctc_count = 0, 0
        for c in ctc.readlines():
            ctc_count+=1
            arr1 = c.split(',')
            n1 = int(arr1[0])
            x1 = int(arr1[1])
            y1 = int(arr1[2])
            gt = open(gt_path, 'r')
            gt_count = 0
            for g in gt.readlines():
                gt_count+=1
                arr2 = g.split(',')
                n2 = int(arr2[0])
                x2 = int(arr2[1])
                y2 = int(arr2[2])
                if n1==n2:
                    if x2-chk<=x1 and x1<x2+chk and y2-chk<=y1 and y1<y2+chk:
                        cnt+=1     
            gt.close()        
        if cnt == 0 or gt_count ==0 or ctc_count ==0 :
            #cnt is   
            txt = file+'\n'
        else :
            #name cnt ctc-count gt-count recall pricision
            recall = round(float(cnt)/float(gt_count), 4)
            pric = round(float(cnt)/float(ctc_count), 4)
            txt = file+'\t'+str(cnt)+'\t'+str(ctc_count)+'\t'+str(gt_count)+'\t'+str(recall)+'\t'+str(pric)+'\n'
        result.writelines(txt)
        gt.close()
    ctc.close()
result.close()
        
