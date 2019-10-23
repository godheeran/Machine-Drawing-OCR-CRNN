import os
import cv2
import numpy as np
import tensorflow as tf
import classifiy
import math
from tensorflow.python.platform import gfile

INPUT_NODE_NAME = 'input:0'
OUTPUT_CODE = 'dense_decode:0'
OUTPUT_PROB = 'CTCBeamSearchDecoder:3'
CKPT_PATH = '../checkpoint/pool_2/'
#GRAPH_PB_PATH = '../checkpoint/data_copyborder_v2/sparse_tensor_to dense_model_v2.pb'
PATH = '/home/god/Desktop/daedong/combine_img/'
TXT_PATH = '/home/god/Desktop/daedong/combine_pool_2_result/'
CHARSET = '0123456789'
MAX_STEP = 28

# def loadIMG(path): #imgs/infer 
#    imgs_input = []
#    for root, sub_folder, file_list in (os.walk(path)):
#       for f in file_list:
#          f_path = os.path.join(root, f)
#          im = cv2.imread(f_path,0).astype(np.float32)/255.
#          im = np.reshape(im, [28, 28, 1])
#          imgs_input.append(im)
#    imgs_input = np.asarray(imgs_input)
#    return imgs_input

def encode(dense_decoded_code):
   decoded_expression = []
   for item in dense_decoded_code:
      expression = ''
      for i in item:
         if i == -1:
            expression += ''
         else:
            expression += decode_maps[i]
      decoded_expression.append(expression)
   return decoded_expression

def write_file(result_txt,str_list):
   txt = open(result_txt, 'w')  
   for l in str_list:
      txt.write(l+'\n')

encode_maps = {}
decode_maps = {}
for i, char in enumerate(CHARSET, 1):
    encode_maps[char] = i
    decode_maps[i] = char

SPACE_INDEX = 0
SPACE_TOKEN = ''
encode_maps[SPACE_TOKEN] = SPACE_INDEX
decode_maps[SPACE_INDEX] = SPACE_TOKEN

with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())
   # print("Import the TF graph ...")
   # with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
   #     graph_def = tf.GraphDef()
   # graph_def.ParseFromString(f.read())
   # sess.graph.as_default()
   # tf.import_graph_def(graph_def, name='')
   #network
   saver = tf.train.import_meta_graph(CKPT_PATH+'ocr-model-76001.meta')
   saver.restore(sess, tf.train.latest_checkpoint(CKPT_PATH))

   print("Reading the image folder ...")
   for root, sub_folder, file_list in os.walk(PATH):
      for file in file_list:  
         file_path = os.path.join(root, file)
         txt_path = file.split('.')[0]+'.txt'
         print(file_path)
         #classify.py 
         crop_point, img_list = classifiy.readIMG(file_path)
         str_list = []
         total_step = len(img_list)
         for step in range(int(total_step)):
            img = [img_list[step]]
            prob, code = sess.run(
               [OUTPUT_PROB, OUTPUT_CODE], feed_dict={INPUT_NODE_NAME: img }
            )
            output = encode(code)
            prob = round(math.exp(prob), 4)
            output = str(output[0])
            #make txt file
            if output == '' or prob < 0.98:
               #file write pass
               continue
            else :
               p = ','.join(str(x) for x in crop_point[step])
               str_list.append(output+','+p)
         write_file(TXT_PATH+txt_path, str_list)
