import os
import cv2
import numpy as np
import tensorflow as tf
import utils
import math

INPUT_NODE_NAME = 'input:0'
OUTPUT_CODE = 'dense_decode:0'
OUTPUT_PROB = 'CTCBeamSearchDecoder:3'
CKPT_PATH = '../checkpoint/'
PATH = './test/'
CHARSET = '0123456789'
MAX_STEP = 28

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
   print("Import .ckpt model...")
   saver = tf.train.import_meta_graph(os.path.join(CKPT_PATH,'ocr-model-76001.meta'))
   saver.restore(sess, tf.train.latest_checkpoint(CKPT_PATH))

   print("Read test folder ...")
   file_list = os.listdir(PATH)
   for file in file_list:  
      file_path = os.path.join(PATH, file)
      txt_path = file.split('.')[0]+'.txt'
      #utils.py 
      crop_point, img_list = utils.readIMG(file_path)
      str_list = []
      total_step = len(img_list)
      for step in range(int(total_step)):
         img = img_list[step]
         prob, code = sess.run(
            [OUTPUT_PROB, OUTPUT_CODE], feed_dict={INPUT_NODE_NAME: [img] }
         )
         output = encode(code)
         prob = round(math.exp(prob), 4)
         output = str(output[0])
         
         #make txt file
         if output == '' or prob < 0.98:
            continue
         else :
            p = ','.join(str(x) for x in crop_point[step])
            str_list.append(output+','+p)

      result_folder = os.path.join(PATH,'result')
      if not os.path.exists(result_folder):
         os.mkdir(result_folder)
      write_file(os.path.join(result_folder, txt_path), str_list)
