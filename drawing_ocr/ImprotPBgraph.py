import os
import cv2
import math
import numpy as np
import tensorflow as tf
#from tensorflow.python.platform import gfile

INPUT_NODE = 'input:0'
OUTPUT_CODE = 'dense_decode:0'
OUTPUT_PROB = 'CTCBeamSearchDecoder:3'
GRAPH_PB_PATH = '../checkpoint/onlyResize28/onlyResized28.pb'
FOLDER_PATH = '../test/'

CHARSET = '0123456789'
SPACE_INDEX = 0
SPACE_TOKEN = ''
encode_maps = {}
decode_maps = {}
for i, char in enumerate(CHARSET, 1):
    encode_maps[char] = i
    decode_maps[i] = char
encode_maps[SPACE_TOKEN] = SPACE_INDEX
decode_maps[SPACE_INDEX] = SPACE_TOKEN

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

dirlist = os.listdir(FOLDER_PATH)
imgs = []
for i in dirlist:
       im = cv2.imread(os.path.join(FOLDER_PATH,i),0).astype(np.float32)/255.
       im = np.reshape(im, [28,28,1])
       imgs.append(im)
   
with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())
   print("Import the TF graph ...")
   with tf.gfile.GFile(GRAPH_PB_PATH,'rb') as f:
       graph_def = tf.GraphDef()
   graph_def.ParseFromString(f.read())
   sess.graph.as_default()
   tf.import_graph_def(graph_def, name='')

   for step in range(len(imgs)):
          im = imgs[step]
          prob, code = sess.run(
             [OUTPUT_PROB, OUTPUT_CODE], feed_dict={INPUT_NODE: [im]}
          )
          decoded = encode(code)
          prob = round(math.exp(prob),4 )
          print(decoded, prob)
   #for test
   # graph_nodes=[n for n in graph_def.node]
   # names = []
   # for t in graph_nodes:
   #    names.append(t.name)
   # print(names)