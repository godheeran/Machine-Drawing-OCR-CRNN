import os
import cv2
import numpy as np
import tensorflow as tf
import classifiy
from tensorflow.python.platform import gfile

INPUT_NODE_NAME = 'input:0'
OUTPUT_NODE_NAME = 'CTCBeamSearchDecoder:0'
GRAPH_PB_PATH = '../checkpoint/CTCBeamSearchDecoder.pb'
PATH = './chk/0.png'
CHARSET = '0123456789'
encode_maps = {}
decode_maps = {}
for i, char in enumerate(CHARSET, 1):
    encode_maps[char] = i
    decode_maps[i] = char

SPACE_INDEX = 0
SPACE_TOKEN = ''
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
with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())
   print("Import the TF graph ...")
   with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
       graph_def = tf.GraphDef()
   graph_def.ParseFromString(f.read())
   sess.graph.as_default()
   tf.import_graph_def(graph_def, name='')
   
   # imgs/infer
   im = cv2.imread(PATH,0).astype(np.float32)/255.
   im = np.reshape(im, [28, 28, 1])
   im = np.asarray(im)

   code = sess.run(
      OUTPUT_NODE_NAME, feed_dict={INPUT_NODE_NAME : [im]}
   )
   #decoded = tf.sparse_tensor_to_dense(code, default_value=0)

   print(code.dense_shape())