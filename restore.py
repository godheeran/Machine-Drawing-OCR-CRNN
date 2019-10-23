import tensorflow as tf
import os
import cv2
import numpy as np
import cnn_lstm_otc_ocr
import math

IMG_PATH = './imgs/test/'
CKPT_PATH = './checkpoint/data_copyborder_v2/'
INPUT_NODE_NAME = 'input:0'

# model = cnn_lstm_otc_ocr.LSTMOCR('infer')
# model.build_graph()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    #network
    saver = tf.train.import_meta_graph(CKPT_PATH+'ocr-model-499001.meta')
    saver.restore(sess, tf.train.latest_checkpoint(CKPT_PATH))
    
    #make input
    img = []
    for root, sub_folder, file_list in os.walk(IMG_PATH):
        for f in file_list :
            im = cv2.imread(os.path.join(root, f),0).astype(np.float32) /255.
            im = np.reshape(im, [28, 28, 1])
            img.append(im)
    #infer
    step = len(img)
    for s in range(int(step)):
        input_img = [img[s]]
        prob, code = sess.run(['CTCBeamSearchDecoder:3','dense_decode:0'], feed_dict={INPUT_NODE_NAME:input_img})
        print(code)
        print(round(math.exp(prob),4))