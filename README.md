# Reference - CNN_LSTM_CTC_Tensorflow
https://github.com/watsonyanghx/CNN_LSTM_CTC_Tensorflow/


This is the program which recognize the numbers on the drawing. 
The training network is CRNN with CTC loss layer.

- main.py
학습을 진행시키는 스크립트

- help.py 
데이터베이스를 train과 val 랜덤 분할해주는 스크립트
EX) 1.png .... 10000.png / labels.txt(폴더 밖에 위치)

- utils.py
학습에 대한 파라미터를 가지고 있는 스크립트

-cnn_lstm_otc.py
directional lstm cell (2stacks)로 구성된 네트워크

- cnn_blstm_otc_ocr.py
bidirectional lstm cell (2stacks)로 구성된 네트워크

- Convert2PBwithBathNorm.py
batchNormalization layer를 사용한 네트워크를 기존 python api script를 사용해 .pb로 변환할 때 에러가 발생
batchNormalization layer의 node name을 변경하면서 .pb로 만드는 스크립트 

- ./drawing_ocr
학습된 모델을 이용하여 도면상의 ocr을 진행
