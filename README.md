# OCR - CNN_LSTM_CTC_Tensorflow
Used network </br>
https://github.com/watsonyanghx/CNN_LSTM_CTC_Tensorflow/


This is the program which recognize the numbers on the drawing. 
Training network is CRNN with CTC loss layer.

- Convert2PBwithBathNorm.py </br>
학습모델을 freezing 해서 pb파일로 만들어 openCV에 모델 임포트할 때 batchNormalization layer 에서 에러 발생</br>
(batchNormalization layer의 node name을 변경해 freezing)

- ./drawing_ocr </br>
학습된 모델을 불러와 실제 이미지에 적용하여 테스트

Result </br>
![result](https://github.com/godheeran/Machine-Drawing-OCR-CRNN/blob/master/imgs/output.png)
