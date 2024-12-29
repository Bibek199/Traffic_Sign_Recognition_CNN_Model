import cv2 
import numpy as np
import pickle
import tensorflow as tf
import os
os.environ['KMP_DUPLICATE_LIB_OK']= 'True'
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.models import load_model



frameWidth = 1640  
frameHeight = 1480
brightness = 180
prediction_threshold = 0.5
font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

# Importing our model
model = load_model('signclassifier.h5') # Open a new model


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img


while cap.isOpened():
    _,imgo = cap.read()
    img = np.asarray(imgo)
    img=cv2.resize(imgo,(32,32),interpolation = cv2.INTER_AREA)
    img = preprocessing(img)
    cv2.imshow('Video',img)
    
    img = img.reshape(1, 32, 32, 1)
    predictions = model.predict(img)
    print(predictions)
    
    if predictions > 0.5: 
        print('Right')
    else:
        print('Stop')
    if cv2.waitKey(1) & 0xFF==ord('q'): 
        break

# PROCESS IMAGE
#     img = np.asarray(imgOrignal)
#     img = cv2.resize(img, (32, 32))
# img = preprocessing(img)
# cv2.imshow("Processed Image", img)
    # img = img.reshape(1, 32, 32, 1)
# cv2.putText(imgOrignal, "CLASS: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
# cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
# PREDICT IMAGE
# predictions = model.predict(img)
# classIndex = model.predict_classes(img)
# probabilityValue = np.amax(predictions)
# if probabilityValue > threshold:
#     # print(getCalssName(classIndex))
#     cv2.putText(imgOrignal, str(classIndex) + " " + str(getCalssName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
#     cv2.putText(imgOrignal, str(round(probabilityValue * 100, 2)) + "%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
#     cv2.imshow("Result", imgOrignal)

