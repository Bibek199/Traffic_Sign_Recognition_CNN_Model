# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 22:01:48 2023

@author: bevec
"""
import cv2 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageFilter

from tensorflow import keras

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

from tensorflow.keras.models import load_model

classes = ['Turn left ahead',
           'Turn right ahead',
           'Stop',
           'Speed limit (20km/h)',
           'Speed limit (60km/h)',
           ]


model = load_model('multiclassifier_Robot.h5') # Load the trained model

def preprocessing(img):
    img_res = img.resize((80,80))
    img_gray = img_res.convert("L")
    img_smooth = img_gray.filter(ImageFilter.SMOOTH)
    return img_smooth  

#testing accuracy on test dataset
y_test = pd.read_csv('GSRTB_Robot/Test_Robot.csv')
labels = y_test["ClassId"].values
imgs = y_test["Path"].values
# print(labels.shape)
print(imgs.shape)
test_data=[]
for img in imgs:
   image = Image.open(img)
   image = preprocessing(image)
   test_data.append(np.array(image))
X_test=np.array(test_data)

print(X_test.shape)
plt.imshow(X_test[51],cmap='gray')
plt.show()


y_predict = np.argmax(model.predict(X_test), axis=1)
# pred = model.predict_classes(X_test)
#Accuracy with the test data

print(y_predict)
print(len(y_predict))
accuracy = accuracy_score(labels, y_predict)
print('Accuracy: %f' % accuracy)
precision = precision_score(labels, y_predict, average='macro')
print('Precision: %f' % precision)
recall = recall_score(labels, y_predict, average='macro')
print('Recall: %f' % recall)
f1score = f1_score(labels, y_predict, average='macro')
print('F1 score: %f' % f1score)   
print(y_predict)
print(labels)

# cm = confusion_matrix(labels, y_predict)
# # print confusion matrix for each label
# for i in range(len(cm)):
#     print("Confusion Matrix for Label", i)
#     print(np.array2string(cm[i], separator=', '))

# recall_data = []

# for i in range(43):
#     recall_a = recall_score(labels, y_predict, labels=[i], average=None)
#     print(recall_a[0])
#     recall_data.append(recall_a[0])
 
# print(recall_data)   
# from statistics import mean 
# print(mean(recall_data))


print(precision_score(labels, y_predict, labels=[0], average=None))
print(recall_score(labels, y_predict, labels=[0], average=None))

# Confusion matrix
cm = confusion_matrix(labels, y_predict)
sns.set(rc = {'figure.figsize':(25,15)})
ax = plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax);
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); ax.set_title('Confusion Matrix'); 

#Classification Report

print(classification_report(labels, y_predict, labels=[0,1,2,3,4]))

# Test a single Image: 
    
path='GSRTB_Robot/CNN_Robot_Model_Test/2.png'

test_image1 = keras.preprocessing.image.load_img(path)
sns.set(rc = {'figure.figsize':(5,5)})
plt.imshow(test_image1)
plt.show()
test_image= preprocessing(test_image1)
plt.imshow(test_image,cmap='gray')
plt.show() 
test_image = keras.preprocessing.image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# Predict the class of the test image
predictions = model.predict(test_image)
class_index = np.argmax(predictions)

# Print the predicted class
print("The traffc sign predicted is: ", classes[class_index])