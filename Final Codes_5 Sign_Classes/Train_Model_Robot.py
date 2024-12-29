# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 21:37:33 2023

@author: bevec
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns 
from PIL import Image, ImageFilter #Perform Iterations over different Classes

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf  

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy

def preprocessing(img):
   img_res = img.resize((80,80))
   img_gray = img_res.convert("L")
   img_smooth = img_gray.filter(ImageFilter.SMOOTH)
   return img_smooth   
    
path = "GSRTB_Robot/Train"
data = []; labels = []
classes = 5
for i in range(classes):
    img_path = os.path.join(path,str(i)) 
    for img in os.listdir(img_path):
        img = Image.open(img_path +'/'+ img)    
        img = preprocessing(img)                 # Calls Image pre-processing function
        img= np.array(img)
        data.append(img)
        labels.append(i)
        
data = np.array(data); 
labels = np.array(labels)

print(data.shape) # The shape of the data gives no. of images, size of images and filters

# print(labels[2429]) # Each image is associated with a label here.

# labels names 
classes = ['Turn left ahead',
           'Turn right ahead',
           'Stop',
           'Speed limit (20km/h)',
           'Speed limit (60km/h)',
           ]

print(classes[0])
    
#COnvert labels to int (to plot a data distribution graph)
# labels=labels.astype(int)
unique_labels, label_counts = np.unique(labels, return_counts=True)
unique_labels
# label_counts

# Create a horizontal bar plot
plt.figure(figsize=(15,10))
plt.barh(unique_labels, label_counts)
plt.xlabel('Count')
plt.ylabel('Label')
plt.title('Label Counts in Train Data')
plt.yticks(unique_labels, [classes[label] for label in range(5)])
plt.show()

a=labels[2500]
print(a)
print(classes[a])
print(data.shape)

indices = np.random.choice(data.shape[0], size=10, replace=False)
print(data.shape[0])
print(indices[1])
samples = data[indices]
print(len(samples))

sns.set(rc = {'figure.figsize':(10,10)})

fig, axes = plt.subplots(2, 5, figsize=(12, 6))

# Sample 10 random images from the dataset and plot with its labels

for i in range(len(samples)):
    ax = axes[i // 5, i % 5]
    ax.imshow(samples[i],cmap='gray')
    ax.set_title(classes[labels[indices[i]]])
    ax.axis('off')
        
    
plt.tight_layout(pad=0,h_pad=0, w_pad=0)
plt.show()

x_train, x_val, y_train, y_val= train_test_split(data, labels, test_size= 0.2, random_state=10)

print("training_shape: ", x_train.shape,y_train.shape)
print("validation_shape: ", x_val.shape,y_val.shape)

# y_train = to_categorical(y_train, 43)
# y_val = to_categorical(y_val, 43)

y_train = tf.one_hot(y_train,5)
y_val = tf.one_hot(y_val,5)

#building the CNN Model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=(80,80,1)))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(5, activation='softmax'))

model.summary()

#Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Fitting the model
epochs=20
history= model.fit(x_train,y_train, epochs=epochs, batch_size=64,validation_data=(x_val, y_val))

plt.figure(0)
plt.plot(history.history['accuracy'], label="Training accuracy")
plt.plot(history.history['val_accuracy'], label="val accuracy")
plt.title("Accuracy Graph")
plt.xlabel("epochs")
plt.ylabel("accuracy (0,1)")
plt.legend()

plt.figure(1)
plt.plot(history.history['loss'], label="training loss")
plt.plot(history.history['val_loss'], label="val loss")
plt.title("Loss Graph")
plt.xlabel("epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

model.save(os.path.join('multiclassifier_Robot.h5')) # save the model to current directory