# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 21:04:52 2023

@author: bevec
"""

import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow import keras
import seaborn as sns
import os
os.environ['KMP_DUPLICATE_LIB_OK']= 'True'

import socket
import time

server_ip = '10.2.44.215'  # <-- IP of my WiFi card on server
server_port = 80

url = "http://10.2.45.96:8080//shot.jpg"



while True:
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    image = cv2.imdecode(img_arr, -1)
    # cv2.imshow("Original_Image", image)
    
    image1 = image.copy()
    image_contour = image.copy()
    image_bound_rect = image.copy()
    image_bound_sqr = image.copy()
    
    
      
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_image = cv2.GaussianBlur(gray_image,(3,3),1)
    
    # cv2.imshow("Blurred", blur_image)
    
    thresh = cv2.Canny(blur_image,80,100)
    
    # ret,thresh = cv2.threshold(blur_image,180,255,cv2.THRESH_BINARY)
    
    cv2.imshow("Thresh", thresh)
    
    rect_contours = []
    # Find the contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        image_contours = cv2.drawContours(image, [cnt], -1, (0,255,0), 1)
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4 and cv2.contourArea(cnt)>500: 
            # print(cv2.contourArea(cnt))
            rect_contours.append(cnt)

    # Show all contours 
    cv2.imshow("image_contours",image_contours)
    
    for cnt in rect_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image_bound_rect,(x,y),(x + w, y + h), (0, 255, 0), 2)  


    # cv2.imshow("Bounding Rect", image_bound_rect)
    
    #Check for aspect ratio: `
    aspect_ratios = []
    square_contours = []

    # Choose square contours in rectangle contours
    for cnt in rect_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/float(h)
        aspect_ratios.append(aspect_ratio)
        if  0.9 <=aspect_ratio<= 1.1:
            square_contours.append(cnt)
    
    for cnt in square_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image_bound_sqr,(x,y),(x + w, y + h), (0, 255, 0), 2)
        print("Box area",cv2.contourArea(cnt))
            

    # Show square Bounding_box only
    # cv2.imshow("Square Rect", image_bound_sqr)
    print("Square Contours Detected: ",len(square_contours))
    
    
    image_cropped = image1[y:y+h,x:x+w]
   
    # cv2.imshow("Cropped_IMage", image_cropped)
    
    classes = ['Turn left ahead',
               'Turn right ahead',
               'Stop',
               'Speed limit (20km/h)',
               'Speed limit (60km/h)',
               ]
    
    # Load the trained model
    model = load_model('multiclassifier_Robot.h5') 
    
    def preprocessing(img):
        res_image = cv2.resize(img,(80,80))
        gray_image = cv2.cvtColor(res_image, cv2.COLOR_BGR2GRAY)
        blur_image = cv2.GaussianBlur(gray_image,(3,3),1)
        return blur_image 
    
    if (len(square_contours)==1 or len(square_contours)==2):
        test_image = image_cropped

        test_image = preprocessing(test_image)
        test_image = keras.preprocessing.image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        predictions = model.predict(test_image)
        class_index = np.argmax(predictions)
        image1 = cv2.putText(image1,classes[class_index], (x,y+h+15),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2, cv2.LINE_AA)
       
    cv2.imshow("Result", image1)
    
    image_cropped=[]
    
    end = cv2.waitKey(1)
    if end == 27:
        break
    elif end == 32:
        time.sleep(0.2)
        print("Robot movement:Move Forward")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((server_ip,server_port))
            s.sendall(b'G') # Move Forward
     
    movement_cmd=0
    for cnt in square_contours:  
        if cv2.contourArea(cnt)>1000:
            movement_cmd = True
            print("Detected Sign",classes[class_index])
            
    if movement_cmd==True and class_index == 2:
        print("Detected Sign",classes[class_index])
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((server_ip,server_port))
            s.sendall(b'S') # Stop
     
    if movement_cmd==True and class_index == 0:
        print("Detected Sign",classes[class_index])
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((server_ip,server_port))
            s.sendall(b'L') # Left
            time.sleep(1.5)
            s.sendall(b'S') #Stop
            
            
    if movement_cmd==True and class_index == 1:
        print("Detected Sign",classes[class_index])
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((server_ip,server_port))
            s.sendall(b'R') # Right
            time.sleep(1.5)
            s.sendall(b'S') # Stop
            
    if movement_cmd==True and class_index == 3:
        print("Detected Sign",classes[class_index])
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((server_ip,server_port))
            s.sendall(b'B') # Speed60
    
    if movement_cmd==True and class_index == 4:
        print("Detected Sign",classes[class_index])
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((server_ip,server_port))
            s.sendall(b'A') # Speed20        
    
    # Communication To Robot Movement
    
    # while (movement_cmd):    
    #     def switch(class_index):
    #         if  class_index == "0":
    #             print("Robot movement:Turn Left")
    #             # with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    #             #     s.connect((server_ip,server_port))
    #             #     s.sendall(b'L') # Turn left
                    
    #         if  class_index == "1":
    #             print("Robot movement:Turn Right")
    #             # with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    #             #     s.connect((server_ip,server_port))
    #             #     s.sendall(b'R') # Turn right
                
    #         if class_index == "2":
    #             print("Robot movement:Stop")
                # with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                #     s.connect((server_ip,server_port))
                #     s.sendall(b'S') # Stop
                
            # if class_index == "3":
            #     print("Robot movement:Speed Limit 20")
            #     # with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            #     #     s.connect((server_ip,server_port))
            #     #     s.sendall(b'A') # Slow Down
                
            # if class_index == "4":
            #     print("Robot movement:Speed Limit 60")  
            #     # with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            #     #     s.connect((server_ip,server_port))
            #     #     s.sendall(b'B') # Speed up
            
      
    
cv2.destroyAllWindows() 