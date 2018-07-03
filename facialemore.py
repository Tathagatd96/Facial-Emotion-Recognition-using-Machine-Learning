# -*- coding: utf-8 -*-
#importing libraries
import os,cv2
import numpy as np
import dlib

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

from keras import backend as K
K.set_image_dim_ordering('tf')

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop
from sklearn.metrics import classification_report,confusion_matrix
import itertools

#creating classifiers
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye = cv2.CascadeClassifier('haarcascade_eye.xml')
smile = cv2.CascadeClassifier('haarcascade_smile.xml')

detector = dlib.get_frontal_face_detector() #Face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Landmark identifier. Set the filename to whatever you named the downloaded file

#creating stream object
stream=cv2.VideoCapture(0)

#loop for boot-up screen
while True:
    
    #getting background
    frame =cv2.imread('tr.jpg')
    
    #displaying text
    font=cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,'FACIAL EMOTION RECOGNITION',(210,410),font,1.2,(255,255,255),3)
    cv2.putText(frame,'PRESS S TO START',(400,465),font,0.7,(255,255,255),2)
    
    #resizing window 
    cv2.namedWindow('Facial Emotion Recognition - Press S To Start',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Facial Emotion Recognition - Press S To Start', 600,340)
    img = cv2.resize(frame, (600, 340)) 
    
    #displaying boot-up screen
    cv2.imshow('Facial Emotion Recognition - Press S To Start',img)
   
    if cv2.waitKey(1) & 0xFF==ord('s'):
        break
        
cv2.destroyAllWindows()
#boot-up screen closed

#loop for main logic
while True:
    
    #getting video
    ret, frame=stream.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY) #grayscale
    
    #detecting face
    faces = face.detectMultiScale(gray, 1.3  , 5)
    fac = face.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        #detecting eyes
        eyes = eye.detectMultiScale(roi_gray, 1.3  , 5)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    
        #detecting smiles
        smiles = smile.detectMultiScale(roi_gray, 1.7  , 22)
        for (x,y, w, h) in smiles:
            cv2.rectangle(roi_color, (x, y), (x+w, y+h), (0, 0, 255), 1)
            
        
    #creating clahe
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray)
    #cv2.imshow('clahe',clahe_image)
    
   
    
    
    #cv2.putText(frame, "Number of faces detected: " + str(fac.shape[0]), (0,frame.shape[0] -10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,0), 1)
    
    #print type(fac)
    #print fac
    #print fac.shape
    #print "Number of faces detected: " + str(fac.shape[0])
    
     #creating detector
    detections = detector(clahe_image, 1) #Detect the faces in the image
    
    #displaying landmarks in real time
    for k,d in enumerate(detections): #For each detected face
        
        shape = predictor(clahe_image, d) #Get coordinates
        print shape
        for i in range(1,68): #There are 68 landmark points on each face
                cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0,255,0), thickness=1) #For each point, draw a red circle with thickness2 on the original frame
    frame1=cv2.resize(frame, (600, 340)) 
    #displaying text
    font=cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,'Scanning... Please Wait..',(480,90),font,0.9,(255,255,255),2)
    cv2.putText(frame,'Press C To Capture',(550,705),font,0.7,(255,255,255),2)
    
    def crop_face(gray, face): #Crop the given face
        for (x, y, w, h) in face:
            faceslice = gray[y:y+h, x:x+w]
            return faceslice
    
    
    #resizing the frame
    cv2.namedWindow('Facial Detector - Press C To Capture',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Facial Detector - Press C To Capture', 600,340)
    img = cv2.resize(frame, (600, 340)) 
    cv2.imshow('Facial Detector - Press C To Capture',img)
   
    
    if cv2.waitKey(1) & 0xFF== ord('c'):
        face=crop_face(gray,faces)
        cv2.imwrite('capture.jpg',face)
        cv2.imwrite('display.jpg',frame1)
        break  
   
stream.release()
cv2.destroyAllWindows()

while True:
    
       #getting background
    frame =cv2.imread('display.jpg')
    
    #displaying text
    font=cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,'Captured Image',(250,50),font,0.5,(255,255,255),1)
    cv2.putText(frame,'Press P To Predict',(264,325),font,0.3,(255,255,255),1)
    
    
    #resizing window 
    cv2.namedWindow('Facial Emotion Recognition - Captured Image',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Facial Emotion Recognition - Captured Image', 600,340)
    img = cv2.resize(frame, (600, 340)) 
    
    #displaying boot-up screen
    cv2.imshow('Facial Emotion Recognition - Captured Image',img)
    
    if cv2.waitKey(1) & 0xFF== ord('p'):
        break  
   
stream.release()
cv2.destroyAllWindows()



img_rows=128
img_cols=128
num_channel=1
num_epoch=20

# Define the number of classes
num_classes = 4


from keras.models import load_model
model=load_model('RUN7.hdf5')

model.summary()
model.get_config()
 
# Evaluating the model


# Testing a new image
test_image = cv2.imread('capture.jpg')
test_image=cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
test_image=cv2.resize(test_image,(128,128))
test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255
print (test_image.shape)
   
if num_channel==1:
	if K.image_dim_ordering()=='th':
		test_image= np.expand_dims(test_image, axis=0)
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
	else:
		test_image= np.expand_dims(test_image, axis=3) 
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
		
else:
	if K.image_dim_ordering()=='th':
		test_image=np.rollaxis(test_image,2,0)
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
	else:
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
		

# Predicting the test image
output = model.predict(test_image)
output = output.tolist()
output = output[0]
anger = output[0]
disgust = output[1]
happy = output[2]
neutral = output[3]
surprise = output[4]
#print(model.predict(test_image))

frame =cv2.imread('display.jpg')
    
#displaying text
font=cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(frame,'EMOTIONS DETECTED :',(410,120),font,0.41,(255,255,255),1)
cv2.putText(frame,'Anger=' + str(anger * 100) + "%",(410,150),font,0.41,(255,255,255),1)
cv2.putText(frame,'Disgust=' + str(disgust * 100) + "%",(410,170),font,0.41,(255,255,255),1)
cv2.putText(frame,'Happy=' + str(happy * 100) + "%",(410,190),font,0.41,(255,255,255),1)
cv2.putText(frame,'Neutral=' + str(neutral * 100) + "%",(410,210),font,0.41,(255,255,255),1)
cv2.putText(frame,'Surprise=' + str(surprise * 100) + "%",(410,230),font,0.41,(255,255,255),1)
    
    
#resizing window 
cv2.namedWindow('Facial Emotion Recognition - Captured Image',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Facial Emotion Recognition - Captured Image', 600,340)
img = cv2.resize(frame, (600, 340)) 
    
#displaying boot-up screen
cv2.imshow('Facial Emotion Recognition - Captured Image',img)
   




