#---------------------------------------------------IMPORT LIBRARIES------------------------------------------------------------------------------------
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import os
from keras.utils.image_utils import img_to_array

#---------------------------------------LOAD FILE DETECT FACE IN PICTURE && LOAD MODEL------------------------------------------------------------------------------------
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model("MaskDetection.h5")

#---------------------------------------------------BUILD FUNCTIONS------------------------------------------------------------------------------------
def predict(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(150,150))
    image = img_to_array(image)
    image = image.astype('float32')
    image = np.expand_dims(image,axis=0)
    image = image/255
    prediction = model.predict(image).argmax()

    return prediction


def detector(gray_image,frame):
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1,minNeighbors=4)
    for(x,y,w,h) in faces:
        face_test = frame[y:y+h,x:x+w]
        mask = predict(face_test)
        print('ket qua du doan là: ',mask)
        if mask ==1 :
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
            cv2.putText(frame,'Khong deo khau trang', ((x-40),y-40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
        else:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
            cv2.putText(frame,'Co deo khau trang', ((x-40),y-40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
    return frame

#---------------------------------------------------MAIN FUNCTION------------------------------------------------------------------------------------
cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    detect = detector(gray_frame,frame)
    cv2.imshow("Mask Detection",detect)
    if cv2.waitKey(1)& 0xFF == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()    