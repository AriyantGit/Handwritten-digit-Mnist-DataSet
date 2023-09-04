import cv2
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
##############################
width=640
height=480
##############################
device_ip = '192.168.0.100'
port_number = '4747'

# Create the video capture object using the DroidCam URL
droidcam_url = f'http://{device_ip}:{port_number}/video'
#cap = cv2.VideoCapture(droidcam_url)
cap=cv2.VideoCapture(0)
#####################

cap.set(3,width)
cap.set(4,height)


##########################
pickle_in=open("CNN_Model_trained.h5","rb")
model=pickle.load(pickle_in)
pickle_in.close()

#model = tf.keras.models.load_model('HMnist.keras')
#########################

def preProcessing(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #img=cv2.equalizeHist(img)
    img=img/255
    return img


while True:
    success,imgOriginal=cap.read()
    img=np.asarray(imgOriginal)
    img=cv2.resize(img,(28,28))
    img=preProcessing(img)
    
    #img=img.reshape(1,28,28,1)
    classindex=model.predict(img.reshape(1,28,28,1))
    
    probability=np.amax(classindex)
    classname=classindex.argmax(axis=1)[0]
    if(probability>0.80):
        cv2.putText(imgOriginal,str(probability)+" "+str(classname),
                    (150,150),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,(0,0,255),1)
        print(classindex.argmax(axis=1)[0],probability)
    cv2.imshow("Image",imgOriginal)
    if cv2.waitKey(1) & 0xFF==ord("q"):

        break


