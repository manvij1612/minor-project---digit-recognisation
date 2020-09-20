import cv2
import numpy as np
import tensorflow as tf


m_new=tf.keras.models.load_model('minor_project.h5')

img=np.ones([400,400],dtype='uint8')*255

img[50:350,50:350]=0

wname='Canvas'

cv2.namedWindow(wname)
state= False
def shape(event,x,y,flags,param):
    global state
    if event ==cv2.EVENT_LBUTTONDOWN:
        state=True
        cv2.circle(img,(x,y),10,(255,255,255),-1)

    elif event==cv2.EVENT_MOUSEMOVE:
        if (state==True):
            cv2.circle(img,(x,y),10,(255,255,255),-1)
        else:
            state=False



cv2.setMouseCallback(wname,shape)

while True:
    cv2.imshow(wname,img)
    key= cv2.waitKey(1)
    if key ==ord('q'):
        break
    elif key==ord('c'):
        img[50:350,50:350]=0
    elif key==ord('w'):
        out= img[50:350,50:350]
        cv2.imwrite('output.jpg')
    elif key==ord('o'):
        image_test=img[50:350,50:350]
        image_test_resize=cv2.resize(image_test,(28,28)).reshape(1,28,28)
        z=m_new.predict_classes(image_test_resize)
        print(z)

cv2.destroyAllWindows()
            
