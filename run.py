import cv2
import numpy as np
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
import argparse
import time


# input arg parsing to define and handle command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--fullscreen',
                    help='Display window in full screen', action='store_true')
parser.add_argument(
    '-d', '--debug', help='Display debug info', action='store_true')
parser.add_argument(
    '-fl', '--flip', help='Flip incoming video signal', action='store_true')
args = parser.parse_args()


# create model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu', input_shape=(48, 48, 1)))    #First Convolution layer
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))     #Second Convolution layer
model.add(MaxPooling2D(pool_size=(2, 2)))		         #Pooling layer						
model.add(Dropout(0.25))					 #Regularization technique to prevent overfitting

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())		#Flatten 3D output to 1D
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))  #The final layer is a fully connected layer with 7 neurons(7 emotions) and the softmax activation function

model.load_weights('model.h5')


# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful",
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


# start the webcam feed
cap = cv2.VideoCapture(0)
while True:
    # time for fps
    start_time = time.time()

    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()    #reads a frame from the video.'ret' indicates whether the frame was successfully read.
    if args.flip:
        frame = cv2.flip(frame, 1)
    if not ret:
        break
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #creates a CascadeClassifier object for detection faces using the Haar Cascade method
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   #converts the frame from its default color format (BGR) to grayscale
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)    #Detcets faces

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)   #Iterates over each detected face and draws a blue rectangle
	#for each detected face it extracts the region of interest and crops it to 48x48 pixel image
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(
            cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)    #fed into model for prediction
        maxindex = int(np.argmax(prediction))
        emotion_label = emotion_dict[maxindex]
        cv2.putText(frame, emotion_label, (x+20, y-60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)    #Display predicted emotion

    # full screen
    if args.fullscreen:
        cv2.namedWindow("video", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("video", cv2.WND_PROP_FULLSCREEN, 1)

   
    cv2.imshow('video', cv2.resize(
        frame, (800, 480), interpolation=cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):       #Enter 'q' to quit
        break

cap.release()     #releases the resources that are being used to captute the video frames
cv2.destroyAllWindows()
