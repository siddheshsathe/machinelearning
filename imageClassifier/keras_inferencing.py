import argparse
import os
import numpy as np
#################
parser = argparse.ArgumentParser()
parser.add_argument('--image', '-i', help='Input image', required=True)
parser.add_argument('--model', help='Saved model path', required=True)
args = parser.parse_args()
################

from keras.models import load_model
import cv2

model = load_model(args.model)

def getPersonName(prediction):
    nameDict = {
        0: 'Siddhesh',
        1: 'Ketaki',
        2: 'Unknown'
    }
    return nameDict[np.argmax(prediction)]

def getPrediction(inputImage):
    inputImage = inputImage.reshape(1, 62500)
    opt = model.predict(inputImage)
    return getPersonName(opt)

def getImage():
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        # This will show the rectangle box on the screen of size 250*250.
        # Bring the face in that rectangle and prediction will happen on that region only
        cv2.rectangle(frame, (150, 150), (400, 400), (0, 255, 0), 1)
        cv2.imshow('Frame', frame)
        # Making sure that the input size of the image is 250*250 only.
        inputImage = frame[150:400, 150:400]
        inputImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
        print('\rHi {}'.format(getPrediction(inputImage)), end=' ')
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            break

getImage()
