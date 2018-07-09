import argparse
###############################################
parser = argparse.ArgumentParser()
parser.add_argument('--XdataFile', '-xd', help='X Datafile path', required=True)
parser.add_argument('--YlabelFile', '-yl', help='X Label path', required=True)
parser.add_argument('--epochs', '-e', help='Number of epochs', type=int, default=10)
parser.add_argument('--batchSize', '-b', help='Batch size', type=int, default=100)
parser.add_argument('--save', help='Save model name', default='model_keras_conv2d.h5')
args = parser.parse_args()
###############################################
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D, BatchNormalization
from keras.optimizers import SGD

class SignLanguageClassifier(object):
    """
    Class for Sign Language classifier
    """
    def __init__(self, args):
        """
        Initializes variables required for script
        """
        self.args = args
        self.model = None
        self.numClasses = 0
        self._inputShape = None
        self.activation = 'relu'
        self.kernelSize = (4, 4)

    @property
    def inputShape(self):
        """
        Returns the input shape for the input x data
        """
        if not self._inputShape:
            self._inputShape = (self.xTest.shape[1], self.xTest.shape[2], 1, ) # Considering only one channel is present in image. i. e, for GRAY image
        return self._inputShape

    def createModel(self):
        """
        Model
        """
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=self.kernelSize, padding='same', input_shape=self.inputShape, activation=self.activation))
        self.model.add(Conv2D(32, kernel_size=self.kernelSize, padding='same', activation=self.activation))
        self.model.add(Conv2D(32, kernel_size=self.kernelSize, padding='same', activation=self.activation))
        self.model.add(MaxPool2D(pool_size=4))
        self.model.add(BatchNormalization())
        self.model.add(Flatten())
        self.model.add(Dense(500, activation=self.activation))
        self.model.add(Dense(self.numClasses, activation='softmax'))
        self.model.summary()
        self.model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001), metrics=['accuracy'])

    def loadAndProcessData(self):
        """
        Loads the data and processes it for feeding in to network
        """
        xData = np.load(self.args.XdataFile)
        xData = xData.reshape(xData.shape[0], xData.shape[1], xData.shape[2], 1)
        yLabels = np.load(self.args.YlabelFile)
        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(xData, yLabels, test_size=0.2)
        self.numClasses = self.yTest.shape[1]

    def train(self):
        history = self.model.fit(self.xTrain, self.yTrain, batch_size=self.args.batchSize, epochs=self.args.epochs, verbose=1)

        if self.args.save:
            print('Saving the model to {}'.format(self.args.save))
            self.model.save(self.args.save)

        print(history)
        fig1, ax_acc = plt.subplots()
        plt.plot(history.history['acc'])
        plt.plot(history.history['loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        # This is a blocking call, user will have to manually close the window opened by matplotlib showing accuracy vs loss data.
        plt.show()

    def evaluate(self):
        score = self.model.evaluate(self.xTest, self.yTest)
        print(score)

if __name__ == "__main__":
    s = SignLanguageClassifier(args)
    s.loadAndProcessData()
    s.createModel()
    s.train()
    s.evaluate()