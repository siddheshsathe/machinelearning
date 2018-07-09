import argparse
###############################################
parser = argparse.ArgumentParser()
parser.add_argument('--dataFile', '-d', help='Datafile path', required=True)
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
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D
from keras.optimizers import SGD
from keras.models import load_model


class ImageClassifier(object):
    """
    Image classifier class
    """
    def __init__(self, args):
        """
        Initializes the required variables
        """
        self.args = args
        self.model = None
        self.data = None
        self.df = None
        self.x = None
        self.y = None
        self.xTrain = None
        self.xTest = None
        self.yTrain = None
        self.yTest = None
        self.numClasses = 0
        self.activation = 'relu'

    def createModel(self):
        """
        Creates a Conv2D sequential Keras model
        One can play with the numbers of conv layers and others except input shape.
        I've optimized below numbers considering my HW limitations
        """
        self.model = Sequential()
        self.model.add(Conv2D(16, kernel_size=5, padding='same', input_shape=(250 , 250, 1, ), activation=self.activation))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Conv2D(8, kernel_size=5, padding='same', activation=self.activation))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.3))
        self.model.add(Flatten())
        self.model.add(Dense(300, activation=self.activation))
        self.model.add(Dense(self.numClasses, activation='softmax'))
        self.model.summary()
        self.model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001), metrics=['accuracy'])

    def loadAndPrepareData(self):
        """
        Loads the data from npz file and prepares it for feeding it to network
        """
        #I've used data file as npz. And images are stored in (250 * 250 * 1) i.e, GRAY format of CV2
        self.data = np.load(self.args.dataFile)

        # Loading the data in dataFrame of pandas
        self.df = pd.DataFrame(self.data.items()[0][1])

        # Normalizing the data by dividing the individual element by 255
        self.x = np.array([np.divide(item, 255) for item in self.df[0]])

        # Loading the xData in x
        self.y = np.array([item for item in self.df[1]])
        # Reshaping it for accomodation on model. New shape = (250, 250, 1, 1)
        self.x = self.x.reshape(self.x.shape[0], self.x.shape[1], self.x.shape[2], 1)

        # Splitting the xTrain, xTest, yTrain and yTest from x and y
        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(self.x, self.y, test_size=0.2)

        # Number of classes are 
        # 1. Myself (Siddhesh)
        # 2. My beautiful wife (Ketaki)
        # 3. Unknown (None of us or ppl whom the model had never seen)
        self.numClasses = self.y.shape[1]

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
    imageClassifier = ImageClassifier(args)
    imageClassifier.loadAndPrepareData()
    imageClassifier.createModel()
    imageClassifier.train()
    imageClassifier.evaluate()