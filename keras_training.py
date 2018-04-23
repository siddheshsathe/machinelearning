import argparse
###############################################
parser = argparse.ArgumentParser()
parser.add_argument('--dataFile', '-d', help='Datafile path', required=True)
parser.add_argument('--epochs', '-e', help='Number of epochs', type=int, default=10)
parser.add_argument('--batchSize', '-b', help='Batch size', type=int, default=100)
parser.add_argument('--save', help='Save model', action='store_true', default=False)
args = parser.parse_args()
###############################################
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.models import load_model

df = pd.DataFrame(np.load(args.dataFile, encoding='latin1'))

x = np.array([row for row in df[0]])
y = np.array([row for row in df[1]])


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

numClasses = 3
model = Sequential()
model.add(Dense(500, activation='sigmoid', input_shape=(250 * 250, )))
model.add(Dense(200, activation='sigmoid'))
model.add(Dense(500, activation='sigmoid'))
model.add(Dense(numClasses, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001), metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=args.batchSize, epochs=args.epochs, verbose=1)
if args.save:
	print('Saving the model')
	model.save('model.h5')
