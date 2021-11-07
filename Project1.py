'''
Made by Matthew Farias, Matteus Di Pietro, Jack Pistagnesi

This is a simple CNN set to be trained over the MNIST database

'''

import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import numpy as np

import csv
import time

batch_size = 96
num_classes = 10
epochs = 100

# used for printing to csv
testName='Final'
start=time.time()

# input image dimensionspip
img_rows, img_cols = 28, 28


# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# shape the matrix into 60000 28x28x1 images
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#convert class vectors to binary class matrices
y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)


# create the model and add the layers to it
model = Sequential()
model.add(tensorflow.keras.Input(shape = (28, 28, 1)))
model.add(Conv2D(64, kernel_size=(3,3), activation = 'selu'))
model.add(Conv2D(64, kernel_size=(3,3), activation = 'selu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Flatten())

# adding the droout layer
model.add(Dropout(0.1))
model.add(Dense(num_classes, activation = 'softmax'))

# model.summary()
model.compile(loss = tensorflow.keras.losses.categorical_crossentropy,
        optimizer = tensorflow.keras.optimizers.Adamax(),
        metrics = ['accuracy'])

# train the model
data = model.fit(x_train, y_train,
        batch_size = batch_size,
        epochs = epochs,
        validation_split= 0.1,
        validation_data=(x_test, y_test))

# test the model
score = model.evaluate(x_test, y_test, verbose=0)

# getting the overall runtime of the model
end=time.time()

# print the results of the testing
print('Test name:', testName)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Write the results of the test as a new line in the results file
with open('CNNPractice\\final.csv','a', newline='') as outfile:
        listWriter = csv.writer(outfile)
        line=[]
        line.append(testName)
        line.append(score[0])
        line.append(score[1])
        line.append((end-start)/60)
        listWriter.writerow(line)
outfile.close()

