import keras as kr
from keras.datasets import mnist # automatically downloaded once function is called.
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# MNIST convolutional neural network (CNN)

# Adapted from: https://github.com/wxs/keras-mnist-tutorial/blob/master/MNIST%20in%20Keras.ipynb

# input image dimensions
# 28x28 pixel images. 
img_rows, img_cols = 28, 28

# Load the pre-shuffled MNIST dataset from keras.datasets -> Dataset of 60,000 28x28 grayscale images of 10 digits & test set of 10,000 images
# 2 tuples ( immutable data structure/lists consisting of mulitple parts):
# x_train, x_test = Uint8 array of grayscale image data with shape (num_samples, 28,28)
# y_train, y_test = Uint8 array of digit labels (integers in range 0-9) with shape(num_samples)
# Adapted from: https://keras.io/datasets/
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# this assumes our data format 
# depending on format...Arrange the data in a certain way
# For 3D data, "channels_last" assumes (conv_dim1, conv_dim2, conv_dim3, channels) while 
# "channels_first" assumes (channels, conv_dim1, conv_dim2, conv_dim3).
if K.image_data_format() == 'channels_first':
    
    # We must declare a dimension of depth of the input image
    # a full-color image with all 3 RGB channels will have a depth of 3
    # MNIST images only have a depth of 1 (greyscale) but we need to declare that..
    # we want to transform our dataset from having shape (n, width, height) to (n, depth, width, height).
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)

else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

input_shape = (img_rows, img_cols, 1)

# Format/reshape the data for training
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# normalize inputs from 0-255 to 0-1
x_train /= 255
x_test /= 255

# there are 10 image classes
num_classes = 10 

# Convert 1-dimensional class arrays to 10-dimensional class matrices
y_train = kr.utils.to_categorical(y_train, num_classes)
y_test = kr.utils.to_categorical(y_test, num_classes)
# This means that if the array is:
# [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] then digit = 0
# [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] then digit = 1
# [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] then digit = 2
# [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] then digit = 3
# [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] then digit = 4 and so on..to 9

# ================== CREATE MODEL ============================================================
# A model is understood as a sequence or a graph of standalone, 
# ..fully-configurable modules that can be plugged together with as little restrictions as possible.
# For more: https://keras.io/
# Creating model and a neural network
# model is used to organise layers
# Create our model using sequential model which is a linear stack of layers
model = Sequential()

# Using 2D convolution layer. Ref: https://keras.io/layers/convolutional/
# Declare input layer
# 32 = the number output of filters in the convolution
# kernel_size = list of 2 integers, specifying the width and height of the 2D convolution window
# activation function 'relu' which means rectified linear unit
# input_shape = (1,28,28) = (depth, width, height)
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 kernel_initializer='he_normal',
                 input_shape=input_shape))

# Dropout method for regularizing our model in order to prevent overfitting
# Add another convolution layer
model.add(Conv2D(64, (3, 3), activation='relu'))
# MaxPooling2D is a way to reduce the number of parameters in our model.. 
# by sliding a 2x2 pooling filter across the previous layer and taking the max of the 4 values in the 2x2 filter
model.add(MaxPooling2D(pool_size=(2, 2)))
# Dropout method for regularizing our model in order to prevent overfitting
model.add(Dropout(0.25)) # float between 0 and 1. Fraction of the input units to drop.
# Flatten the weights to 1 dimensional before passing to dense layer
model.add(Flatten())
# Add dense layer
# 128 = output size
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


# ================ TRAIN THE MODEL ======================================================
# Configure and compile the model for training.
# Uses the adam optimizer and categorical cross entropy as the loss function.
# using adam optimizer algorithm for gradient descent
# Loss function is the objective that the model will try to minimize
# Add in some extra metrics - accuracy being the only one
model.compile(optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy"])


# Fit the model using our training data.
# epochs is the number of times the training algorithm will iterate over the entire training set before terminating
# batch_size is the number of training examples being used simultaneously during a single iteration
# verbose is used to log the model being trained
# verbose=1 means verbose mode 1 which is a progress bar
# Verbosity mode: 0 = silent, 1 = progress bar, 2 = one line per epoch.
model.fit(x_train, y_train, batch_size=128, epochs=4, verbose=1)

# ================== TEST MODEL ==========================================================
# Evaluate the model using the test data set.
# model.evaluate compare answers
# Verbose mode: 0 = silent
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

# Output the accuracy of the model.
print("\n\nLoss: %6.4f\tAccuracy: %6.4f" % (loss, accuracy))

# ================== Prediction ===========================================================
# Predict the digit
# using model.predict
# numpy.around to evenly round the given data
# numpy.expand_dims to expand the shape of an array
prediction = np.around(model.predict(np.expand_dims(x_test[0], axis=0))).astype(np.int)[0]

# print out the actual digit and the prediction
print("Actual: %s\tEstimated: %s" % (y_test[0].astype(np.int), prediction))

# Json file used to save structure of model 
# faster way to have the model loaded when called from the app
# parse/serialise model to json format
model_json = model.to_json()
with open("mnistModel.json", "w") as json_file:
    json_file.write(model_json)

# Save the model weights (learned values)
# h5 is the file format for keras to save the weights of model
model.save("mnistModel.h5")