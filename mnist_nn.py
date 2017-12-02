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
#
# Adapted from: https://github.com/wxs/keras-mnist-tutorial/blob/master/MNIST%20in%20Keras.ipynb
#

num_classes = 10

# input image dimensions
#28x28 pixel images. 
img_rows, img_cols = 28, 28

# Load the MNIST dataset from keras.datasets -> Dataset of 60,000 28x28 grayscale images of 10 digits & test set of 10,000 images
# 2 tuples ( immutable data structure/lists consisting of mulitple parts):
# x_train, x_test = Uint8 array of grayscale image data with shape (num_samples, 28,28)
# y_train, y_test = Uint8 array of digit labels (integers in range 0-9) with shape(num_samples)
# Adapted from: https://keras.io/datasets/
(x_train, y_train), (x_test, y_test) = mnist.load_data()


#this assumes our data format
#For 3D data, "channels_last" assumes (conv_dim1, conv_dim2, conv_dim3, channels) while 
#"channels_first" assumes (channels, conv_dim1, conv_dim2, conv_dim3).
if K.image_data_format() == 'channels_first':
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
x_train /= 255
x_test /= 255

#print("Training matrix shape", x_train.shape)
#print("Testing matrix shape", x_test.shape)

# Convert the data to binary matrices
y_train = kr.utils.to_categorical(y_train, num_classes)
y_test = kr.utils.to_categorical(y_test, num_classes)


# Create our model
model = Sequential()

model.add(Conv2D(128, kernel_size=(3, 3),
                 activation='relu',
                 kernel_initializer='he_normal',
                 input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


model.compile(optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy"])

# Train the model
model.fit(x_train, y_train, batch_size=128, epochs=4, verbose=1)

# Test the model
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

# Output the accuracy of the model.
print("\n\nLoss: %6.4f\tAccuracy: %6.4f" % (loss, accuracy))

# Prediction
prediction = np.around(model.predict(np.expand_dims(x_test[0], axis=0))).astype(np.int)[0]

# Check which items we got right / wrong
#correct_indices = np.nonzero(prediction == y_test)[0]
#incorrect_indices = np.nonzero(prediction != y_test)[0]

print("Actual: %s\tEstimated: %s" % (y_test[0].astype(np.int), prediction))

# Json file used to save structure of model 
# parse/serialise model to json format
model_json = model.to_json()
with open("mnistModel.json", "w") as json_file:
    json_file.write(model_json)

# Save the model 
# h5 is the file format for keras to save the weights of model
model.save("mnistModel.h5")