import keras as kr
from keras.datasets import mnist # automatically downloaded once function is called.
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt

#
# Adapted from: https://github.com/wxs/keras-mnist-tutorial/blob/master/MNIST%20in%20Keras.ipynb
#


# Load the MNIST dataset from keras.datasets -> Dataset of 60,000 28x28 grayscale images of 10 digits & test set of 10,000 images
# 2 tuples ( immutable data structure/lists consisting of mulitple parts):
# x_train, x_test = Uint8 array of grayscale image data with shape (num_samples, 28,28)
# y_train, y_test = Uint8 array of digit labels (integers in range 0-9) with shape(num_samples)
# Adapted from: https://keras.io/datasets/
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Format/reshape the data for training
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#print("Training matrix shape", x_train.shape)
#print("Testing matrix shape", x_test.shape)

# Convert the data to binary matrices
y_train = kr.utils.to_categorical(y_train, num_classes=10)
y_test = kr.utils.to_categorical(y_test, num_classes=10)


# Create our model
model = Sequential()

model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))

model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))


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

# parse/serialise model to json format
model_json = model.to_json()
with open("mnistModel.json", "w") as json_file:
    json_file.write(model_json)

# Save the model 
# h5 is the file format for keras
model.save("mnistModel.h5")