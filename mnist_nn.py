import keras as kr
from keras.datasets import mnist # automatically downloaded once function is called.
from keras.models import Sequential


# Load the MNIST dataset from keras.datasets -> Dataset of 60,000 28x28 grayscale images of 10 digits & test set of 10,000 images
# 2 tuples ( immutable data structure/lists consisting of mulitple parts):
# x_train, x_test = Uint8 array of grayscale image data with shape (num_samples, 28,28)
# y_train, y_test = Uint8 array of digit labels (integers in range 0-9) with shape(num_samples)
# Adapted from: https://keras.io/datasets/
(X_train, y_train), (X_test, y_test) = mnist.load_data()




# Split up and shuffle the data

# Convert the data to binary matrices
y_train = kr.utils.to_categorical(y_train, num_classes=None)
y_test = kr.utils.to_categorical(y_test, num_classes=None)


# Create our model
model = Sequential()

# Train the model

# Test the model
 
# Prediction

# Save the model 
# h5 is the file format for keras