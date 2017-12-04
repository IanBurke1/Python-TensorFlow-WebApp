# A Digit Recognition Web Application
#### *Emerging Technologies Module - Lecturer: [Dr Ian McLoughlin](ianmcloughlin.github.io) - 4th Year Software Development*
For my project in [Emerging Technologies](https://emerging-technologies.github.io/), I am required to create a web application in [Python](https://www.python.org/) to recognise digits in images. Users will be able to visit the web application through their browser, submit (or draw) an image containing a single digit, and the web application will respond with the digit contained in the image. [Flask](http://flask.pocoo.org/) will be used to run the web application and [Keras](https://keras.io/) will be used to help with the digit recognition.

**_For more: [Project Instructions](https://emerging-technologies.github.io/problems/project.html)_**

## How to run 
1. Click [here](https://github.com/ianburkeixiv/Python-TensorFlow-WebApp/archive/master.zip) to download the zip of the project.
2. Unzip the project.
3. Open a command terminal and cd into project directory
4. Enter the following to run the app:

```python

python app.py

```
## Architecture

## Python 
[![PyPI](https://img.shields.io/pypi/pyversions/Django.svg)]()

The main programming language used in this problem sheet is [Python](https://www.python.org/)

## Flask
[Flask](http://flask.pocoo.org/) is a Python micro web framework that provides tools, libraries and technologies that allow us to build a web application. 

## Numpy
[NumPy](http://www.numpy.org/) is a library for the [Python](https://www.python.org/) programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays. It is the fundamental package for scientific computing with Python.

## TensorFlow
![](https://user-images.githubusercontent.com/22341150/33095338-3573a9cc-cefb-11e7-9030-42e3f298e0b7.png)

[TensorFlow](https://www.tensorflow.org/) is a Python open source library for fast numerical computing created and released by Google. It is used for machine learning applications such as [neural networks](https://en.wikipedia.org/wiki/Artificial_neural_network). It allows users to express arbitary computation as a graph of data flows. Nodes in this graph represent mathematical operations and the edges represent data that is communicated from one node to another. Data in TensorFlow are represented as tensors which are multidimensional arrays. 

## Keras
![](https://user-images.githubusercontent.com/22341150/33095362-4cf67246-cefb-11e7-87e5-cad404557eec.png)

[Keras](https://keras.io/) is as high level neural network API running on top of TensorFlow. Designed to enable fast and easy experimentation with deep neural networks. Keras is more minimal than TensorFlow as it runs seamlessly on CPU and GPU. It allows for fast prototyping due to its user friendliness.

### Neural Networks
In each hemisphere of our brain, humans have a primary visual cortex, also known as V1, containing 140 million neurons, with tens of billions of connections between them. And yet human vision involves not just V1, but an entire series of visual cortices - V2, V3, V4, and V5 - doing progressively more complex image processing. We carry in our heads a supercomputer, tuned by evolution over hundreds of millions of years, and superbly adapted to understand the visual world. Recognizing handwritten digits isn't easy. We humans are astoundingly good at making sense of what our eyes show us. But nearly all that work is done unconsciously. And so we don't usually appreciate how tough a problem our visual systems solve. Adapted from: http://neuralnetworksanddeeplearning.com/chap1.html

![](http://neuralnetworksanddeeplearning.com/images/tikz11.png)

#### Artificial Neural Networks
*_[Artificial Neural Network](https://en.wikipedia.org/wiki/Artificial_neural_network)_* is a computational model that is inspired by the way biological neural networks in the human brain process information. Artificial Neural Networks have generated a lot of excitement in Machine Learning research and industry. The basic unit of computation in a neural network is the neuron, often called a node or unit. It receives input from some other nodes, or from an external source and computes an output. Each input has an associated weight (w), which is assigned on the basis of its relative importance to other inputs.

![](https://ujwlkarn.files.wordpress.com/2016/08/screen-shot-2016-08-09-at-3-42-21-am.png?w=768&h=410)

### Convolutional Neural Network (CNN)
[Convolutional Neural Networks (CNN)](http://cs231n.github.io/convolutional-networks/) are multi-layer neural networks that have successfully been applied to analyzing visual imagery. The main feature of CNN's is that it can drastically reduce the number of parameters that need to be tuned. This means that CNN's can efficiently handle the high dimensionality of raw images.

Convolutional neural networks are more complex than standard multi-layer perceptrons. In this project, we use 8 layers.
1. The first hidden layer is a convolutional layer called a Convolution2D. In our model, the layer has 32 filters, which with the size of 3×3 and a rectified linear unit activation function. This is the input layer, expecting images with the structure outline (pixels, width, height). 
2. We add another covolutional layer with 64 filters.  
3. Next we define a pooling layer called MaxPooling2D which is a way to reduce the number of parameters in our model. It is configured by sliding a 2x2 pooling filter across the previous layer and taking the max of the 4 values in the 2x2 filter.
4. The next layer is a regularization layer using dropout called Dropout. It is configured to randomly exclude 25% of neurons in the layer in order to reduce overfitting. 
5. Next is a layer that converts the 2D matrix data to a vector called Flatten. It allows the output to be processed by standard fully connected layers.
6. Next a fully connected layer with 128 neurons and rectified linear unit activation function. 
8. Finally, the output layer has 10 neurons for the 10 classes and a softmax activation function to output probability-like predictions for each class.

## Conclusion
Using flask, we have published a web app that displays a canvas where a user can draw a digit inside the canvas with their mouse. The canvas data is then converted into an appropiate image to pass into our loaded model which predicts the digit and returns the result to the user. I have used a convolutional neural network which is renowned as the most accurate with a low loss rate.

Video: [![Neural Network 3D Simulation](https://img.youtube.com/vi/https://www.youtube.com/watch?v=3JQ3hYko51Y/0.jpg)](https://www.youtube.com/watch?v=3JQ3hYko51Y)

## MNIST Dataset
[MNIST]( http://yann.lecun.com/exdb/mnist/) is a famous dataset that consists of handwritten digits commonly used for training various image processing systems and also used in machine learning. The dataset contains 60,000 training images and 10,000 testing images. Each image is a 28x28 pixel square (784 pixels in total). A standard split of the dataset is used to evaluate and compare models. Excellent results achieve a prediction error/loss of less than 1% and accuracy of up to 0.99%.

![](https://www.tensorflow.org/images/mnist_digits.png)

## References
[The MNIST Database](http://yann.lecun.com/exdb/mnist/)

[CNN Tutorial](https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/)

[Keras tutorial](https://elitedatascience.com/keras-tutorial-deep-learning-in-python)

[Deep learning](http://neuralnetworksanddeeplearning.com/chap1.html)
