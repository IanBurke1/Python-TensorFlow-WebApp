# A Digit Recognition Web Application
#### *Emerging Technologies Module - Lecturer: [Dr Ian McLoughlin](ianmcloughlin.github.io) - 4th Year Software Development*
For my project in [Emerging Technologies](https://emerging-technologies.github.io/), I am required to create a web application in [Python](https://www.python.org/) to recognise digits in images. Users will be able to visit the web application through their browser, submit (or draw) an image containing a single digit, and the web application will respond with the digit contained in the image. [Flask](http://flask.pocoo.org/) will be used to run the web application and [TensorFlow](https://www.tensorflow.org/) will be used to help with the digit recognition.

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
[Artificial Neural Network](https://en.wikipedia.org/wiki/Artificial_neural_network) is a computational model that is inspired by the way biological neural networks in the human brain process information. Artificial Neural Networks have generated a lot of excitement in Machine Learning research and industry. The basic unit of computation in a neural network is the neuron, often called a node or unit. It receives input from some other nodes, or from an external source and computes an output. Each input has an associated weight (w), which is assigned on the basis of its relative importance to other inputs.

![](https://ujwlkarn.files.wordpress.com/2016/08/screen-shot-2016-08-09-at-3-42-21-am.png?w=768&h=410)

### Convultional Neural Network (CNN)
[Convolutional Neural Networks (CNN)](http://cs231n.github.io/convolutional-networks/) are multi-layer neural networks that have successfully been applied to analyzing visual imagery. The main feature of CNN's is that it can drastically reduce the number of parameters that need to be tuned. This means that CNN's can efficiently handle the high dimensionality of raw images.

## MNIST Dataset
[MNIST]( http://yann.lecun.com/exdb/mnist/) is a famous dataset that consists of handwritten digits commonly used for training various image processing systems and also used in machine learning. The dataset contains 60,000 training images and 10,000 testing images. Each image is a 28x28 pixel square (784 pixels in total). A standard split of the dataset is used to evaluate and compare models. Excellent results achieve a prediction error/loss of less than 1% and accuracy of up to 0.99%.

![](https://www.tensorflow.org/images/mnist_digits.png)


## Conclusion
In my keras model, I have implemented a convoltional neural network which has the best loss and accuracy rate of all other machine learning methods known. Using 4 epochs, I have achieved a loss of 0.0324 and an accuracy of 0.9906. I would have achieved a better accuracy if I had increased the number of epochs.  

## References
- [The MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [Tutorial](https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/)
- [Keras tutorial](https://elitedatascience.com/keras-tutorial-deep-learning-in-python)