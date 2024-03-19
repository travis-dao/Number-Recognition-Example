# Number-Recognition-Example
An example convolution network ([LeNet](https://en.wikipedia.org/wiki/LeNet)) for classifying numbers (0-9) through TensorFlow.

![LeNet reference](https://upload.wikimedia.org/wikipedia/commons/thumb/c/cc/Comparison_image_neural_networks.svg/480px-Comparison_image_neural_networks.svg.png)

Tensorflow has a built-in number dataset for usage. Import tensorflow_datasets and load the "mnist" dataset. To create your model, use the base Sequential model from Keras (part of Tensorflow). The code also exports the model into a .keras file for external usage.

If you are confused with using Tensorflow, you can read the [documentation](https://www.tensorflow.org/api_docs). [Python Machine Learning](https://drive.google.com/file/d/1LHkjN1eSgomkU1C9yiUaMbFIrL2lD9bB/view?usp=sharing) is an amazing book to understand the fundamentals of using Python (Tensorflow) for AI and machine learning.

MatPlotLib is also used to graph the accuracy of the models over multiple epochs.
