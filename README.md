# Keras & TensorFlow Datasets Example

This example shows how [Keras](https://keras.io/) and [TensorFlow Datasets](https://github.com/tensorflow/datasets) can be used together to train a simple [MNIST](http://yann.lecun.com/exdb/mnist/) classifier to >98% test set accuracy using the classic [LeNet-5](http://yann.lecun.com/exdb/lenet/) [LeCun et al., 1998] convolutional neural network architecture.

This project uses [Pantsbuild](https://www.pantsbuild.org/). To setup, build, and run simply use:

```bash
./pants run src/python:keras-tfds-example
```
