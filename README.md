# LeNet-MNIST-Classification
Mnist dataset classification with LeNet (98 percent accuracy)
## Mnist Dataset
First of all i want talk about our dataset.here we use mnist datase and i think all of you at least heard about it. this is hello world dataset to machine learning and deep learning.even for fully conected networks you can use mnist and get good accuracy on it.

<p align="center">
  <img src="https://aizawan.github.io/das/img/app.gif">
</p>

In this data set we have 70000 images and their labels. we can use 60000 of them for learning and use 10000 of them for our test set and validation.
this dataset had 1 channel and 28*28 pixel.

## LeNet
LeNet is first convolutional neural network that developed by Yann LeCun.you can read full paper 
[HERE](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf).But here we try to intoduce the architect of LeNet simply.first of all i must say in main paper for activation function Le used tanh but we use ReLU becuase it have better result and far from gradient vanishing.
The structure of this network is like this:
```
INPUT => CONV => RELU => POOL => CONV => RELU => POOL => FC => RELU => FC
```
Thre convolution layer with relu activation function and pooling layer then classification fully connected layer.

## Describing Code
This code have 5 file.3 of them are main codes and 2 of them are model saved informations.First we define our CNN in <LeNet.py> in subclass type.then train with mnist dataset in <LeNet mnist.py> and save it.and finally use saved model in <LeNet_predict.py> for predict some of images and show them with opencv library.before saying bye i must say you can read comments in code for better informations and you can load model weights for predicting before training.
