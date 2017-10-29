# CarND-Behavioral-Cloning-P3
## Udacity Self Driving Car Nanodegree - Behavioral Cloning

Requirements

You need:
1) Python3
2) Miniconda
3) Using Miniconda you can install OpenCv, Tensorflow and Keras.
4) [Graphviz][http://www.graphviz.org/] (Used to generate image files)

1) sudo apt-get install graphviz
or
2) conda install graphviz
or
3) pip3 install graphviz

In Ubuntu 16.04, I had to run '1' and '3'. If you go directly to step 2 or 3, it might fail because one of its dependencies is missing.

# Overview

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

We have provided a simulator where you can steer a car around a track for data collection. You'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

We also want you to create a detailed writeup of the project. Check out the writeup template for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting five files:

model.py (script used to create and train the model)
drive.py (script to drive the car - feel free to modify this file)
model.h5 (a trained Keras model)
a report writeup file (either markdown or pdf)
video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)
