#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/model.png "Model Visualization"
[image2]: ./images/my_conv_net.png "My Convnet architecture"
[image3]: ./images/mse.png "Model Loss"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* video.mp4 autonomous drive for at least one lap on track one

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My initial approach was to use the simple model proposed in Lesson 7 (Lambda + Flatten layers), but the car was driving in circles. Next, I tried LeNet. Accuracy and loss weren't good. (see section *Solution Design Approach*).
Finally, I decided to try the nVidia Autonomous Car Group model:

![alt text][image2]

#### 2. Attempts to reduce overfitting in the model

I decided to set only three to four epochs and increase the data set with lots of data, since I removed Dropouts and Maxpooling which at the beginning were making my model keep having high loss rate. 
After that, the model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 141).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used center, left and right cameras from track one and two. I also used techniques such as flipping image and normalization. Images from every position combined with these techniques were used to train the model.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My initial approach was to use the simple model proposed in Project Behavioral Cloning - Lesson 7, but the car was driving in cirles. Next, I tried LeNet, this time the car drove until the beginning of the first corner without getting of the track. Accuracy and loss weren't good, so I tried increasing epochs but the model started overfitting. I added dropout models but couldn't get better results. After this, I decided to try the nVidia Autonomous Car Group model with Dropout between each Convolutional model pair. Results were a little bit better, but after the first corner, the car drove into the lake. and the car drove the complete first track. Then I removed the dropout layers and decide to increase the number of images for my train and validation set using left and right camera images with 0.2 correction in steering to make them as center images. 
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 31-48) consisted of a convolution neural network with the following layers and layer sizes (based on nVidia Autonomous Car Group model:
![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior I used a PS4 gamepad. Its key pressure sensitivy helped very much. Laptop's keyboard was doing no good with steering precision. 
I first recorded one and half lap on track one using center lane driving. Then, I recorded recovery in each special part of the track (on the track near grass, during the curve near the lake, near the bridge, and where the side of the track was was striped white/red). To make it even better after completting the deep neural network pipeline, I tried getting data driving one lap on the second track. To augment the data set, I also flipped images and angles. The model was trained with shuffled data during 3 to 4 epochs.

After that, the data set was shuffled randomly and 20% of the data was put into a validation set. 
The validation set helped determine if the model was over or under fitting. The ideal number of epochs was between 3 to 4 as evidenced by the data set/validation set loss chart down below. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][image3]
