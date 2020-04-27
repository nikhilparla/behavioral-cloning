# Behavioral Cloning Project

### **My Notes**

Train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

Provided a simulator where you can steer a car around a track for data collection. Use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

###### **Files**

* model.py (script used to create and train the model)
* drive.py (script to drive the car)
* model.h5 (a trained Keras model)
* video.py (create video from images)

###### The Project

* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.

###### Progress
* Used pandas library for parsing driving log csv file
* model.py has cv2 calls for visualizing images from driving log
* Populated the images and measurements array with the images from centre camera and steering angle
* Added initial model and trained according to instructions from the project notes.
  * Model trained to 20 epochs, saved to h5 file and used and trained.
  * loss: 1752.2630 - val_loss: 2690.2254
* drive.py able to load model from h5 and drive the car. All connections fine. 
* Added a lambda layer for normalizing values dividing by 255.0
  * Trained to 15 epochs
  * loss: 0.0463 - val_loss: 0.0594
  * Much better performance than the non-normalzed model when driven autonomously
* Normalized values to 0 by sbtracting 0.5
  * 10 epochs
  * loss: 5.6755 - val_loss: 4.1779
  * Not good performance
* Lenet architecture implemented
  * 10 epochs
  * loss: 0.0163 - val_loss: 0.0115
  * Much cleaner driving, but not enough data for performance
* Data augmentation by flipping the centre images
  * 10 epochs
  * loss: 0.0250 - val_loss: 0.0184\
  * Car travelled for almost half the distance with no issues.
* Added right and left camera images
  * 10 epochs
  * loss: 0.0072 - val_loss: 0.0315
  * I see that it is very capable of correcting itslef at the corners but still goes oit of the path
* Cropped top 75 pixels and bottom 25
  * 10 epochs
  * loss: 0.0234 - val_loss: 0.0340
  * Better performace at curves
* NVDIA autodriving team model
  * 10 epochs
  * loss 0.0099 
  * Less performace than lenet in my case
* Collected more data with corner cases, removed flipping and left, right images
  * total 8000 images with clockwise and counter clockwise driving training
  * Same NVIDIA architecture, 10 epochs
  * Collected some scenarious driving back when water encountered
  * 0.0203 - val_loss: 0.2632
  * Runs, but eventually falls off the path.
* Need more data, implementing generator
* Implemented generator for centre images
  * No flipping the images in the round, just testing the basic generator
  * loss: 0.0305 - val_loss: 0.0609
  * Decent performace but fell into water
* Got data from the other track 
  * More validation loss. Over fitting.
  * Reduced epocs to 8
  * Track completed.
###### Issues
* Training the model immediately after collecting data doent seem to work since the 
gpu doesnt seem to release free memory. Of the 12 GB available, only 400 MB is shown as 
free.Restarting the workspace solves it.


---------------------------------------------------

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---

This repository contains starting files for the Behavioral Cloning Project.

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

We have provided a simulator where you can steer a car around a track for data collection. You'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting five files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

This README file describes how to output the video in the "Details About Files In This Directory" section.

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/432/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).

### Tips
- Please keep in mind that training images are loaded in BGR colorspace using cv2 while drive.py load images in RGB to predict the steering angles.

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

