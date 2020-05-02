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
* Randomly deleting 20 percent of data where steering angle is 0
  * The zero angles are too many in the training data so reducing them by not adding every 5th element into the list
  * Performance bad
* Added dropout layers to prevent overfitting
  * dropout layer with dropout rate 50% added after the dense layers
  * Better performance
* Added one more track of data for track 2 driving counterclockwise
  * track2 completed autonomously


###### Issues
* Training the model immediately after collecting data doent seem to work since the 
gpu doesnt seem to release free memory. Of the 12 GB available, only 400 MB is shown as 
free.Restarting the workspace solves it.
  * I think this is because of the GPU loading all the images into memory since we arent using a co-routine.Did not see this issue after the genertor function has been added
  * Also sometimes if the training has been stopped midway wtih ctrl-c, the application doesnt stop and needs to forcefully killed
  * ```$ ps -ef | grep python```
