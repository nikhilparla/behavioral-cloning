import pandas as pd
import cv2
import numpy as np
import math

df = pd.read_csv('../train_data/driving_log.csv')

from sklearn.model_selection import train_test_split
import sklearn

samples = []  # empty 2d list to hold the names of all the centre camera images and steering angles
number = 0

for row in range(df.shape[0]):
    #print('row = ', row)
    # get the image name after splitting the line.
    # img is the last value so using -1
    centre_img_name = df['center'][row].split("/")[-1]
    left_img_name = df['left'][row].split("/")[-1]
    right_img_name = df['right'][row].split("/")[-1]
    centre_angle = (df['steering'][row])
#     if(centre_angle == 0):
#         number+=1
#     if(number % 10 ==0):
#         pass
#     else:
    samples.append(["../train_data/IMG/"+ centre_img_name, centre_angle, "../train_data/IMG/"+ left_img_name,"../train_data/IMG/"+ right_img_name ])

# import random
# def remove(l,n):
#     return random.sample(l,int(len(l)*(1-n)))

# print('before random deletion', len(samples))
# samples = remove(samples,0.25)
# print('after random deletion', len(samples))

#import matplotlib.pyplot as plt
#plt.hist([i[1] for i in samples], bins=20)
#plt.show()
#train_samples, validation_samples = train_test_split(samples,test_size = 0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                centre_image = cv2.cvtColor(cv2.imread(batch_sample[0]), cv2.COLOR_BGR2RGB)
                left_image = cv2.cvtColor(cv2.imread(batch_sample[2]), cv2.COLOR_BGR2RGB)
                right_image = cv2.cvtColor(cv2.imread(batch_sample[3]), cv2.COLOR_BGR2RGB)
                
                centre_angle = float(batch_sample[1])
                left_angle = float(batch_sample[1] + 0.3)
                right_angle = float(batch_sample[1] - 0.3)
                
                flipped_image = cv2.flip(centre_image,1)
                centre_angle_flipped = -1 * centre_angle
                
                images.append(centre_image)
                measurements.append(centre_angle)
                images.append(left_image)
                measurements.append(left_angle)
                images.append(right_image)
                measurements.append(right_angle)
                images.append(flipped_image)
                measurements.append(centre_angle_flipped)

            X_train = np.array(images)
            y_train = np.array(measurements)
#             print(np.shape(images[2]))
#             print(np.shape(X_train))
#             print(np.shape(y_train))
            yield sklearn.utils.shuffle(X_train, y_train)

batch_size = 8 # 32 overall
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# Most basic network - flattend image conencted to a single output node
# The single output node will predict the steering angle which makes this a regression network
# For classification network we might apply a softmax activation function to op layer
# Since this is regression, the single output node directly predicts the steering measurement
# So not activation function here
# For loss function, we use mean squared error, mse instead of the cross-entroy function,
# again because this is regressioon and not classsification
# Basically this is to minimize the error bw the predicted steering and ground truth steering
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.3))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
print(model.summary())
model.compile(loss='mse', optimizer='adam')

from keras.models import load_model
model = load_model('model.h5')

model.fit_generator(train_generator, 
                    steps_per_epoch=math.ceil(len(train_samples)/batch_size), 
                    validation_data=validation_generator, 
                    validation_steps=math.ceil(len(validation_samples)/batch_size), 
                    epochs=3, verbose=1)
model.save('model.h5')
exit()
