import pandas as pd
import cv2
import numpy as np

cv2.namedWindow("Input")

df = pd.read_csv('./train_data/driving_log.csv')

images = []
measurements = []

def get_image(img_name):
    #print("./train_data/IMG/"+ img_name)
    image = cv2.imread("./train_data/IMG/"+ img_name)
    images.append(image)
"""    
    #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    cv2.imshow("Input",image)
    cv2.waitKey(33)
"""

def get_measurements(meas, flag):
    if(flag == 'center'):
        measurements.append(meas)
    else if(flag == 'right'):
        measurements.append(meas- 0.2)
    else if(flag == 'left'):
        measurements.append(meas + 0.2)

for row in range(df.shape[0]):
    #print('row = ', row)
    # get the image name after splitting the line. 
    # img is the last value so using -1
    #print(df['center'][row].split("/")[-1])
    get_image(df['center'][row].split("/")[-1])
    get_measurements(df['steering'][row], 'center')
    get_image(df['right'][row].split("/")[-1])
    get_measurements(df['steering'][row], 'right')
    get_image(df['left'][row].split("/")[-1])
    get_measurements(df['steering'][row], 'left')

augmented_images, augmented_measurements = [],[]
for image,meas in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(meas)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(meas*-1.0)

print('Images length', len(augmented_images))
print('measurements length', len(augmented_measurements))

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)
print('images shape = ',np.shape(augmented_images))

from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda
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
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2,shuffle=True, nb_epoch=15)

model.save('model.h5')
exit()
