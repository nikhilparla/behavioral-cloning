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

def get_measurements(meas):
    #print(meas)
    measurements.append(meas)

for row in range(df.shape[0]):
    #print('row = ', row)
    # get the image name after splitting the line. 
    # img is the last value so using -1
    #print(df['center'][row].split("/")[-1])
    get_image(df['center'][row].split("/")[-1])
    get_measurements(df['steering'][row])

print('Images length', len(images))
print('measurements length', len(measurements))

X_train = np.array(images)
y_train = np.array(measurements)
print('images shape = ',np.shape(images))

from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda

# Most basic network - flattend image conencted to a single output node
# The single output node will predict the steering angle which makes this a regression network
# For classification network we might apply a softmax activation function to op layer
# Since this is regression, the single output node directly predicts the steering measurement
# So not activation function here
# For loss function, we use mean squared error, mse instead of the cross-entroy function, 
# again because this is regressioon and not classsification
# Basically this is to minimize the error bw the predicted steering and ground truth steering
model = Sequential()
model.add(Lambda(lambda x: x/255.0, input_shape=(160,320,3)))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2,shuffle=True, nb_epoch=15)

model.save('model.h5')
exit()
