import pandas as pd
import cv2

cv2.namedWindow("Input")

df = pd.read_csv('./train_data/driving_log.csv')


def get_image(img_name):
    print("./train_data/IMG/"+ img_name)
    image = cv2.imread("./train_data/IMG/"+ img_name)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    cv2.imshow("Input",image)
    cv2.waitKey(33)


for row in range(df.shape[0]):
    #print('row = ', row)
    # get the image name after splitting the line. 
    # img is the last value so using -1
    #print(df['center'][row].split("/")[-1])
    get_image(df['center'][row].split("/")[-1])

