import csv
import cv2
import numpy as np
from sklearn.utils import shuffle

ch = 3
row = 160
col = 320

def generator(samples, batch_size=64):
    num_samples = len(samples)
    shuffle(samples)
    while 1:
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
        
            images = []
            angles = []
            for imagePath, measurement in batch_samples:
                image = cv2.imread(imagePath)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)
                angles.append(measurement)
                images.append(cv2.flip(image,1))
                angles.append(measurement*-1.0)

            X_data = np.array(images)
            y_data = np.array(angles)
            yield shuffle(X_data, y_data)

def nVidiaModel():
    """
    Creates nVidea Autonomous Car Group model
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.) - 0.5, input_shape=(row,col,ch)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

def nVidiaModifiedModel():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.) - 0.5, input_shape=(row,col,ch)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    model.add(Convolution2D(24, 5, 5, border_mode="same", subsample=(2,2), activation="elu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(36, 5, 5, border_mode="same", subsample=(2,2), activation="elu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(48, 5, 5, border_mode="valid", subsample=(2,2), activation="elu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="elu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="elu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation="elu"))
    model.add(Dense(50, activation="elu"))
    model.add(Dense(10, activation="elu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))

def MyTestModel():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0, input_shape=(160,320,3)))
    model.add(Flatten())
    model.add(Dense(1))
    #LeNet
    model = Sequential()
    model.add(Convolution2D(6,5,5,activation="relu", input_shape=(160,320,3)))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

lines = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)  

images = []
measurements = []
correction = 0.2
common_path = '../data/IMG/'

for line in lines:
    center_path = line[0]
    filename = center_path.split('/')[-1]
    center_img_file = common_path + filename

    left_path = line[1]
    filename = left_path.split('/')[-1]
    left_img_file = common_path + filename

    right_path = line[2]
    filename = right_path.split('/')[-1]
    right_img_file = common_path + filename

    steering_center = float(line[3])
    steering_left = float(line[3]) + correction
    steering_right = float(line[3]) - correction

    images.append(center_img_file)
    images.append(left_img_file)
    images.append(right_img_file)
    measurements.append(steering_center)
    measurements.append(steering_left)
    measurements.append(steering_right)

# Splitting samples and creating generators.
from sklearn.model_selection import train_test_split
samples = list(zip(images, measurements))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, ELU, Dropout, SpatialDropout2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Activation
from keras.layers import Cropping2D


model = nVidiaModel()
model.compile(loss='mse', optimizer='adam')
# history_info = model.fit(X_train, y_train, validation_split=0.20, shuffle=True, nb_epoch=10)

history_info = model.fit_generator(train_generator, samples_per_epoch= \
            len(train_samples), validation_data=validation_generator, \
            nb_val_samples=len(validation_samples), nb_epoch=4)

model.save('model.h5')

print(history_info.history.keys())
print('Loss')
print(history_info.history['loss'])
print('Validation Loss')
print(history_info.history['val_loss'])
