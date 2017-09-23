import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam, Nadam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from keras.layers import Lambda, Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Dense, Flatten, ELU, Activation, BatchNormalization, Cropping2D, ZeroPadding2D

import os
import config as cf

def get_nvidia_model():
    model = Sequential()

    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=cf.INPUT_SHAPE))
    # We have a series of 3 5x5 convolutional layers with a stride of 2x2
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(BatchNormalization())

    # This is then followed by 2 3x3 convolutional layers with a 1x1 stride
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(BatchNormalization())

    # Flattening the output of last convolutional layer before entering fully connected phase
    model.add(Flatten())

    # Fully connected layers
    # model.add(Dense(1164, activation='relu'))
    # model.add(BatchNormalization())

    model.add(Dense(100, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(50, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(10, activation='relu'))
    model.add(BatchNormalization())

    # Output layer
    model.add(Dense(1))

    model.compile(loss = "mse", optimizer = Adam(lr = 0.001))
    model.summary()

    return model
