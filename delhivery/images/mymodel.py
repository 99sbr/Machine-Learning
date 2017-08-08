import numpy as np
import logging
from keras.datasets import mnist
from keras.layers.normalization import BatchNormalization
from keras.utils.visualize_util import plot
from keras.models import Sequential, Model
from keras.layers import Dense, Lambda
from keras.layers import Dropout
from keras.layers import Flatten, Activation
from keras.layers.convolutional import Convolution2D, SeparableConvolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling2D, AveragePooling2D
from keras.layers import Input, merge
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint,History
from keras import backend as K
from keras.utils.layer_utils import layer_from_config
from new import x_train,y_train
K.set_image_dim_ordering('tf')
import model_functions as Model_Blocks
weightsOutputFile = 'inception.{epoch:02d}-{val_precision:.3f}.hdf5'

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)
print(X_test.shape)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
X_train = X_train / 255
X_test = X_test / 255
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

def CNN_Model():
    model = Sequential()
    
    model.add(Convolution2D(34, 5, 5,border_mode='valid', input_shape=(150,150,3),name='conv1_1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(34, 5, 5,border_mode='valid', name='conv1_2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(75, 3, 3, border_mode='valid', name='conv2_1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(75, 3, 3,border_mode='valid', name='conv2_2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(75, 3, 3, border_mode='valid', name='conv2_3'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(75, 3, 3,border_mode='valid', name='conv2_4'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(150, 1, 1,border_mode='valid', name='conv3_1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(150, 1, 1,border_mode='valid', name='conv3_2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))


    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(5))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    return model 


x_train = x_train / 255
y_train = np_utils.to_categorical(y_train)

img_rows, img_cols = 150, 150
img_channels = 3
Input = Input(shape=(img_rows, img_cols, img_channels))
model_number = 3
nb_classes = 5
model = CNN_Model()
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['precision'])



layer_name = 'conv3_2'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(X_train[0])