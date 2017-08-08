import numpy as np
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
checkpointer = ModelCheckpoint(
    weightsOutputFile, monitor='val_precision', save_best_only=False, mode='auto')

history=History()
model.fit(x_train, y_train, batch_size=10, nb_epoch=20, verbose=1,
          validation_split=0.2,shuffle=True, callbacks=[checkpointer,history])

logging.basicConfig(filename='example.log',level=logging.INFO)
logging.info(history.history)
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
