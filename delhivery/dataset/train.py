import numpy
seed = 7
numpy.random.seed(seed)
from keras.utils.visualize_util import plot
from keras.datasets import mnist
from keras.models import Sequential,    Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling2D, AveragePooling2D
from keras.layers import Input
from keras.utils import np_utils
from keras import backend as K
from keras.layers.normalization import BatchNormalization
K.set_image_dim_ordering('th')
from keras.callbacks import ModelCheckpoint , History
from keras.metrics import top_k_categorical_accuracy
import numpy as np
from inception_res_v1 import Stem, Inception_A, Inception_B, Inception_C, Reduction_A, Reduction_B
import logging
import matplotlib.pyplot as plt


##################################################################################################
weightsOutputFile = 'Digit_classifier.{epoch:02d}-{val_loss:.3f}.hdf5'
img_rows, img_cols = 28, 28
nb_filters_reduction_factor = 8
channel_axis = -1
img_channels = 1
k = 256
l = 256
m = 384
n = 384
num_A_blocks = 5
num_B_blocks = 10
num_C_blocks = 5
nb_classes = 10
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

inputs = Input(shape=(img_rows, img_cols, img_channels))
x = Stem(inputs)

for i in range(num_A_blocks):
    x = Inception_A(x)
x = Reduction_A(x)

for i in range(num_B_blocks):
    x = Inception_B(x)
x = Reduction_B(x)

for i in range(num_C_blocks):
    x = Inception_C(x)


x = AveragePooling2D(pool_size=(3, 3), strides=(
    1, 1), border_mode='valid', dim_ordering='tf')(x)

x = Dropout(0.2)(x)
x = Flatten()(x)
predictions = Dense(10, activation='relu')(x)
predictions = Dense(nb_classes, activation='softmax')(predictions)
model = Model(input=inputs, output=predictions)
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
checkpointer = ModelCheckpoint(
    weightsOutputFile, monitor='val_loss', save_best_only=False, mode='auto')

history=History()
model.fit(X_train, y_train, batch_size=50, nb_epoch=20, verbose=1,
          validation_data=(X_test, y_test), callbacks=[checkpointer,history])

plot(model, to_file='model.png')
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
