import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import merge, Input
from keras.utils import np_utils
from keras.models import Model, Sequential
from keras import backend as K
K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


def inceptionModule(input_img):
    input_img = MaxPooling2D((2, 2), strides=(1, 1), border_mode='same')(input_img)
    tower_1 = Convolution2D(30, 1, 1, border_mode='same', activation='relu')(input_img)
    tower_2 = Convolution2D(30, 3, 3, border_mode='same', activation='relu')(tower_1)
    tower_3 = Convolution2D(30, 5, 5, border_mode='same', activation='relu')(tower_1)
    tower_4 = MaxPooling2D((3, 3), strides=(1, 1), border_mode='same')(input_img)
    tower_4 = Convolution2D(15, 1, 1, border_mode='same', activation='relu')(tower_4)
    output = merge([tower_1,tower_2, tower_3,tower_4], mode='concat', concat_axis=1)
    return output

inputs=Input(shape=(1,28,28))
fire1=inceptionModule(inputs)
fire2=Dropout(0.4)(fire1)
fire3=Flatten()(fire2)
pred1=Dense(4*num_classes, activation='relu')(fire3)
pred2=Dense(2*num_classes, activation='relu')(pred1)
pred3=Dense(num_classes, activation='softmax')(pred2)
model=Model(input=inputs,output=pred3)
model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test),
          nb_epoch=10, batch_size=200, verbose=2)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))
