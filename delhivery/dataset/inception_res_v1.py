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
from keras import backend as K
from keras.utils.layer_utils import layer_from_config
K.set_image_dim_ordering('tf')


def Stem(Input):
    path1 = Convolution2D(32 // nb_filters_reduction_factor, 3, 3,
                          subsample=(2, 2), border_mode='valid', activation='relu')(Input)
    path1 = Convolution2D(32 // nb_filters_reduction_factor,
                          3, 3, border_mode='same', activation='relu')(path1)
    path1 = Convolution2D(64 // nb_filters_reduction_factor,
                          3, 3, border_mode='same', activation='relu')(path1)
    path1 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(path1)
    path1 = Convolution2D(80 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(path1)
    path1 = Convolution2D(192 // nb_filters_reduction_factor,
                          3, 3, border_mode='same', activation='relu')(path1)
    path1 = Convolution2D(256 // nb_filters_reduction_factor, 3, 3,
                          subsample=(2, 2), border_mode='valid', activation='relu')(path1)
    path1=BatchNormalization()(path1)
    path1=Activation('relu')(path1)
    return path1


def Inception_A(Input):
    path1 = Convolution2D(32 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)
    path1 = Convolution2D(32 // nb_filters_reduction_factor,
                          3, 3, border_mode='same', activation='relu')(path1)
    path1 = Convolution2D(32 // nb_filters_reduction_factor,
                          3, 3, border_mode='same', activation='relu')(path1)

    path2 = Convolution2D(32 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)
    path2 = Convolution2D(32 // nb_filters_reduction_factor,
                          3, 3, border_mode='same', activation='relu')(path2)

    path3 = Convolution2D(32 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)

    out = merge([path1, path2, path3], mode='concat', concat_axis=-1)
    out = Convolution2D(256 // nb_filters_reduction_factor,
                        1, 1, border_mode='same', activation='linear')(out)

    output = merge([out, Input], mode='sum')
    output=BatchNormalization()(output)
    output = Activation('relu')(output)
    return output


def Reduction_A(Input):
    path1 = Convolution2D(k // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)
    path1 = Convolution2D(l // nb_filters_reduction_factor,
                          3, 3, border_mode='same', activation='relu')(path1)
    path1 = Convolution2D(m // nb_filters_reduction_factor, 3, 3,
                          subsample=(2, 2), border_mode='valid', activation='relu')(path1)

    path2 = Convolution2D(n // nb_filters_reduction_factor, 3, 3,
                          subsample=(2, 2), border_mode='valid', activation='relu')(Input)

    path3 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(Input)

    output = merge([path1, path2, path3], mode='concat', concat_axis=-1)
    output=BatchNormalization()(output)
    output=Activation('relu')(output)
    return output


def Inception_B(Input):
    path1 = Convolution2D(128 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)
    path1 = Convolution2D(128 // nb_filters_reduction_factor,
                          1, 7, border_mode='same', activation='relu')(path1)
    path1 = Convolution2D(128 // nb_filters_reduction_factor,
                          7, 1, border_mode='same', activation='relu')(path1)

    path2 = Convolution2D(128 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)
    out = merge([path1, path2], mode='concat', concat_axis=-1)
    out = Convolution2D(896 // nb_filters_reduction_factor,
                        1, 1, border_mode='same', activation='linear')(out)

    output = merge([out, Input], mode='sum')
    output=BatchNormalization()(output)
    output = Activation('relu')(output)
    return output


def Reduction_B(Input):
    path1 = Convolution2D(256 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)
    path1 = Convolution2D(256 // nb_filters_reduction_factor,
                          3, 3, border_mode='same', activation='relu')(path1)
    path1 = Convolution2D(256 // nb_filters_reduction_factor, 3, 3,
                          subsample=(2, 2), border_mode='valid', activation='relu')(path1)

    path2 = Convolution2D(256 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)
    path2 = Convolution2D(256 // nb_filters_reduction_factor, 3, 3,
                          subsample=(2, 2), border_mode='valid', activation='relu')(path2)

    path3 = Convolution2D(256 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)
    path3 = Convolution2D(384 // nb_filters_reduction_factor, 3, 3,
                          subsample=(2, 2), border_mode='valid', activation='relu')(path3)

    path4 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(Input)

    output = merge([path1, path2, path3, path4], mode='concat', concat_axis=-1)
    output=BatchNormalization()(output)
    output=Activation('relu')(output)
    return output


def Inception_C(Input):

    path1 = Convolution2D(192 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)
    path1 = Convolution2D(192 // nb_filters_reduction_factor,
                          1, 3, border_mode='same', activation='relu')(path1)
    path1 = Convolution2D(192 // nb_filters_reduction_factor,
                          3, 1, border_mode='same', activation='relu')(path1)

    path2 = Convolution2D(192 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)

    out = merge([path1, path2], mode='concat', concat_axis=-1)
    out = Convolution2D(1792 // nb_filters_reduction_factor,
                        1, 1, border_mode='same', activation='linear')(out)

    output = merge([out, Input], mode='sum', concat_axis=-1)
    output=BatchNormalization()(output)
    output = Activation('relu')(output)
    return output

img_rows, img_cols = 150, 150
nb_filters_reduction_factor = 4
channel_axis = -1
img_channels = 3
k = 192
l = 192
m = 256
n = 384
num_A_blocks = 5
num_B_blocks = 10
num_C_blocks = 5
nb_classes = 5
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

x = Dropout(0.5)(x)
x = Flatten()(x)
x = Dense(268, activation='relu')(x)
predictions = Dense(nb_classes, activation='softmax')(x)
model = Model(input=inputs, output=predictions)
print(model.summary())
