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
##########################################################################
nb_filters_reduction_factor = 8
alpha = 0.1
concat_axis = -1
channel_axis = -1
##########################################################################


def Stem_V4(Input):
    x = Convolution2D(32 // nb_filters_reduction_factor,
                      3, 3, subsample=(2,2), activation='relu',init='he_normal')(Input)
    x = Convolution2D(32 // nb_filters_reduction_factor,
                      3, 3, activation='relu',init='he_normal')(x)
    x = Convolution2D(32 // nb_filters_reduction_factor,
                      3, 3, activation='relu',init='he_normal')(x)
    x = Convolution2D(64 // nb_filters_reduction_factor,
                      3, 3, border_mode='same', activation='relu',init='he_normal')(x)

    path1 = MaxPooling2D((3, 3), strides=(2, 2))(x)  # changed
    path2 = Convolution2D(96 // nb_filters_reduction_factor,
                          3, 3, subsample=(2,2), activation='relu',init='he_normal')(x)  # changed
    y = merge([path1, path2], mode='concat')

    a = Convolution2D(64 // nb_filters_reduction_factor,
                      1, 1, border_mode='same', activation='relu',init='he_normal')(y)
    a = Convolution2D(64 // nb_filters_reduction_factor,
                      3, 1, border_mode='same', activation='relu',init='he_normal')(a)
    a = Convolution2D(64 // nb_filters_reduction_factor,
                      1, 3, border_mode='same', activation='relu',init='he_normal')(a)
    a = Convolution2D(96 // nb_filters_reduction_factor,
                      3, 3, border_mode='valid', activation='relu',init='he_normal')(a)

    b = Convolution2D(64 // nb_filters_reduction_factor,
                      1, 1, border_mode='same', activation='relu',init='he_normal')(y)
    b = Convolution2D(96 // nb_filters_reduction_factor,
                      3, 3, border_mode='valid', activation='relu',init='he_normal')(b)

    z = merge([a, b], mode='concat')
    z1 = MaxPooling2D((3, 3), strides=(2, 2))(z)
    z2 = Convolution2D(192 // nb_filters_reduction_factor, 3,
                       3, subsample=(2,2), border_mode='valid', activation='relu',init='he_normal')(z)

    c = merge([z1, z2], mode='concat', concat_axis=concat_axis)
    return c


def Stem_V1(Input):
    path1 = Convolution2D(32 // nb_filters_reduction_factor, 3, 3,
                          subsample=(2, 2), border_mode='valid', activation='relu',init='he_normal')(Input)
    path1 = Convolution2D(32 // nb_filters_reduction_factor,
                          3, 3, border_mode='same', activation='relu',init='he_normal')(path1)
    path1 = Convolution2D(64 // nb_filters_reduction_factor,
                          3, 3, border_mode='same', activation='relu',init='he_normal')(path1)
    path1 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(path1)
    path1 = Convolution2D(80 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu',init='he_normal')(path1)
    path1 = Convolution2D(192 // nb_filters_reduction_factor,
                          3, 3, border_mode='same', activation='relu',init='he_normal')(path1)
    path1 = Convolution2D(256 // nb_filters_reduction_factor, 3, 3,
                          subsample=(2, 2), border_mode='valid', activation='relu',init='he_normal')(path1)
    path1 = BatchNormalization()(path1)
    path1 = Activation('relu')(path1)
    return path1


def Inception_A_V4(Input):
    path1 = Convolution2D(64 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)
    path1 = Convolution2D(96 // nb_filters_reduction_factor,
                          3, 3, border_mode='same', activation='relu')(path1)
    path1 = Convolution2D(96 // nb_filters_reduction_factor,
                          3, 3, border_mode='same', activation='relu')(path1)

    path2 = Convolution2D(64 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)
    path2 = Convolution2D(96 // nb_filters_reduction_factor,
                          3, 3, border_mode='same', activation='relu')(path2)

    path3 = Convolution2D(96 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)

    path4 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(Input)
    path4 = Convolution2D(96 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(path4)

    output = merge([path1, path2, path3, path4],
                   mode='concat', concat_axis=concat_axis)
    return output


def Inception_A_V1(Input):
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
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    return output


def Inception_A_V2(Input):
    Input = Activation('relu')(Input)
    path1 = Convolution2D(32 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)
    path1 = Convolution2D(48 // nb_filters_reduction_factor,
                          3, 3, border_mode='same', activation='relu')(path1)
    path1 = Convolution2D(64 // nb_filters_reduction_factor,
                          3, 3, border_mode='same', activation='relu')(path1)

    path2 = Convolution2D(32 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)
    path2 = Convolution2D(32 // nb_filters_reduction_factor,
                          3, 3, border_mode='same', activation='relu')(path2)

    path3 = Convolution2D(32 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)

    out = merge([path3, path1, path2], mode='concat', concat_axis=concat_axis)
    out = Convolution2D(384 // nb_filters_reduction_factor,
                        1, 1, border_mode='same', activation='linear')(out)
    out = Lambda(lambda x: x * alpha)(out)

    output = merge([out, Input], mode='sum')
    output = BatchNormalization(axis=channel_axis)(output)
    output = Activation('relu')(output)

    return output


def Reduction_A(Input, k, l, m, n):
    path1 = Convolution2D(k // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)
    path1 = Convolution2D(l // nb_filters_reduction_factor,
                          3, 3, border_mode='same', activation='relu')(path1)
    path1 = Convolution2D(m // nb_filters_reduction_factor, 3, 3,
                          subsample=(2, 2), border_mode='valid', activation='relu')(path1)

    path2 = Convolution2D(n // nb_filters_reduction_factor, 3, 3,
                          subsample=(2, 2), border_mode='valid', activation='relu')(Input)

    path3 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(Input)

    output = merge([path1, path2, path3], mode='concat',
                   concat_axis=concat_axis)
    return output


def Inception_B_V4(Input):
    path1 = Convolution2D(192 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)
    path1 = Convolution2D(192 // nb_filters_reduction_factor,
                          1, 7, border_mode='same', activation='relu')(path1)
    path1 = Convolution2D(224 // nb_filters_reduction_factor,
                          7, 1, border_mode='same', activation='relu')(path1)
    path1 = Convolution2D(224 // nb_filters_reduction_factor,
                          1, 7, border_mode='same', activation='relu')(path1)
    path1 = Convolution2D(256 // nb_filters_reduction_factor,
                          7, 1, border_mode='same', activation='relu')(path1)

    path2 = Convolution2D(192 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)
    path2 = Convolution2D(223 // nb_filters_reduction_factor,
                          1, 7, border_mode='same', activation='relu')(path2)
    path2 = Convolution2D(256 // nb_filters_reduction_factor,
                          1, 7, border_mode='same', activation='relu')(path2)

    path3 = Convolution2D(384 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)

    path4 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(Input)
    path4 = Convolution2D(128 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(path4)

    output = merge([path1, path2, path3, path4],
                   mode='concat', concat_axis=concat_axis)

    return output


def Inception_B_V1(Input):
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
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    return output


def Inception_B_V2(Input):
    Input = Activation('relu')(Input)
    path1 = Convolution2D(128 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)
    path1 = Convolution2D(160 // nb_filters_reduction_factor,
                          1, 7, border_mode='same', activation='relu')(path1)
    path1 = Convolution2D(192 // nb_filters_reduction_factor,
                          7, 1, border_mode='same', activation='relu')(path1)

    path2 = Convolution2D(192 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)

    out = merge([path1, path2], mode='concat', concat_axis=channel_axis)
    out = Convolution2D(1152 // nb_filters_reduction_factor,
                        1, 1, border_mode='same', activation='linear')(out)
    out = Lambda(lambda x: x * alpha)(out)
    output = merge([out, Input], mode='sum')
    output = BatchNormalization(axis=channel_axis)(output)
    output = Activation('relu')(output)

    return output


def Reduction_B_V4(Input):
    path1 = Convolution2D(256 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)
    path1 = Convolution2D(256 // nb_filters_reduction_factor,
                          1, 7, border_mode='same', activation='relu')(path1)
    path1 = Convolution2D(320 // nb_filters_reduction_factor,
                          7, 1, border_mode='same', activation='relu')(path1)
    path1 = Convolution2D(320 // nb_filters_reduction_factor, 3, 3,
                          subsample=(2, 2), border_mode='valid', activation='relu')(path1)

    path2 = Convolution2D(192 // nb_filters_reduction_factor,
                          1, 1, border_mode="same", activation='relu')(Input)
    path2 = Convolution2D(192 // nb_filters_reduction_factor, 3, 3,
                          subsample=(2, 2), border_mode="valid", activation='relu')(path2)

    path3 = MaxPooling2D((3, 3), strides=(2, 2))(Input)

    output = merge([path1, path2, path3], mode='concat',
                   concat_axis=concat_axis)
    return output


def Reduction_B_V1(Input):
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
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    return output


def Reduction_B_V2(Input):
    path1 = Convolution2D(256 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)
    path1 = Convolution2D(288 // nb_filters_reduction_factor,
                          3, 3, border_mode='same', activation='relu')(path1)
    path1 = Convolution2D(320 // nb_filters_reduction_factor, 3, 3,
                          subsample=(2, 2), border_mode='valid', activation='relu')(path1)

    path2 = Convolution2D(256 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)
    path2 = Convolution2D(288 // nb_filters_reduction_factor, 3, 3,
                          subsample=(2, 2), border_mode='valid', activation='relu')(path2)

    path3 = Convolution2D(256 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)
    path3 = Convolution2D(384 // nb_filters_reduction_factor, 3, 3,
                          subsample=(2, 2), border_mode='valid', activation='relu')(path3)

    path4 = MaxPooling2D((3, 3), strides=(2, 2))(Input)

    output = merge([path1, path2, path3, path4],
                   mode='concat', concat_axis=channel_axis)
    output = BatchNormalization(axis=channel_axis)(output)
    return output


def Inception_C_V4(Input):
    path1 = Convolution2D(384 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)
    path1 = Convolution2D(448 // nb_filters_reduction_factor,
                          1, 3, border_mode='same', activation='relu')(path1)
    path1 = Convolution2D(512 // nb_filters_reduction_factor,
                          3, 1, border_mode='same', activation='relu')(path1)
    path1_a = Convolution2D(256 // nb_filters_reduction_factor,
                            1, 3, border_mode='same', activation='relu')(path1)
    path1_b = Convolution2D(256 // nb_filters_reduction_factor,
                            3, 1, border_mode='same', activation='relu')(path1)

    path2 = Convolution2D(384 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)
    path2_a = Convolution2D(256 // nb_filters_reduction_factor,
                            3, 1, border_mode='same', activation='relu')(path2)
    path2_b = Convolution2D(256 // nb_filters_reduction_factor,
                            1, 3, border_mode='same', activation='relu')(path2)

    path3 = Convolution2D(256 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)

    path4 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(Input)
    path4 = Convolution2D(256 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='linear',name='layer')(Input)

    output = merge([path1_a,path1_b, path2_a,path2_b, path3, path4],
                   mode='concat', concat_axis=concat_axis)
    return output


def Inception_C_V1(Input):

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
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    return output


def Inception_C_V2(Input):
    Input = Activation('relu')(Input)
    path1 = Convolution2D(192 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)
    path1 = Convolution2D(224 // nb_filters_reduction_factor,
                          1, 3, border_mode='same', activation='relu')(path1)
    path1 = Convolution2D(256 // nb_filters_reduction_factor,
                          3, 1, border_mode='same', activation='relu')(path1)

    path2 = Convolution2D(192 // nb_filters_reduction_factor,
                          1, 1, border_mode='same', activation='relu')(Input)
    out = merge([path1, path2], mode='concat', concat_axis=channel_axis)
    out = Convolution2D(2144 // nb_filters_reduction_factor,
                        1, 1, border_mode='same', activation='linear')(out)
    out = Lambda(lambda x: x * alpha)(out)
    output = merge([out, Input], mode='sum')
    output = BatchNormalization(axis=channel_axis)(output)
    output = Activation('relu')(output)
    return output
