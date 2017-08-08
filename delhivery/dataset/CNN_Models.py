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
from keras import backend as K
from keras.utils.layer_utils import layer_from_config
from keras.datasets import mnist
K.set_image_dim_ordering('tf')
import model_functions as Model_Blocks
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)
print(X_test.shape)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
X_train = X_train / 255
def Model_Selection(Input, model_number, nb_classes):
    if model_number == 1:
        print('This in Inception V4')
        num_Inception_A_blocks = 1
        num_Inception_B_blocks = 1
        num_Inception_C_blocks = 1
        k = 192
        l = 224
        m = 256
        n = 384
        x = Model_Blocks.Stem_V4(Input)
        for i in range(num_Inception_A_blocks):
            x = Model_Blocks.Inception_A_V4(x)
        x = Model_Blocks.Reduction_A(x, k, l, m, n)
        for i in range(num_Inception_B_blocks):
            x = Model_Blocks.Inception_B_V4(x)
        x = Model_Blocks.Reduction_B_V4(x)
        for i in range(num_Inception_C_blocks):
            x = Model_Blocks.Inception_C_V4(x)
        x = AveragePooling2D(pool_size=(2, 2), strides=(
            1, 1), border_mode='valid', dim_ordering='tf')(x)
        x = Dropout(0.2)(x)
        x = Flatten(name='flat')(x)
        x = Dense(384)(x)
        x = Dense(192)(x)
        predictions = Dense(
            nb_classes, activation='softmax', init='uniform')(x)
        model = Model(input=Input, output=predictions)

        layer_name = 'flat'
        intermediate_layer_model = Model(input=Input,
                                         output=model.get_layer(layer_name).output)
        intermediate_output = intermediate_layer_model.predict(X_train)
        print(intermediate_output.shape)
        return model

    elif model_number == 2:
        print('This is Inception-Resnet-V1')
        num_Inception_A_blocks = 5
        num_Inception_B_blocks = 10
        num_Inception_C_blocks = 5
        img_channels = 3
        k = 192
        l = 192
        m = 256
        n = 384
        x = Model_Blocks.Stem_V1(Input)
        for i in range(num_Inception_A_blocks):
            x = Model_Blocks.Inception_A_V1(x)
        x = Model_Blocks.Reduction_A(x, k, l, m, n)
        for i in range(num_Inception_B_blocks):
            x = Model_Blocks.Inception_B_V1(x)
        x = Model_Blocks.Reduction_B_V1(x)
        for i in range(num_Inception_C_blocks):
            x = Model_Blocks.Inception_C_V1(x)
        x = AveragePooling2D(pool_size=(3, 3), strides=(
            1, 1), border_mode='valid', dim_ordering='tf')(x)
        x = Dropout(0.2)(x)
        x = Flatten()(x)
        x=Dense(224,activation='relu')(x)
        predictions = Dense(
            nb_classes, activation='softmax', init='uniform')(x)
        model = Model(input=Input, output=predictions)
        return model

    elif model_number == 3:
        print('This is Inception-Resnet-V2')
        num_Inception_A_blocks = 5
        num_Inception_B_blocks = 10
        num_Inception_C_blocks = 5
        img_channels = 3
        k = 256
        l = 256
        m = 384
        n = 384
        x = Model_Blocks.Stem_V4(Input)
        for i in range(num_Inception_A_blocks):
            x = Model_Blocks.Inception_A_V2(x)
        x = Model_Blocks.Reduction_A(x, k, l, m, n)
        for i in range(num_Inception_B_blocks):
            x = Model_Blocks.Inception_B_V2(x)
        
        x = Model_Blocks.Reduction_B_V2(x)
        for i in range(num_Inception_C_blocks):
            x = Model_Blocks.Inception_C_V2(x)
        x = AveragePooling2D(pool_size=(2, 2), strides=(
            1, 1), border_mode='valid', dim_ordering='tf')(x)
        x = Dropout(0.2)(x)
        x = Flatten()(x)
        x = Dense(536)(x)
        x = Dense(268)(x)
        predictions = Dense(
            nb_classes, activation='softmax', init='uniform')(x)
        model = Model(input=Input, output=predictions)
        return model

    else:
        print('Error. Invalid Model Selection')
        exit(0)


img_rows, img_cols = 150, 150
img_channels = 3
Input = Input(shape=(img_rows,img_cols,1))
model_number = 1
nb_classes = 5
model = Model_Selection(Input, model_number, nb_classes)
print(model.summary())


