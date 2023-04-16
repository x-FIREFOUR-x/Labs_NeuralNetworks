import tensorflow as tf
from keras import Model
from keras.layers import Conv2D, Activation, BatchNormalization, MaxPool2D, Input, GlobalAveragePooling2D,\
    AveragePooling2D, Dense, Dropout, Flatten, concatenate


def conv_with_batch_normalisation(prev_layer, filters, filter_size, strides=(1, 1), padding='same'):
    x = Conv2D(filters=filters, kernel_size=filter_size, strides=strides, padding=padding)(prev_layer)
    x = BatchNormalization(axis=3)(x)
    x = Activation(activation='relu')(x)
    return x


def stem_block(prev_layer):
    x = conv_with_batch_normalisation(prev_layer, filters=32, filter_size=(3, 3), strides=(2, 2))
    x = conv_with_batch_normalisation(x, filters=32, filter_size=(3, 3))
    x = conv_with_batch_normalisation(x, filters=64, filter_size=(3, 3))
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = conv_with_batch_normalisation(x, filters=80, filter_size=(1, 1))
    x = conv_with_batch_normalisation(x, filters=192, filter_size=(3, 3))
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
    return x


def inception_block_A(prev_layer, filters_branch3):
    branch1 = conv_with_batch_normalisation(prev_layer, filters=64, filter_size=(1, 1))
    branch1 = conv_with_batch_normalisation(branch1, filters=96, filter_size=(3, 3))
    branch1 = conv_with_batch_normalisation(branch1, filters=96, filter_size=(3, 3))

    branch2 = conv_with_batch_normalisation(prev_layer, filters=48, filter_size=(1, 1))
    branch2 = conv_with_batch_normalisation(branch2, filters=64, filter_size=(3, 3))  # may be 3*3

    branch3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(prev_layer)
    branch3 = conv_with_batch_normalisation(branch3, filters=filters_branch3, filter_size=(1, 1))

    branch4 = conv_with_batch_normalisation(prev_layer, filters=64, filter_size=(1, 1))

    output = concatenate([branch1, branch2, branch3, branch4], axis=3)
    return output


def inception_block_B(prev_layer, filters_br1_br2):
    branch1 = conv_with_batch_normalisation(prev_layer, filters=filters_br1_br2, filter_size=(1, 1))
    branch1 = conv_with_batch_normalisation(branch1, filters=filters_br1_br2, filter_size=(7, 1))
    branch1 = conv_with_batch_normalisation(branch1, filters=filters_br1_br2, filter_size=(1, 7))
    branch1 = conv_with_batch_normalisation(branch1, filters=filters_br1_br2, filter_size=(7, 1))
    branch1 = conv_with_batch_normalisation(branch1, filters=192, filter_size=(1, 7))

    branch2 = conv_with_batch_normalisation(prev_layer, filters=filters_br1_br2, filter_size=(1, 1))
    branch2 = conv_with_batch_normalisation(branch2, filters=filters_br1_br2, filter_size=(1, 7))
    branch2 = conv_with_batch_normalisation(branch2, filters=192, filter_size=(7, 1))

    branch3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(prev_layer)
    branch3 = conv_with_batch_normalisation(branch3, filters=192, filter_size=(1, 1))

    branch4 = conv_with_batch_normalisation(prev_layer, filters=192, filter_size=(1, 1))

    output = concatenate([branch1, branch2, branch3, branch4], axis=3)
    return output


def inception_block_C(prev_layer):
    branch1 = conv_with_batch_normalisation(prev_layer, filters=448, filter_size=(1, 1))
    branch1 = conv_with_batch_normalisation(branch1, filters=384, filter_size=(3, 3))
    branch1_1 = conv_with_batch_normalisation(branch1, filters=384, filter_size=(1, 3))
    branch1_2 = conv_with_batch_normalisation(branch1, filters=384, filter_size=(3, 1))
    branch1 = concatenate([branch1_1, branch1_2], axis=3)

    branch2 = conv_with_batch_normalisation(prev_layer, filters=384, filter_size=(1, 1))
    branch2_1 = conv_with_batch_normalisation(branch2, filters=384, filter_size=(1, 3))
    branch2_2 = conv_with_batch_normalisation(branch2, filters=384, filter_size=(3, 1))
    branch2 = concatenate([branch2_1, branch2_2], axis=3)

    branch3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(prev_layer)
    branch3 = conv_with_batch_normalisation(branch3, filters=192, filter_size=(1, 1))

    branch4 = conv_with_batch_normalisation(prev_layer, filters=320, filter_size=(1, 1))

    output = concatenate([branch1, branch2, branch3, branch4], axis=3)
    return output


def reduction_block_A(prev_layer):
    branch1 = conv_with_batch_normalisation(prev_layer, filters=64, filter_size=(1, 1))
    branch1 = conv_with_batch_normalisation(branch1, filters=96, filter_size=(3, 3))
    branch1 = conv_with_batch_normalisation(branch1, filters=96, filter_size=(3, 3), strides=(2, 2))

    branch2 = conv_with_batch_normalisation(prev_layer, filters=384, filter_size=(3, 3), strides=(2, 2))

    branch3 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(prev_layer)

    output = concatenate([branch1, branch2, branch3], axis=3)
    return output


def reduction_block_B(prev_layer):
    branch1 = conv_with_batch_normalisation(prev_layer, filters=192, filter_size=(1, 1))
    branch1 = conv_with_batch_normalisation(branch1, filters=192, filter_size=(1, 7))
    branch1 = conv_with_batch_normalisation(branch1, filters=192, filter_size=(7, 1))
    branch1 = conv_with_batch_normalisation(branch1, filters=192, filter_size=(3, 3), strides=(2, 2),
                                            padding='valid')

    branch2 = conv_with_batch_normalisation(prev_layer, filters=192, filter_size=(1, 1))
    branch2 = conv_with_batch_normalisation(branch2, filters=320, filter_size=(3, 3),
                                            strides=(2, 2), padding='valid')

    branch3 = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(prev_layer)

    output = concatenate([branch1, branch2, branch3], axis=3)
    return output


def auxiliary_classifier(prev_layer, output_size):
    x = AveragePooling2D(pool_size=(5, 5), strides=(3, 3))(prev_layer)
    x = conv_with_batch_normalisation(x, filters=128, filter_size=(1, 1))
    x = Flatten()(x)
    x = Dense(units=768, activation='relu')(x)
    x = Dropout(rate=0.2)(x)
    x = Dense(output_size, activation='softmax', name='aux_output')(x)
    return x


def InceptionV3(input_size, output_size):
    input_layer = Input(shape=input_size)

    x = stem_block(input_layer)

    x = inception_block_A(x, filters_branch3=32)
    x = inception_block_A(x, filters_branch3=64)
    x = inception_block_A(x, filters_branch3=64)

    x = reduction_block_A(x)

    x = inception_block_B(x, filters_br1_br2=128)
    x = inception_block_B(x, filters_br1_br2=160)
    x = inception_block_B(x, filters_br1_br2=160)
    x = inception_block_B(x, filters_br1_br2=192)

    Aux = auxiliary_classifier(x, output_size)

    x = reduction_block_B(x)

    x = inception_block_C(x)
    x = inception_block_C(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(output_size, activation='softmax', name='output')(x)

    model = Model(input_layer, [x, Aux])

    return model
