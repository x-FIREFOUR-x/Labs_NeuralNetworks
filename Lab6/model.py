import tensorflow as tf
from keras import Model
from keras.layers import Dense, Conv2D, Add, BatchNormalization, SeparableConv2D, Activation, MaxPool2D, GlobalAvgPool2D


def conv_bn(x, filters, filter_size, strides=1):
    x = Conv2D(filters=filters, kernel_size=filter_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    return x


def sep_bn(x, filters, filter_size, strides=1):
    x = SeparableConv2D(filters=filters, kernel_size=filter_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    return x



def entry_flow(x):
    x = conv_bn(x, filters=32, filter_size=3, strides=2)
    x = Activation('relu')(x)
    x = conv_bn(x, filters=64, filter_size=3, strides=1)
    skip_tensor = Activation('relu')(x)

    def entry_flow_block(skip_tensor, x, filters):
        x = Activation('relu')(x)
        x = sep_bn(x, filters=filters, filter_size=3)
        x = Activation('relu')(x)
        x = sep_bn(x, filters=filters, filter_size=3)
        x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

        skip_tensor = conv_bn(skip_tensor, filters=filters, filter_size=1, strides=2)
        x = Add()([skip_tensor, x])

        return skip_tensor, x

    skip_tensor, x = entry_flow_block(skip_tensor, x, 128)
    skip_tensor, x = entry_flow_block(skip_tensor, x, 256)
    skip_tensor, x = entry_flow_block(skip_tensor, x, 728)

    return x


def middle_flow(x):
    def middle_flow_block(x):
        skip_tensor = x

        x = Activation('relu')(x)
        x = sep_bn(x, filters=728, filter_size=3)
        x = Activation('relu')(x)
        x = sep_bn(x, filters=728, filter_size=3)
        x = Activation('relu')(x)
        x = sep_bn(x, filters=728, filter_size=3)
        x = Activation('relu')(x)

        x = Add()([skip_tensor, x])
        return x

    for _ in range(8):
        x = middle_flow_block(x)

    return x


def exit_flow(x):
    skip_tensor = x

    x = Activation('relu')(x)
    x = sep_bn(x, filters=728, filter_size=3)
    x = Activation('relu')(x)
    x = sep_bn(x, filters=1024, filter_size=3)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    skip_tensor = conv_bn(skip_tensor, filters=1024, filter_size=1, strides=2)
    x = Add()([skip_tensor, x])

    x = sep_bn(x, filters=1536, filter_size=3)
    x = Activation('relu')(x)
    x = sep_bn(x, filters=2048, filter_size=3)
    x = GlobalAvgPool2D()(x)

    return x



def Xception(input_size, output_size):
    input_layer = tf.keras.layers.Input(shape=input_size)
    x = entry_flow(input_layer)
    x = middle_flow(x)
    x = exit_flow(x)

    if output_size == 1:
        x = Dense(1, activation='sigmoid', name='output')(x)
    else:
        x = Dense(output_size, activation='softmax', name='output')(x)

    model = Model(input_layer, x)
    return model
