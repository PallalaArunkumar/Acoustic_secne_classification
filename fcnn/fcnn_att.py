import tensorflow as tf
from tensorflow import keras
from attention_layer import channel_attention


# network definition
def resnet_layer(inputs,num_filters=16,kernel_size=3,strides=1,learn_bn = True,wd=1e-4,use_relu=True):
  x=inputs

  x=tf.keras.layers.Conv2D(num_filters,kernel_size=kernel_size,strides=strides,padding='valid',kernel_initializer='he_normal',
                  kernel_regularizer='l2',use_bias=False)(x)
  x=tf.keras.layers.BatchNormalization(center=learn_bn, scale=learn_bn)(x)
  if use_relu:
    x=tf.keras.layers.Activation('relu')(x)
  return x

def conv_layer1(inputs, num_channels=6, num_filters=14, learn_bn=True, wd=1e-4, use_relu=True):
    kernel_size1 = [5, 5]
    kernel_size2 = [3, 3]
    strides1 = [2, 2]
    strides2 = [1, 1]
    x = inputs
    x =tf.keras.layers.BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    x =tf.keras.layers.ZeroPadding2D(padding=(2, 2), data_format='channels_last')(x)
    x =tf.keras.layers.Conv2D(num_filters*num_channels, kernel_size=kernel_size1, strides=strides1,
               padding='valid', kernel_initializer='he_normal',
               kernel_regularizer='l2', use_bias=False)(x)
    x =tf.keras.layers.BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        #x = Activation(relu6)(x)
        x =tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.ZeroPadding2D(padding=(1, 1), data_format='channels_last')(x)
    x = tf.keras.layers.Conv2D(num_filters * num_channels, kernel_size=kernel_size2, strides=strides2,
               padding='valid', kernel_initializer='he_normal',
               kernel_regularizer='l2', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        #x = Activation(relu6)(x)
        x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),padding='valid')(x)
    return x

def conv_layer2(inputs, num_channels=6, num_filters=28, learn_bn=True, wd=1e-4, use_relu=True):
    kernel_size = [3, 3]
    strides = [1, 1]
    x = inputs
    x =tf.keras.layers.ZeroPadding2D(padding=(1, 1), data_format='channels_last')(x)
    x =tf.keras.layers.Conv2D(num_filters*num_channels, kernel_size=kernel_size, strides=strides,
               padding='valid', kernel_initializer='he_normal',
               kernel_regularizer='l2', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        #x = Activation(relu6)(x)
        x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.ZeroPadding2D(padding=(1, 1), data_format='channels_last')(x)
    x = tf.keras.layers.Conv2D(num_filters * num_channels, kernel_size=kernel_size, strides=strides,
               padding='valid', kernel_initializer='he_normal',
               kernel_regularizer='l2', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        #x = Activation(relu6)(x)
        x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid')(x)
    return x

def conv_layer3(inputs, num_channels=6, num_filters=56, learn_bn=True, wd=1e-4, use_relu=True):
    kernel_size = [3, 3]
    strides = [1, 1]
    x = inputs
    # 1
    x = tf.keras.layers.ZeroPadding2D(padding=(1, 1), data_format='channels_last')(x)
    x = tf.keras.layers.Conv2D(num_filters*num_channels, kernel_size=kernel_size, strides=strides,
               padding='valid', kernel_initializer='he_normal',
               kernel_regularizer='l2', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        #x = Activation(relu6)(x)
        x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    #2
    x = tf.keras.layers.ZeroPadding2D(padding=(1, 1), data_format='channels_last')(x)
    x = tf.keras.layers.Conv2D(num_filters*num_channels, kernel_size=kernel_size, strides=strides,
               padding='valid', kernel_initializer='he_normal',
               kernel_regularizer='l2', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        #x = Activation(relu6)(x)
        x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    # 3
    x = tf.keras.layers.ZeroPadding2D(padding=(1, 1), data_format='channels_last')(x)
    x = tf.keras.layers.Conv2D(num_filters*num_channels, kernel_size=kernel_size, strides=strides,
               padding='valid', kernel_initializer='he_normal',
               kernel_regularizer='l2', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        #x = Activation(relu6)(x)
        x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    #4
    x = tf.keras.layers.ZeroPadding2D(padding=(1, 1), data_format='channels_last')(x)
    x = tf.keras.layers.Conv2D(num_filters*num_channels, kernel_size=kernel_size, strides=strides,
               padding='valid', kernel_initializer='he_normal',
               kernel_regularizer='l2', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        #x = Activation(relu6)(x)
        x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid')(x)
    return x



def model_fcnn(num_classes, input_shape=[None, 128, 6], num_filters=[24, 48, 96], wd=1e-3):
    inputs = tf.keras.layers.Input(shape=input_shape)
    ConvPath1 = conv_layer1(inputs=inputs,
                            num_channels=input_shape[-1],
                            num_filters=num_filters[0],
                            learn_bn=True,
                            wd=wd,
                            use_relu=True)
    ConvPath2 = conv_layer2(inputs=ConvPath1,
                            num_channels=input_shape[-1],
                            num_filters=num_filters[1],
                            learn_bn=True,
                            wd=wd,
                            use_relu=True)
    ConvPath3 = conv_layer3(inputs=ConvPath2,
                            num_channels=input_shape[-1],
                            num_filters=num_filters[2],
                            learn_bn=True,
                            wd=wd,
                            use_relu=True)

    # output layers after last sum
    OutputPath = resnet_layer(inputs=ConvPath3,
                              num_filters=num_classes,
                              strides=1,
                              kernel_size=1,
                              learn_bn=False,
                              wd=wd,
                              use_relu=True)

    OutputPath = tf.keras.layers.BatchNormalization(center=False, scale=False)(OutputPath)
    OutputPath = channel_attention(OutputPath, ratio=2)

    OutputPath = tf.keras.layers.GlobalAveragePooling2D()(OutputPath)
    OutputPath =tf.keras.layers.Activation('softmax')(OutputPath)
    # Instantiate model.
    model = tf.keras.Model(inputs=inputs, outputs=OutputPath)
    return model
