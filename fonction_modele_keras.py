'''
Ensemble des fonctions retournent un modele keras
'''

from tensorflow import keras
from functools import partial


def create_model_AlexNet1(lx, ly, lz):

    regularized_conv2D = partial(keras.layers.Conv2D,
                                 kernel_initializer=keras.initializers.HeNormal(),
                                 kernel_regularizer=keras.regularizers.l2(1),
                                 padding = 'same',
                                 activation = 'relu')

    regularized_pooling = partial(keras.layers.MaxPooling2D,
                                 padding = 'valid',
                                 strides=2,)



    regularized_dense = partial(keras.layers.Dense,
                                activation='relu',
                                kernel_initializer=keras.initializers.HeNormal(),
                                kernel_regularizer= keras.regularizers.l2(0))

    # Premier modele teste
    model = keras.models.Sequential()

    model.add(regularized_conv2D(96, (11, 11), strides=4, input_shape=(lx, ly, lz))) # 227 x 227
    model.add(regularized_pooling((3, 3))) # 55 x 55

    model.add(regularized_conv2D(256, (5, 5), strides=1)) # 27 x 27
    model.add(regularized_pooling((3, 3))) # # 13 x 13

    model.add(regularized_conv2D(384, (3, 3), strides=1))  # # 13 x 13

    model.add(regularized_conv2D(384, (3, 3), strides=1))  # # 13 x 13

    model.add(regularized_conv2D(256, (3, 3), strides=1))  # # 13 x 13
    model.add(regularized_pooling((3, 3))) # 6 x 6

    model.add(keras.layers.Flatten())

    model.add(regularized_dense(4096))
    model.add(keras.layers.Dropout(0.5))

    model.add(regularized_dense(4096))
    model.add(keras.layers.Dropout(0.5))

    model.add(regularized_dense(36, activation='softmax'))

    return model

def create_model_AlexNet2(lx, ly, lz):

    regularized_conv2D = partial(keras.layers.Conv2D,
                                 kernel_initializer=keras.initializers.HeNormal(),
                                 kernel_regularizer=keras.regularizers.l1_l2(1, 0.2),
                                 padding = 'same',
                                 activation = 'ReLU')

    regularized_pooling = partial(keras.layers.MaxPooling2D,
                                 padding = 'valid',
                                 strides=2,)



    regularized_dense = partial(keras.layers.Dense,
                                activation='ReLU',
                                kernel_initializer=keras.initializers.HeNormal(),
                                kernel_regularizer= keras.regularizers.l2(0))

    # Premier modele teste
    model = keras.models.Sequential()

    model.add(regularized_conv2D(96, (11, 11), strides=4, input_shape=(lx, ly, lz))) # 227 x 227
    model.add(regularized_pooling((3, 3))) # 55 x 55

    model.add(regularized_conv2D(256, (5, 5), strides=1)) # 27 x 27
    model.add(regularized_pooling((3, 3))) # # 13 x 13

    model.add(regularized_conv2D(384, (3, 3), strides=1))  # # 13 x 13

    model.add(regularized_conv2D(384, (3, 3), strides=1))  # # 13 x 13

    model.add(regularized_conv2D(256, (3, 3), strides=1))  # # 13 x 13
    model.add(regularized_pooling((3, 3))) # 6 x 6

    model.add(keras.layers.Flatten())

    model.add(regularized_dense(4096))
    model.add(keras.layers.Dropout(0.3))

    model.add(regularized_dense(4096))
    model.add(keras.layers.Dropout(0.3))

    model.add(regularized_dense(36, activation='sigmoid', kernel_initializer="glorot_uniform"))

    model.summary()
    return model



