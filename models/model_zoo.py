"""Definitions of neural net models."""
import os
import tensorflow as tf

from collections import namedtuple
from keras.models import Model, Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Activation
from keras.layers import Dropout, Flatten, merge
from keras.layers.normalization import BatchNormalization

from models.utils import sequential


def lenet_like_convnet(input_image, dropout=0.25):
    """
    Create simple image convnet based on old lenet.

    Lenet is usually used to identify handwritten digits.

    Stolen from here:
    https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py

    Args:
        input_image: input image layer

    Returns:
        simple convnet model
    """
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (3, 3)
    dense_size = 128

    # Found no way to use models.Sequential with Input tensor
    return sequential(input_image,
                      Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                                    border_mode='valid'),
                      BatchNormalization(),
                      Activation('relu'),
                      Convolution2D(nb_filters, kernel_size[0],
                                    kernel_size[1]),
                      BatchNormalization(),
                      Activation('relu'),
                      MaxPooling2D(pool_size=pool_size),
                      BatchNormalization(),
                      Dropout(dropout),
                      Flatten(),
                      Dense(dense_size))


def psin(x):
    """Compute b * sin(a * x) activation."""
    alpha = tf.constant(1.0)
    beta = tf.constant(1.0)
    return beta * tf.sin(alpha * x)


def pcos(x):
    """Compute b * cos(a * x) activation."""
    alpha = tf.constant(1.0)
    beta = tf.constant(1.0)
    return beta * tf.cos(alpha * x)


ModelInfo = namedtuple('ModelInfo', ['model', 'height', 'width', 'patch_size',
                                     'out_size', 'model_weights'])


def baseline_whiteboard_detector():
    """Variant of whiteboard detector trained on 50 epochs."""
    model_weights = 'models/weights/corner_detector_baseline.hf5'
    input_height = 200
    input_width = 150
    nb_filters = 32
    patch_size = 5
    agg_size = 6
    dropout_pb = 0.5
    nb_cv_features = 5
    out_size = (input_width // patch_size, input_height // patch_size)

    model = Sequential()
    model.add(BatchNormalization(input_shape=(input_height, input_width,
                                              nb_cv_features),
                                 name='input_norm'))
    model.add(Convolution2D(nb_filters, patch_size, patch_size,
                            subsample=(patch_size, patch_size),
                            name='patch_conv', activation=psin))
    model.add(Dropout(dropout_pb))
    model.add(BatchNormalization(name='hidden_norm'))
    model.add(Convolution2D(nb_filters, agg_size, agg_size, subsample=(1, 1),
                            name='aggregated_conv', border_mode='same',
                            activation=pcos))
    model.add(Dropout(dropout_pb))
    model.add(BatchNormalization(name='patch_norm'))
    # probability of corner for each patch using aggregated feature maps
    model.add(Convolution2D(1, 1, 1, subsample=(1, 1), activation='sigmoid',
                            name='sigmoid_conv'))

    if os.path.exists(model_weights):
        model.load_weights(model_weights, by_name=True)

    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    return ModelInfo(model=model, height=input_height, width=input_width,
                     patch_size=patch_size, out_size=out_size,
                     model_weights=model_weights)


def wide_whiteboard_detector():
    """Variant of whiteboard detector trained for 500 epochs."""
    model_weights = 'models/weights/corner_detector_2.hf5'
    input_height = 150
    input_width = 150
    nb_filters = 48
    patch_size = 5
    agg_size = 6
    dropout_pb = 0.5
    nb_cv_features = 5
    out_size = (input_width // patch_size, input_height // patch_size)

    model = Sequential()
    model.add(BatchNormalization(input_shape=(input_height, input_width,
                                              nb_cv_features),
                                 name='input_norm'))
    model.add(Convolution2D(nb_filters, patch_size, patch_size,
                            subsample=(patch_size, patch_size),
                            name='patch_conv',
                            activation=psin,
                            init='glorot_uniform'))
    model.add(Dropout(dropout_pb))
    model.add(BatchNormalization(name='hidden_norm'))
    model.add(Convolution2D(nb_filters, agg_size, agg_size, subsample=(1, 1),
                            name='aggregated_conv', border_mode='same',
                            activation=pcos,
                            init='glorot_uniform'))
    model.add(Dropout(dropout_pb))
    model.add(BatchNormalization(name='patch_norm'))
    # compute probability of corner for each patch
    # using aggregated feature maps.
    model.add(Convolution2D(1, 1, 1, subsample=(1, 1), activation='sigmoid',
                            name='sigmoid_conv'))
    if os.path.exists(model_weights):
        model.load_weights(model_weights, by_name=True)

    model.compile(optimizer='adam', loss='binary_crossentropy')
    return ModelInfo(model=model, height=input_height, width=input_width,
                     patch_size=patch_size, out_size=out_size,
                     model_weights=model_weights)


def whiteboard_detector(input_image, convnet):
    """
    Create model for whiteboard detector.

    Whiteboard is 4x2 matrix:
        x1 x2 x3 x4
        y1 y2 y3 y4
    Whiteboard also have color: black or white.

    Network returns:
        whiteboard_present: sigmoid will be close to 1
        if whiteboard is on screen
        whiteboard_color: sigmoid will be close to 1
        if whiteboard is white
        [x1 x2 x3 x4]: vector of x coords between 0 and height
        [y1 y2 y3 y4]: vector of y coords between 0 and width

    Args:
        input_image: input image
        convnet: some convnet model

    Returns:
        whiteboard detector model
    """
    noisy_model = Dropout(0.5)(convnet)
    noisy_model = BatchNormalization()(noisy_model)
    noisy_model = Dense(64)(noisy_model)
    noisy_model = BatchNormalization()(noisy_model)
    noisy_model = Dropout(0.5)(noisy_model)
    noisy_model = Dense(32)(noisy_model)
    noisy_model = BatchNormalization()(noisy_model)
    noisy_model = Dense(16)(noisy_model)
    whiteboard_present = Dense(1, activation='sigmoid',
                               name='whiteboard_present')(noisy_model)
    whiteboard_color = Dense(1, activation='sigmoid',
                             name='whiteboard_color')(noisy_model)
    xs = Dense(4, activation='sigmoid')(noisy_model)
    ys = Dense(4, activation='sigmoid')(noisy_model)

    # Need to merge present bit and coords into single vector
    # to use custom loss function on it
    merged_whiteboard = merge([whiteboard_present,
                               whiteboard_color, xs, ys], mode='concat',
                              name='whiteboard')
    return Model(input=input_image, output=merged_whiteboard)


def whiteboard_loss(y_true, y_pred):
    """
    Loss tensor function for whiteboard detector.

    Arguments have following shape:
        [whiteboard_present whiteboard_color x1 x2 x3 x4 y1 y2 y3 y4]

    Args:
        y_true: from training labels (given) normalized to image size
        y_pred: algorithm prediction
    """
    wp_t = y_true[:, 0]
    wp_p = y_pred[:, 0]
    wc_t = y_true[:, 1]
    wc_p = y_pred[:, 1]
    xy_t = y_true[:, 2:]
    xy_p = y_pred[:, 2:]
    difference = tf.squared_difference(xy_t, xy_p)
    losses = [tf.nn.sigmoid_cross_entropy_with_logits(labels=wp_t,
                                                      logits=wp_p),
              wp_t * tf.nn.sigmoid_cross_entropy_with_logits(labels=wc_t,
                                                             logits=wc_p),
              tf.reduce_sum(difference, 1)]
    return tf.add_n(losses)
