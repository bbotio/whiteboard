"""Definitions of neural net models."""
import tensorflow as tf

from keras.models import Model
from keras.layers import Dense, Convolution2D, MaxPooling2D, Activation
from keras.layers import Dropout, Flatten, merge
from keras.objectives import mean_squared_logarithmic_error

from models.utils import sequential


def lenet_like_convnet(input_image):
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
                      Activation('relu'),
                      Convolution2D(nb_filters, kernel_size[0],
                                    kernel_size[1]),
                      Activation('relu'),
                      MaxPooling2D(pool_size=pool_size),
                      Dropout(0.25),
                      Flatten(),
                      Dense(dense_size))


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

    whiteboard_present = Dense(1, activation='sigmoid',
                               name='whiteboard_present')(noisy_model)
    whiteboard_color = Dense(1, activation='sigmoid',
                             name='whiteboard_color')(noisy_model)
    xs = Dense(4)(noisy_model)
    ys = Dense(4)(noisy_model)

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
        y_true: from training labels (given)
        y_pred: algorithm prediction
    """
    wp_t = y_true[:, 0]
    wp_p = y_pred[:, 0]
    wc_t = y_true[:, 1]
    wc_p = y_pred[:, 1]
    xy_t = y_true[:, 2:]
    xy_p = y_pred[:, 2:]
    # point_mapping = tf.constant([0, 1, 2, 3, 0, 1, 2, 3])
    difference = tf.squared_difference(xy_t, xy_p)
    losses = [tf.square(wp_t - wp_p), wp_t * tf.square(wc_t - wc_p),
              tf.reduce_sum(difference, 1)]
    return tf.add_n(losses)
