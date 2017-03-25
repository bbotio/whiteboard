#!/home/bfilippov/anaconda3/bin/python
"""Train model."""
import os
import keras
import pandas as pd
import ast

import cv2
import tensorflow as tf
import numpy as np
from keras.layers import Input
from keras import backend as K
from keras.models import Model
from keras.layers import (
    Convolution2D, MaxPooling2D,
    BatchNormalization, Dropout, merge, UpSampling2D
)
from keras.layers.advanced_activations import LeakyReLU
from models.model_zoo import residual_block
from models.image_generator import whiteboard_images


os.chdir('/home/bfilippov/whiteboard')

train = pd.read_csv('source/train.csv')
test = pd.read_csv('source/test.csv')
train['labels'] = train['labels'].map(ast.literal_eval)
test['labels'] = test['labels'].map(ast.literal_eval)

valid = train.sample(frac=0.2, random_state=1313)
train = train.drop(valid.index)

smooth = 1e-12


def jaccard_coef(y_true, y_pred):
    """Compute jaccard coefficient."""
    # __author__ = Vladimir Iglovikov
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    """Compute jaccard coefficient over pixels."""
    # __author__ = Vladimir Iglovikov
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)


def jaccard_loss(y_true, y_pred):
    """Loss on jaccard coefficient."""
    return 1 - jaccard_coef(y_true, y_pred)


def patch_generator(dirname, generator):
    """Make patches."""
    class Next:
        def __next__(self):
            batch_x, batch_y = next(generator)
            self.batch_y = batch_y
            return batch_x
    return Next()


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
sess = tf.Session(config=config)
K.set_session(sess)


def scale(N, i, scaler=2):
    """N * (i ** scaler)."""
    return int(N * (scaler**i))


def down_res(name, in_layer, dropout_pb, nb_filters, patch_size):
    """
    Submerge U-net. Increases feature map size and decreases image size.

    Feed it to residual block.
    """
    downsampled = downsample(in_layer, 'pool_' + name)
    normalized = BatchNormalization(name='pool_norm_' + name)(downsampled)
    dropped = Dropout(dropout_pb)(normalized)
    deepened = Convolution2D(nb_filters, patch_size, patch_size,
                             name='deep_' + name,
                             border_mode='same',
                             activation=LeakyReLU())(dropped)
    return residual_block(name, deepened, dropout_pb, nb_filters, patch_size)


def up_res(name, in_layer, par_layer, dropout_pb, nb_filters, patch_size):
    """
    Uplift U-net. Decreases feature map size, increases image size.

    Concat with previous layer. Feed it to residual block.
    """
    normalized = BatchNormalization(name='up_norm_' + name)(in_layer)
    dropped = Dropout(dropout_pb)(normalized)
    shallowed = Convolution2D(nb_filters, patch_size, patch_size,
                              name='shallow_' + name,
                              border_mode='same',
                              activation=LeakyReLU())(dropped)
    renormalized = BatchNormalization(name='shallow_norm_' + name)(shallowed)
    redropped = Dropout(dropout_pb)(renormalized)
    upsampled = upsample(redropped, par_layer, name)
    return residual_block(name, upsampled, dropout_pb, 2 * nb_filters,
                          patch_size)


def downsample(in_layer, layer_name):
    """Decrease image size."""
    return MaxPooling2D(pool_size=(2, 2), name=layer_name)(in_layer)


def upsample(in_layer, par_layer, layer_name):
    """Increase image size and concat with other layer."""
    upsample_name = 'up_' + layer_name
    merge_name = 'concat_' + layer_name
    return merge([UpSampling2D(size=(2, 2), name=upsample_name)(in_layer),
                  par_layer], mode='concat', concat_axis=3, name=merge_name)


model_weights = ('/home/bfilippov/whiteboard/models/weights/'
                 'big_unet_resnet_detector_jaccard_v2.hf5')
nb_cv_features = 3
input_height = 160
input_width = 160
nb_filters = 16
patch_size = 3
dropout_pb = 0.5

img_size = (input_height, input_width)
img_dir = '/ssd/whiteboard/source'
batch_size = 64
epoch_size = 1024


input_layer = Input(shape=(None, None, nb_cv_features),
                    name='input_layer')

downsampled = downsample(input_layer, 'pool_I')
normalized = BatchNormalization(name='pool_norm_I')(downsampled)
dropped = Dropout(dropout_pb)(normalized)
deepened = Convolution2D(nb_filters, patch_size, patch_size, name='deep_I',
                         border_mode='same',
                         activation=LeakyReLU())(dropped)

conv1 = residual_block('I', deepened, dropout_pb, nb_filters, patch_size)


conv2 = down_res('II', conv1, dropout_pb, scale(nb_filters, 1), patch_size)

conv3 = down_res('III', conv2, dropout_pb, scale(nb_filters, 2), patch_size)

conv4 = down_res('IV', conv3, dropout_pb, scale(nb_filters, 3), patch_size)

conv5 = down_res('V', conv4, dropout_pb, scale(nb_filters, 4), patch_size)

conv6 = up_res('VI', conv5, conv4, dropout_pb, scale(nb_filters, 3),
               patch_size)

conv7 = up_res('VII', conv6, conv3, dropout_pb, scale(nb_filters, 2),
               patch_size)

conv8 = up_res('VIII', conv7, conv2, dropout_pb, scale(nb_filters, 1),
               patch_size)

conv9 = up_res('IX', conv8, conv1, dropout_pb, nb_filters, patch_size)

final_upsampling = UpSampling2D(size=(2, 2), name='up_X')(conv9)

final_batch_norm = BatchNormalization(name='norm_X')(final_upsampling)
final_dropout = Dropout(dropout_pb)(final_batch_norm)

# compute probability of corner for
# each patch using aggregated feature maps.
output_layer = Convolution2D(1, patch_size, patch_size, border_mode='same',
                             activation='sigmoid',
                             name='sigmoid_conv')(final_dropout)

model = Model(input=input_layer, output=output_layer)

if os.path.exists(model_weights):
    model.load_weights(model_weights, by_name=True)

training_log_path = 'training.log'
initial_epoch = 0
initial_learning_rate = 0.001
if os.path.exists(training_log_path):
    try:
        training_log = pd.read_csv(training_log_path)
        if training_log.shape[0] > 0:
            initial_epoch = training_log.iloc[-1]['epoch'] + 1
            initial_learning_rate = training_log.iloc[-1]['lr']
    except pd.io.common.EmptyDataError:
        pass


optimizer = keras.optimizers.Adam(lr=initial_learning_rate)
model.compile(optimizer=optimizer, loss=jaccard_loss,
              metrics=[jaccard_coef, jaccard_coef_int])

train_imgs = whiteboard_images(train, img_dir, img_size,
                                     batch_size=batch_size)
valid_imgs = whiteboard_images(valid, img_dir, img_size,
                                     batch_size=batch_size, seed=1313)
test_imgs = whiteboard_images(test, img_dir, img_size,
                                    batch_size=batch_size)

checkpoint = keras.callbacks.ModelCheckpoint(model_weights,
                                             save_weights_only=True,
                                             save_best_only=True)
early_stopping = keras.callbacks.EarlyStopping(patience=50)
tensor_board = keras.callbacks.TensorBoard(histogram_freq=5)
reduce_lr = keras.callbacks.ReduceLROnPlateau()
logger = keras.callbacks.CSVLogger('training.log', append=True)

history = model.fit_generator(train_imgs, samples_per_epoch=epoch_size,
                              nb_epoch=2000, validation_data=valid_imgs,
                              initial_epoch=initial_epoch,
                              nb_val_samples=64,
                              callbacks=[checkpoint, early_stopping,
                                         tensor_board, reduce_lr, logger])

print(history)
