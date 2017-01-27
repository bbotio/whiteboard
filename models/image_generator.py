"""Code to load and transform whiteboard images."""
import itertools
import os
import numpy as np
import cv2

from keras.preprocessing.image import transform_matrix_offset_center
from keras.preprocessing.image import random_channel_shift
from keras.preprocessing.image import flip_axis
from shapely.geometry import MultiPoint
from shapely import affinity
from skimage import img_as_float


whiteboard_label_len = 10


def whiteboard_label(labels):
    """
    Parse single image labels into whiteboard label.

    Args:
        labels: labels of single image

    Returns:
        (whiteboard_present, whiteboard_color,
        [[x1 y1] [x2 y2] [x3 y3] [x4 y4]])
    """
    def rectangle_coords(xs):
        return [int(float(x)) for x in xs.split(';')][:4]

    for label in labels:
        if label['class'] == 'whiteboard':
            xn = rectangle_coords(label['xn'])
            yn = rectangle_coords(label['yn'])
            return (1, int(label.get('id', 'white') == 'white'),
                    list(zip(xn, yn)))
        return 0, 0, list(zip([0] * 4, [0] * 4))


def whiteboard_images(train, img_dir, image_size, batch_size=32, seed=None):
    """
    Endless iterator over whiteboard images.

    Args:
        train: train dataframe with 'path' and decoded 'labels' columns
        img_dir: root directory for paths in train dataframe
        image_size: final image size
        batch_size: batch size
        seed: set to make generator behavior reproducible

    Returns:
        iterator of (batch_x, batch_y) pairs where batch_x is
        batched loaded images and batch_y are batched whiteboard labels
    """
    transform_opts = dict(rotation_range=15,
                          height_shift_range=0.2,
                          width_shift_range=0.2,
                          shear_range=0.3,
                          channel_shift_range=0.2,
                          horizontal_flip=True,
                          vertical_flip=True,
                          dim_ordering='tf',
                          seed=seed)
    n_channels = 3

    def generator():
        global whiteboard_label_len
        batch_x = None
        batch_y = None
        for ri, row in itertools.cycle(train.iterrows()):
            i = ri % batch_size
            if i == 0:
                if batch_x is not None:
                    yield batch_x, batch_y
                batch_x = np.zeros((batch_size,) + image_size + (n_channels,))
                batch_y = np.zeros((batch_size,) + (whiteboard_label_len,))
            img_path = os.path.join(img_dir, row['path'])
            img = cv2.imread(img_path)
            if img is None:
                print("Can't read ", img_path)
                continue
            is_present, color, labels = whiteboard_label(row['labels'])
            img, labels = random_transform(img, labels, **transform_opts)
            scale_y = image_size[0] / img.shape[0]
            scale_x = image_size[1] / img.shape[1]
            img = cv2.resize(img, (image_size[1], image_size[0]))
            labels = affinity.scale(labels, xfact=scale_x, yfact=scale_y,
                                    origin=(0, 0))
            labels = np.asarray(labels)
            x_labels = labels[:, 0]
            y_labels = labels[:, 1]

            label_vec = np.zeros(whiteboard_label_len, np.float32)
            label_vec[0] = is_present
            label_vec[1] = color
            label_vec[2:6] = x_labels
            label_vec[6:10] = y_labels
            batch_x[i] = img
            batch_y[i] = label_vec
    return generator()


def random_transform(image, labels,
                     rotation_range=None,
                     height_shift_range=None,
                     width_shift_range=None,
                     shear_range=None,
                     zoom_range=(1, 1),
                     channel_shift_range=0,
                     horizontal_flip=False,
                     vertical_flip=False,
                     dim_ordering='tf', seed=None):
    """
    Random transformation over given image with labels.

    Stolen from keras.preprocessing.image.ImageDataGenerator and
    repurposed to properly transform image labels together with image.

    Args:
        image: image as numpy array of shape (x, y, n_channels)
        labels: labels as [[x1 y1] [x2 y2] [x3 y3] [x4 y4]] numpy array
        kwargs: see keras.preprocessing.image.ImageDataGenerator

    Returns:
        (transformed_image, transformed_labels as MultiPoint)
    """
    if seed:
        np.random.seed(seed)
    # image is a single image, so it doesn't have image number at index 0
    if dim_ordering == 'th':
        channel_index = 1
        row_index = 2
        col_index = 3
    if dim_ordering == 'tf':
        channel_index = 3
        row_index = 1
        col_index = 2
    img_row_index = row_index - 1
    img_col_index = col_index - 1
    img_channel_index = channel_index - 1

    # use composition of homographies to generate final transform that
    # needs to be applied
    if rotation_range:
        theta = np.pi / 180 * np.random.uniform(-rotation_range,
                                                rotation_range)
    else:
        theta = 0
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    if height_shift_range:
        tx = np.random.uniform(-height_shift_range,
                               height_shift_range) * image.shape[img_row_index]
    else:
        tx = 0

    if width_shift_range:
        ty = np.random.uniform(-width_shift_range,
                               width_shift_range) * image.shape[img_col_index]
    else:
        ty = 0

    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])
    if shear_range:
        shear = np.random.uniform(-shear_range, shear_range)
    else:
        shear = 0
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    transform_matrix = np.dot(np.dot(np.dot(rotation_matrix,
                                            translation_matrix), shear_matrix),
                              zoom_matrix)

    h, w = image.shape[img_row_index], image.shape[img_col_index]
    transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)

    labels, corners, new_corners = transform_labels(labels, image.shape,
                                                    transform_matrix)

    image = img_as_float(image)
    perspective_transform = cv2.getPerspectiveTransform(np.float32(corners),
                                                        np.float32(new_corners)
                                                        )
    new_size = tuple(map(int, new_corners.ptp(0)))
    image = cv2.warpPerspective(image, perspective_transform, new_size)
    image_center = (new_size[0] / 2, new_size[1] / 2)

    if channel_shift_range != 0:
        image = random_channel_shift(image, channel_shift_range,
                                     img_channel_index)

    if horizontal_flip:
        if np.random.random() < 0.5:
            image = flip_axis(image, img_col_index)
            labels = affinity.scale(labels, xfact=-1, origin=image_center)

    if vertical_flip:
        if np.random.random() < 0.5:
            image = flip_axis(image, img_row_index)
            labels = affinity.scale(labels, yfact=-1,
                                    origin=image_center)

    # TODO:
    # channel-wise normalization
    # barrel/fisheye
    return image, labels


def transform_labels(labels, image_shape, matrix):
    r"""
    Perform affine transformation on labels.

    It's done by performing affine transformation on labels,
    then performing same transformation on image corners:
        (0, 0) (0, y), (x, 0), (x, y)
    Then transformed labels are translated to be non-negative

    Args:
        labels: numpy array of shape (N, 2)
        image_shape: image width and height
        matrix: affine transformation matrix like:
            / a b xoff \.
            | d e yoff |
            \ 0 0 1    /

    Returns:
        MultiPoint transformed labels,
        original corners (numpy array) and new corners (also numpy array)
    """
    x_ix = 1
    y_ix = 0
    img_x = image_shape[x_ix]
    img_y = image_shape[y_ix]
    labels = MultiPoint(labels)
    corners = MultiPoint([(0, 0),
                          (0, img_y),
                          (img_x, 0),
                          (img_x, img_y)])

    transform_vec = [matrix[0, 0],
                     matrix[0, 1],
                     matrix[1, 0],
                     matrix[1, 1],
                     matrix[0, 2],
                     matrix[1, 2]]
    labels = affinity.affine_transform(labels, transform_vec)
    transformed_corners = affinity.affine_transform(corners, transform_vec)
    transformed_corners = np.asarray(transformed_corners)
    min_xy = transformed_corners.min(0)
    transformed_corners[:, 0] -= min_xy[0]
    transformed_corners[:, 1] -= min_xy[1]
    labels = affinity.translate(labels, xoff=-min_xy[0], yoff=-min_xy[1])
    return labels, np.asarray(corners), transformed_corners
