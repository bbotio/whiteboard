"""
    Code to load and transform whiteboard images
"""
import itertools
import os
import tempfile
import numpy as np
import skimage.transform

from keras.preprocessing.image import load_img, img_to_array, apply_transform, transform_matrix_offset_center

whiteboard_label_len = 10


def whiteboard_label(labels, transformation):
    """
        Parses single image labels into whiteboard label.

        Args:
            labels: labels of single image
            x_scale: x labels scaling coefficient
            y_scale: y labels scaling coefficient

        Returns:
            [whiteboard_present whiteboard_color x1 x2 x3 x4 y1 y2 y3 y4]
    """
    global whiteboard_label_len
    result = np.zeros((whiteboard_label_len,))

    def rectangle_coords(xs, scale):
        return [int(float(x) * scale) for x in xs.split(';')][:4]

    for label in labels:
        if label['class'] == 'whiteboard':
            result[0] = 1
            result[1] = int(label.get('id', 'white') == 'white')
            xn = rectangle_coords(label['xn'], transformation.x_scale)
            yn = rectangle_coords(label['yn'], transformation.y_scale)
            xn, yn = transformation.transform_label(xn, yn)
            result[2:6] = xn
            result[6:10] = yn
            return result
        return result


def whiteboard_images(train, img_dir, image_size, batch_size=32, grayscale=True):
    """
        Endless iterator over whiteboard images.

        Args:
            train: train dataframe with 'path' and decoded 'labels' columns
            img_dir: root directory for paths in train dataframe
            image_size: final image size
            batch_size: batch size

        Returns:
            iterator of (batch_x, batch_y) pairs where batch_x is
            batched loaded images and batch_y are batched whiteboard labels
    """
    np.random.seed()
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        # TODO: more agressive image augmentation
        rotation_range=20,
        width_shift_range=0,
        height_shift_range=0,
        horizontal_flip=True,
        channel_shift_range=0,
        zoom_range=(1, 1),
        zca_whitening=False,
        fill_mode='constant',
        rescale=0
        )

    def generator():
        global whiteboard_label_len
        batch_x = None
        for ri, row in itertools.cycle(train.iterrows()):
            i = ri % batch_size
            if i == 0:
                if batch_x is not None:
                    yield batch_x, batch_y
                batch_x = np.zeros((batch_size,) + image_size)
                batch_y = np.zeros((batch_size,) + (whiteboard_label_len,))
            img_path = os.path.join(img_dir, row['path'])
            img = skimage.io.imread(img_path, grayscale=grayscale)
            original_size = (img.size[0], img.size[1])
            img = img.resize((image_size[1], image_size[0]))
            img = img_to_array(img)

            transformation = Transformation(original_size, image_size, datagen)
            batch_x[i] = transformation.transform_array(img)
            batch_y[i] = whiteboard_label(row['labels'], transformation)
    return generator()


def random_transform(image, labels,
                     rotation_range=None,
                     height_shift_range=None,
                     width_shift_range=None,
                     shear_range=None,
                     zoom_range=(1, 1),
                     fill_mode='nearest',
                     cval=0,
                     channel_shift_range=0,
                     horizontal_flip=False,
                     vertical_flip=False,
                     dim_ordering='tf'):
    """
        Random transformation over given image with labels.

        Stolen from keras.preprocessing.image.ImageDataGenerator and
        repurposed to properly transform image labels together with image.

        Args:
            image: image as numpy array of shape (x, y, n_channels)
            labels: labels as [[x1 y1] [x2 y2] [x3 y3] [x4 y4]] numpy array
            kwargs: see keras.preprocessing.image.ImageDataGenerator

        Returns:
            (transformed_image, transformed_labels)
    """
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

    # use composition of homographies to generate final transform that needs to be applied
    if rotation_range:
        theta = np.pi / 180 * np.random.uniform(-rotation_range, rotation_range)
    else:
        theta = 0
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    if height_shift_range:
        tx = np.random.uniform(-height_shift_range, height_shift_range) * image.shape[img_row_index]
    else:
        tx = 0

    if width_shift_range:
        ty = np.random.uniform(-width_shift_range, width_shift_range) * image.shape[img_col_index]
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

    transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)

    h, w = image.shape[img_row_index], image.shape[img_col_index]
    transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)

    new_labels = skimage.transform.matrix_transform(labels, transform_matrix)

    skimage_transform = skimage.transform.estimate_transform('similarity',
                                                             labels, new_labels)

    image = skimage.transform.warp(image, skimage_transform, order=3, mode=fill_mode,
                                   cval=cval)
    
    if channel_shift_range != 0:
        image = random_channel_shift(image, channel_shift_range, img_channel_index)

    if horizontal_flip:
        if np.random.random() < 0.5:
            image = flip_axis(image, img_col_index)

    if vertical_flip:
        if np.random.random() < 0.5:
            image = flip_axis(image, img_row_index)

    # TODO:
    # channel-wise normalization
    # barrel/fisheye
    return image, new_labels
