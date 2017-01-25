import itertools
import os
import tempfile
import numpy as np

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

whiteboard_label_len = 10

def sequential(input_tensor, *layers):
    """
        keras.models.Sequential analog for Input tensor

        Args:
            input_tensor: instance of keras.layers.Input
            which is bare tensor that can't be used in
            models.Sequential
            layers: rest of model layers

        Returns:
            all layers sequentially applied to input tensor
    """
    last_layer = input_tensor
    for layer in layers:
        last_layer = layer(last_layer)
    return last_layer


class Transformation(object):
    """
        Stores information about current image transformation:
        resizing and pixel remapping.
    """
    def __init__(self, original_size, final_size, datagen):
        """
            Constructs new transformation using given image generator.

            Args:
                original_size: original image size
                final_size: scaled image size
                datagen: image generator
        """
        self.original_size = original_size
        self.final_size = final_size
        self.x_scale = final_size[0] / original_size[0]
        self.y_scale = final_size[1] / original_size[1]

        identity = np.arange(final_size[0] * final_size[1]).reshape(final_size)
        self.transformation = datagen.random_transform(identity).reshape(final_size)

    def transform_label(self, xn, yn):
        """
            Takes label coords and applies transformation on them.

            Args:
                xn: [x1 x2 x3 x4] already scaled to final_size
                yn: [y1 y2 y3 y4] already scaled to final_size

            Returns:
                transformed (xn, yn)
        """
        mask = np.zeros(self.final_size)

        mask[xn, yn] = 1

        transformed_mask = self.transform_array(mask)
        xnn, ynn, _ = np.nonzero(transformed_mask)
        return xnn, ynn

    def transform_array(self, arr):
        """
            Transforms array according to given transformation vector
        """
        shape = arr.shape
        return arr.flatten()[self.transformation].reshape(shape)


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
        rotation_range=0,
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
            img = load_img(img_path, grayscale=grayscale)
            original_size = (img.size[0], img.size[1])
            img = img.resize((image_size[1], image_size[0]))
            img = img_to_array(img, dim_ordering='tf')

            transformation = Transformation(original_size, image_size, datagen)
            batch_x[i] = transformation.transform_array(img)
            batch_y[i] = whiteboard_label(row['labels'], transformation)
    return generator()
