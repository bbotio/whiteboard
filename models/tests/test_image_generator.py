import pytest
from models import image_generator
import ast
import os
import numpy as np
import pandas as pd
from skimage import img_as_float
from matplotlib import pyplot as plt
from keras.preprocessing.image import load_img, img_to_array

%load_ext autoreload
%autoreload 2
%matplotlib inline

def test_random_transform():
    # given
    train = pd.read_csv('source/train.csv')
    train['labels'] = train['labels'].map(ast.literal_eval)
    image_path = os.path.join('source', train.iloc[0].path)
    all_labels = train.iloc[0]['labels']
    for label in all_labels:
        if label['class'] == 'whiteboard':
            break
    xn = [int(float(x)) for x in label['xn'].split(';')][:4]
    yn = [int(float(y)) for y in label['yn'].split(';')][:4]
    labels = np.zeros((4,2))
    for i in range(4):
        labels[i, 0] = xn[i]
        labels[i, 1] = yn[i]
    img = load_img(image_path)
    img = img_as_float(img)
    kw = dict(rotation_range=20,
            height_shift_range=0,
            width_shift_range=0,
            shear_range=0.2,
            fill_mode='constant',
            cval=0,
            zoom_range=(1.3, 1.3),
            channel_shift_range=0,
            horizontal_flip=False,
            vertical_flip=False,
            dim_ordering='tf')
    # when
    rimg, rlabels = image_generator.random_transform(img, labels, **kw)

    # then

plt.imshow(img)
plt.plot(labels[:, 0], labels[:, 1], 'o')
plt.imshow(rimg)
plt.plot(rlabels[:, 0], rlabels[:, 1], 'o')
