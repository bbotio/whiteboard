"""Just to ensure generator is working."""
import ast
import os
import numpy as np
import pandas as pd
import cv2

from shapely.geometry import MultiPoint
from models import image_generator


def test_random_transform():
    """Load image and transforms it, then check that labels are OK."""
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
    labels = np.zeros((4, 2))
    for i in range(4):
        labels[i, 0] = xn[i]
        labels[i, 1] = yn[i]
    img = cv2.imread(image_path)
    kw = dict(rotation_range=15,
              height_shift_range=0.2,
              width_shift_range=0.2,
              shear_range=0.3,
              channel_shift_range=0.2,
              horizontal_flip=True,
              vertical_flip=True,
              dim_ordering='tf',
              seed=1313)
    # when
    rimg, rlabels = image_generator.random_transform(img, labels, **kw)

    # then just assert transformation isn't changed much
    assert MultiPoint([[224.91875347, 58.05657097],
                       [673.57648317, 189.27244333],
                       [544.23308452, 381.12743459],
                       [70.73339963, 312.7359806]]
                      ).equals_exact(MultiPoint(rlabels), 5)
