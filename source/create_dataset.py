#!/bin/env python
import json
import random

import pandas as pd

def create_dataset(seed, train_test_ratio=5.0):
    with open('labels.json') as f:
        labels = json.load(f)
        labels = [x for x in labels if x['annotations'] != []]
    random.shuffle(labels)

    n_samples = len(labels)
    n_test = int(n_samples / (1.0 + train_test_ratio))

    test = pd.DataFrame(columns=['path', 'labels'])
    train = pd.DataFrame(columns=['path', 'labels'])
    print("Creating test set:")
    for i in range(n_test):
        test.loc[i] = [labels[i]['filename'], labels[i]['annotations']]

    print("Creating training set:")
    for i in range(n_test, n_samples):
        train.loc[i] = [labels[i]['filename'], labels[i]['annotations']]
    return train, test


if __name__ == "__main__":
    seed = 1313
    random.seed(seed)
    train, test = create_dataset(seed)
    train.to_csv('train.csv', index_label='id')
    test.to_csv('test.csv', index_label='id')
