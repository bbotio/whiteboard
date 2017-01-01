#!/bin/env python
import json


if __name__ == "__main__":
    with open('labels.json') as f:
        labels = json.load(f)

    print("Number of labeled images:", len([x for x in labels
                                            if x['annotations'] != []]))
    print("Total number of images:", len(labels))
