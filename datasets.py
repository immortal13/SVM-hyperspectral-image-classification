# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.utils
import torch.utils.data
import os

try:
    # Python 3
    from urllib.request import urlretrieve
except ImportError:
    # Python 2
    from urllib import urlretrieve

from utils import open_file

DATASETS_CONFIG = {
    "IndianPines": {
        "urls": [
            "http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat",
            "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat",
        ],
        "img": "Indian_pines_corrected.mat",
        "gt": "Indian_pines_gt.mat",
    },
}

def get_dataset(dataset_name, datasets=DATASETS_CONFIG):
    palette = None
    if dataset_name not in datasets.keys():
        raise ValueError("{} dataset is unknown.".format(dataset_name))

    dataset = datasets[dataset_name]

    if dataset_name == "IndianPines":
        # Load the image
        img = open_file("Datasets/IndianPines/Indian_pines_corrected.mat")["indian_pines_corrected"]
        gt = open_file("Datasets/IndianPines/Indian_pines_gt.mat")["indian_pines_gt"]
        rgb_bands = (43, 21, 11)  # AVIRIS sensor
        label_values = [
            "Undefined",
            "Alfalfa",
            "Corn-notill",
            "Corn-mintill",
            "Corn",
            "Grass-pasture",
            "Grass-trees",
            "Grass-pasture-mowed",
            "Hay-windrowed",
            "Oats",
            "Soybean-notill",
            "Soybean-mintill",
            "Soybean-clean",
            "Wheat",
            "Woods",
            "Buildings-Grass-Trees-Drives",
            "Stone-Steel-Towers",
        ]
        ignored_labels = [0]
    else:
        print("No data")
    # Filter NaN out
    nan_mask = np.isnan(img.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
        print(
            "Warning: NaN have been found in the data. It is preferable to remove them beforehand. Learning on NaN data is disabled."
        )
    img[nan_mask] = 0
    gt[nan_mask] = 0
    ignored_labels.append(0)

    ignored_labels = list(set(ignored_labels))
    # Normalization
    img = np.asarray(img, dtype="float32")
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img, gt, label_values, ignored_labels, rgb_bands, palette
