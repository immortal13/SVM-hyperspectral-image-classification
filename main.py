# -*- coding: utf-8 -*-
# Torch
import torch
import torch.utils.data as data
# Numpy, scipy, scikit-image, spectral
import time
import numpy as np
import sklearn.svm
import sklearn.model_selection
from skimage import io
import scipy.io as sio
import seaborn as sns
import os
from utils import (
    metrics,
    convert_to_color_,
    convert_from_color_,
    sample_gt,
    build_dataset,
    show_results,
    get_device,
    Draw_Classification_Map
)
from datasets import get_dataset, open_file, DATASETS_CONFIG
import argparse

dataset_names = [
    v["name"] if "name" in v.keys() else k for k, v in DATASETS_CONFIG.items()
]

# Argument parser for CLI interaction
parser = argparse.ArgumentParser(
    description="Run deep learning experiments on" " various hyperspectral datasets"
)
parser.add_argument(
    "--dataset", type=str, default="IndianPines", choices=dataset_names, help="Dataset to use."
)
parser.add_argument(
    "--model",
    type=str,
    default="SVM_grid",
    help="Model to train. Available:\n"
    "SVM (linear), "
    "SVM_grid (grid search on linear, poly and RBF kernels), "
)
parser.add_argument(
    "--cuda",
    type=int,
    default=-1,
    help="Specify CUDA device (defaults to -1, which learns on CPU)",
)
parser.add_argument("--runs", type=int, default=5, help="Number of runs (default: 1)")
# Dataset options
group_dataset = parser.add_argument_group("Dataset")
group_dataset.add_argument(
    "--training_sample",
    type=float,
    default=30,
    help="Percentage of samples to use for training (default: 10%%)",
)
group_dataset.add_argument(
    "--sampling_mode",
    type=str,
    help="Sampling mode" " (random sampling or disjoint, default: random)",
    default="fixed", ## fixed or disjoint
)
group_dataset.add_argument(
    "--train_set",
    type=str,
    default=None,
    help="Path to the train ground truth (optional, this "
    "supersedes the --sampling_mode option)",
)
group_dataset.add_argument(
    "--test_set",
    type=str,
    default=None,
    help="Path to the test set (optional, by default "
    "the test_set is the entire ground truth minus the training)",
)

args = parser.parse_args()
CUDA_DEVICE = get_device(args.cuda)
# % of training samples
SAMPLE_PERCENTAGE = args.training_sample
DATASET = args.dataset
MODEL = args.model
N_RUNS = args.runs
# Sampling mode, e.g random sampling
SAMPLING_MODE = args.sampling_mode
# Training ground truth file
TRAIN_GT = args.train_set
# Testing ground truth file
TEST_GT = args.test_set


hyperparams = vars(args)
# Load the dataset
img, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(DATASET)
# Number of classes
N_CLASSES = len(LABEL_VALUES)
# Number of bands (last dimension of the image tensor)
N_BANDS = img.shape[-1]

# Parameters for the SVM grid search
SVM_GRID_PARAMS = [
    {"kernel": ["rbf"], "gamma": [1e-1, 1e-2, 1e-3], "C": [1, 10, 100, 1000]},
    {"kernel": ["linear"], "C": [0.1, 1, 10, 100, 1000]},
    {"kernel": ["poly"], "degree": [3], "gamma": [1e-1, 1e-2, 1e-3]},
]

if palette is None:
    # Generate color palette
    palette = {0: (0, 0, 0)}
    for k, color in enumerate(sns.color_palette("hls", len(LABEL_VALUES) - 1)):
        palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype="uint8"))
invert_palette = {v: k for k, v in palette.items()}


def convert_to_color(x):
    return convert_to_color_(x, palette=palette)


def convert_from_color(x):
    return convert_from_color_(x, palette=invert_palette)


# Instantiate the experiment based on predefined networks
hyperparams.update(
    {
        "n_classes": N_CLASSES,
        "n_bands": N_BANDS,
        "ignored_labels": IGNORED_LABELS,
        "device": CUDA_DEVICE,
    }
)
hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

results = []
AA_list = []
tr_time = []
te_time = []
# run the experiment several times
for run in range(N_RUNS):
    if TRAIN_GT is not None and TEST_GT is not None:
        train_gt = open_file(TRAIN_GT)['tr']
        test_gt = open_file(TEST_GT)['te']
    elif TRAIN_GT is not None:
        train_gt = open_file(TRAIN_GT)
        test_gt = np.copy(gt)
        w, h = test_gt.shape
        test_gt[(train_gt > 0)[:w, :h]] = 0
    elif TEST_GT is not None:
        test_gt = open_file(TEST_GT)
    else:
        print(SAMPLING_MODE,'SAMPLING_MODE')
        # Sample random training spectra
        train_gt, test_gt = sample_gt(gt, SAMPLE_PERCENTAGE, mode=SAMPLING_MODE)
    print(
        "{} samples selected (over {})".format(
            np.count_nonzero(train_gt), np.count_nonzero(gt)
        )
    )
    print(
        "Running an experiment with the {} model".format(MODEL),
        "run {}/{}".format(run + 1, N_RUNS),
    )

    
    if MODEL == "SVM_grid":
        print("Running a grid search SVM")
        # Grid search SVM (linear and RBF)
        X_train, y_train = build_dataset(img, train_gt, ignored_labels=IGNORED_LABELS)
        clf = sklearn.svm.SVC()
        clf = sklearn.model_selection.GridSearchCV(
            clf, SVM_GRID_PARAMS, verbose=1, n_jobs=4
        )
        tic = time.time()
        clf.fit(X_train, y_train)
        toc = time.time()
        print("training time: ", toc-tic)
        tr_time.append(toc-tic)
        print("SVM best parameters : {}".format(clf.best_params_))

        tic = time.time()
        prediction = clf.predict(img.reshape(-1, N_BANDS))
        toc = time.time()
        print("test time: ", toc-tic)
        te_time.append(toc-tic)
        prediction = prediction.reshape(img.shape[:2])
        Draw_Classification_Map(prediction,"{}_SVM_grid".format(DATASET))
        
    elif MODEL == "SVM":
        X_train, y_train = build_dataset(img, train_gt, ignored_labels=IGNORED_LABELS)
        clf = sklearn.svm.SVC()
        clf.fit(X_train, y_train)
        prediction = clf.predict(img.reshape(-1, N_BANDS))
        prediction = prediction.reshape(img.shape[:2])
        Draw_Classification_Map(prediction,"{}_SVM".format(DATASET))
    
    run_results, AA = metrics(
        prediction,
        test_gt,
        ignored_labels=hyperparams["ignored_labels"],
        n_classes=N_CLASSES,
    )

    mask = np.zeros(gt.shape, dtype="bool")
    for l in IGNORED_LABELS:
        mask[gt == l] = True
    prediction[mask] = 0

    color_prediction = convert_to_color(prediction)
    
    results.append(run_results)
    AA_list.append(AA)

if N_RUNS > 1:
    print("-"*20," results ","-"*20)
    show_results(results,  label_values=LABEL_VALUES, agregated=True)
    print(np.mean(np.array(AA_list),0),"AA_mean")
    print(np.std(np.array(AA_list),0),"AA_std")
    print(np.mean(np.array(tr_time),0),"tr time")
    print(np.mean(np.array(te_time),0),"te time")

