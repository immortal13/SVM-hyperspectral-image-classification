# SVM-hyperspectral-image-classification
support vector machine (SVM) with grid search for hyperspectral image classification

## Setup
`pip install -r requirements.txt`

## Train and test
run the code using:

on CPU, 30 samples for each class

`python main.py --model SVM_grid --dataset IndianPines --training_sample 30`

on GPU, 30 samples for each class

`python main.py --model SVM_grid --dataset IndianPines --training_sample 30 --cuda`

## Record classification result
![classification accuracy](https://github.com/immortal13/SVM-hyperspectral-image-classification/assets/44193495/ad5ffb6f-8148-4198-9eb7-6ce2c7b68baf)
