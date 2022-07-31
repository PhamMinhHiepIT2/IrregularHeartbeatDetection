import os
import cv2
import numpy as np
import warnings
import random

train_test_folder = "/home/hieppm/hieppm/IrregularHeartbeatDetection/train-test-data"

TRAIN_FOLDER = "train"
TEST_FOLDER = "test"

# each class will take 2544 images
IMAGES_IN_CLASS = 2544  # total image of A class

CLASSES_TO_CHECK = ['L', 'N', 'V', 'A', 'R']


def get_train_test_img_index(image_folder: str, train_percent):
    label_class = os.path.basename(image_folder)
    train_indexes = []
    test_indexes = []
    total_class_images = len(os.listdir(image_folder))
    total_images_to_split = IMAGES_IN_CLASS
    if total_class_images < IMAGES_IN_CLASS:
        warnings.warn(
            f"Total images in class {label_class} less than {IMAGES_IN_CLASS}")
        total_images_to_split = total_class_images

    index_ranges = range(os.listdir(image_folder))
    index_ranges = random.shuffle(index_ranges)
    train_indexes = index_ranges[0:round(
        total_images_to_split * train_percent)]
    test_indexes = index_ranges[round(
        total_images_to_split * 0.8):total_images_to_split]

    return train_indexes, test_indexes


def save_train_test_data(X, y, data_type: str):
    if data_type.lower() not in ["train", "test"]:
        raise Exception("Data type must be in list {train, test}")
    if len(X) != len(y):
        raise Exception(
            "Number X (data) records are different from number y (label) records")
    for i in range(len(X)):
        data = X[i]
        label_index = y[i].tolist().index(1)
        label = CLASSES_TO_CHECK[label_index]
        folder_to_save = os.path.join(
            train_test_folder, data_type.lower(), label)
        os.makedirs(folder_to_save, exist_ok=True)
        img_path = os.path.join(folder_to_save, label + str(i) + ".jpeg")
        try:
            cv2.imwrite(img_path, data)
        except Exception as e:
            print(e)
            warnings.warn(
                f"Fail to save image with index {i} of class {label}")
