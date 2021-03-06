# Implmentation of AlexNet model taken from
# https://www.mydatahack.com/building-alexnet-with-keras/
# script that reads data, creates model and trains it

from time import sleep
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten,\
    Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
import numpy as np
import directory_structure
import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from collections import deque
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf
import models

# classes model needs to learn to classify
CLASSES_TO_CHECK = ['L', 'N', 'V', 'A', 'R']
NUMBER_OF_CLASSES = len(CLASSES_TO_CHECK)
IMAGES_TO_TRAIN = 2544  # total number images in class A

# removing warning for tensorflow about AVX support
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def saveMetricsAndWeights(score, model, metric_name, model_name):
    '''
    Save metric and weight data in specific folders

    Args:
        score (list): list containing loss value and accuracy of current model

        model (keras model): compiled model which has been trained with training data
        Returns:
                (dataframe): dataframe contatining image information 
        '''
    loss = score[0]
    current_acc = score[1]

    directory_structure.getWriteDirectory('testing', None)
    weights_path = directory_structure.getWriteDirectory(
        'testing', 'model_weights')
    metrics_path = directory_structure.getWriteDirectory(
        'testing', 'accuracy_metrics')

    if (len(directory_structure.filesInDirectory(metric_name, metrics_path)) == 0):
        # create text file with placeholder accuracy value (i.e 0)
        np.save(metrics_path + metric_name, [0])
        model.save(weights_path + model_name)
        del model
    else:
        highest_acc = np.load(metrics_path + metric_name)[0]
        if (current_acc > highest_acc):
            np.save(metrics_path + metric_name, [current_acc])
            model.save(weights_path + model_name)
            del model
            print('\nAccuracy Increase: ' +
                  str((current_acc - highest_acc)*100) + '%')


def getSignalDataFrame():
    '''
        read signal images present in the directory beat_write_dir
    and save them in a dataframe

        Returns:
                (dataframe): dataframe contatining image information 
        '''
    # get paths for where signals are present
    signal_path = directory_structure.getWriteDirectory('beat_write_dir', None)
    print(f"Signal path: {signal_path}")

    # create dataframe
    df = pd.DataFrame(columns=['Signal ID', 'Signal', 'Type'])

    arrhythmia_classes = directory_structure.getAllSubfoldersOfFolder(
        signal_path)
    print(f"arrhythmia classes: {arrhythmia_classes}")

    image_paths = deque()
    image_ids = deque()
    class_types = deque()
    images = []

    # get path for each image in classification folders
    for classification in arrhythmia_classes:
        classification_path = ''.join([signal_path, classification])
        print(f"Classification path: {classification_path}")
        image_list = directory_structure.filesInDirectory(
            '.png', classification_path)
        print(
            "length of images in class {} is {}".format(classification_path, len(image_list)))
        for beat_id in image_list:
            image_ids.append(directory_structure.removeFileExtension(beat_id))
            class_types.append(classification)
            image_paths.append(''.join([classification_path, '/', beat_id]))

    # read and save images in dataframe
    for path in image_paths:
        images.append(cv2.imread(path))

    print(f"Class type: {class_types}")

    # save information in dataframe
    df['Signal ID'] = image_ids
    df['Type'] = class_types
    df['Signal'] = images

    return df


def normalizeData(X_train, X_test, y_train, y_test):
    '''
    Normalizing the test and train data
    '''

    # image normalization
    X_train = X_train.astype('float32')
    X_train = X_train / 255
    X_test = X_test.astype('float32')
    X_test = X_test / 255

    # label normalization
    y_train = keras.utils.to_categorical(y_train, NUMBER_OF_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUMBER_OF_CLASSES)

    return X_train, X_test, y_train, y_test


def convertToNumpy(X_train, X_test, y_train, y_test):
    '''
    Convert data arrays into numpy arrays
    '''
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


def trainAndTestSplit(df, size_of_test_data):
    '''
        take dataframe and divide it into train and 
    test data for model training

        Args:
                df (dataframe): dataframe with all images information

        images_to_train (int): number of images to get for training
                               from dataframe

        size_of_test_data (float): percentage of data specified for training

        Returns:
                X_train (list): list of training signals

        X_test (list): list of testing signals

        y_train (list): list of training classes

        y_test (list): list of testing classes
        '''
    image_count = 0
    classes_to_check = CLASSES_TO_CHECK.copy()
    images_available_in_class = IMAGES_TO_TRAIN

    # train + test data (signals and classes of signals respectively)
    X = []
    y = []

    for index, row in df.iterrows():
        # check if current row is one of the classes to classify
        if row['Type'] in classes_to_check:
            images_available_in_class = df['Type'].value_counts()[row['Type']]

            X.append(row['Signal'])
            y.append(CLASSES_TO_CHECK.index(str(row['Type'])))

            image_count += 1

            if images_available_in_class < IMAGES_TO_TRAIN:
                if image_count == df['Type'].value_counts()[row['Type']]:

                    image_count = 0
                    classes_to_check.remove(row['Type'])
            else:
                if image_count == IMAGES_TO_TRAIN:

                    image_count = 0
                    classes_to_check.remove(row['Type'])

        # if data collected from all classes break loop
        if len(classes_to_check) == 0:
            break

    # split x and y into train and test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=size_of_test_data)

    # convert to numpy array
    X_train, X_test, y_train, y_test = convertToNumpy(
        X_train, X_test, y_train, y_test)

    # normalize data for easy data processing
    X_train, X_test, y_train, y_test = normalizeData(
        X_train, X_test, y_train, y_test)

    return X_train, X_test, y_train, y_test


def printTestMetrics(score):
    '''
    print prediction score

    Args:
        score (list): list with test loss and test accuracy
    '''
    print()
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def createModel(model_name):
    '''
    Implementation of model to train images (Alexnet or Novelnet)

    Args:
        model_name (str): name of the model to create (can choose from Alexnet and Novelnet)

    Returns:
        model (model): model object implementation of alexnet
    '''
    model = Sequential()

    if model_name == 'Alexnet':
        # -----------------------1st Convolutional Layer--------------------------
        model.add(Conv2D(filters=96, input_shape=(224, 224, 3), kernel_size=(11, 11),
                         strides=(4, 4), padding='valid'))
        model.add(Activation('relu'))
        # Pooling
        model.add(MaxPooling2D(pool_size=(2, 2),
                               strides=(2, 2), padding='valid'))
        # Batch Normalisation before passing it to the next layer
        model.add(BatchNormalization())

        # -----------------------2nd Convolutional Layer---------------------------
        model.add(Conv2D(filters=256, kernel_size=(
            11, 11), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        # Pooling
        model.add(MaxPooling2D(pool_size=(2, 2),
                               strides=(2, 2), padding='valid'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # -----------------------3rd Convolutional Layer----------------------------
        model.add(Conv2D(filters=384, kernel_size=(
            3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # -----------------------4th Convolutional Layer----------------------------
        model.add(Conv2D(filters=384, kernel_size=(
            3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # -----------------------5th Convolutional Layer----------------------------
        model.add(Conv2D(filters=256, kernel_size=(
            3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        # Pooling
        model.add(MaxPooling2D(pool_size=(2, 2),
                               strides=(2, 2), padding='valid'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # Passing it to a dense layer
        model.add(Flatten())
        # -------------------------1st Dense Layer----------------------------
        model.add(Dense(4096, input_shape=(224*224*3,)))
        model.add(Activation('relu'))
        # Add Dropout to prevent overfitting
        model.add(Dropout(0.4))
        # Batch Normalisation
        model.add(BatchNormalization())

        # -------------------------2nd Dense Layer---------------------------
        model.add(Dense(4096))
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.4))
        # Batch Normalisation
        model.add(BatchNormalization())

        # -------------------------3rd Dense Layer---------------------------
        model.add(Dense(1000))
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.4))
        # Batch Normalisation
        model.add(BatchNormalization())

        # --------------------------Output Layer-----------------------------
        model.add(Dense(NUMBER_OF_CLASSES, activation='softmax'))

    elif model_name == 'Novelnet':
        # -----------------------1st Convolutional Layer--------------------------
        model.add(Conv2D(filters=96, input_shape=(224, 224, 3), kernel_size=(13, 13),
                         strides=(4, 4), padding='valid'))
        model.add(Activation('relu'))
        # Pooling
        model.add(MaxPooling2D(pool_size=(2, 2),
                               strides=(2, 2), padding='valid'))
        # Batch Normalisation before passing it to the next layer
        model.add(BatchNormalization())

        # -----------------------2nd Convolutional Layer---------------------------
        model.add(Conv2D(filters=256, kernel_size=(
            11, 11), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        # Pooling
        model.add(MaxPooling2D(pool_size=(2, 2),
                               strides=(2, 2), padding='valid'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # -----------------------3rd Convolutional Layer----------------------------
        model.add(Conv2D(filters=384, kernel_size=(
            3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # -----------------------4th Convolutional Layer----------------------------
        model.add(Conv2D(filters=384, kernel_size=(
            3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # -----------------------5th Convolutional Layer----------------------------
        model.add(Conv2D(filters=384, kernel_size=(
            3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # -----------------------6th Convolutional Layer----------------------------
        model.add(Conv2D(filters=512, kernel_size=(
            3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))
        # Pooling
        model.add(MaxPooling2D(pool_size=(2, 2),
                               strides=(2, 2), padding='valid'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # Passing it to a dense layer
        model.add(Flatten())
        # -------------------------1st Dense Layer----------------------------
        model.add(Dense(4096, input_shape=(224*224*3,)))
        model.add(Activation('relu'))
        # Add Dropout to prevent overfitting
        model.add(Dropout(0.4))
        # Batch Normalisation
        model.add(BatchNormalization())

        # -------------------------2nd Dense Layer---------------------------
        model.add(Dense(4096))
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.6))
        # Batch Normalisation
        model.add(BatchNormalization())

        # -------------------------3rd Dense Layer---------------------------
        model.add(Dense(1000))
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.5))
        # Batch Normalisation
        model.add(BatchNormalization())

        # --------------------------Output Layer-----------------------------
        model.add(Dense(NUMBER_OF_CLASSES, activation='softmax'))

    return model


if __name__ == '__main__':
    from utils import save_train_test_data

    # (2) GET DATA
    df = getSignalDataFrame()

    X_train, X_test, y_train, y_test = trainAndTestSplit(df, 0.2)

    # save train data from dataframe to image
    save_train_test_data(X_train, y_train, data_type="train")
    # save test data from dataframe to image
    save_train_test_data(X_test, y_test, data_type="test")

    # VGG16 model

    print("Training VGG16 model")
    vgg16_model = models.vgg16(
        X_train, y_train, X_test, y_test, batch_size=64, epochs=20, num_classes=5, output_file="resnet152.h5")

    predictions = vgg16_model.predict(X_test)
    score = vgg16_model.evaluate(X_test, y_test, verbose=0)

    printTestMetrics(score)

    saveMetricsAndWeights(score, vgg16_model, "vgg16.npy", "vgg16.h5")

    # VGG19 model
    print("Training VGG16 model")
    vgg19_model = models.vgg19(
        X_train, y_train, X_test, y_test, batch_size=64, epochs=20, num_classes=5, output_file="resnet152.h5")

    predictions = vgg19_model.predict(X_test)
    score = vgg19_model.evaluate(X_test, y_test, verbose=0)

    printTestMetrics(score)

    saveMetricsAndWeights(score, vgg19_model, "vgg19.npy", "vgg19.h5")
