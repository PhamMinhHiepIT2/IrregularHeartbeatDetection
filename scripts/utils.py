import numpy as np
import keras
import cv2
import pandas as pd
import os
import re
from collections import deque
from sklearn.model_selection import train_test_split


from constants import NUMBER_OF_CLASSES, IMAGES_TO_TRAIN, CLASSES_TO_CHECK


def convertToNumpy(X_train, X_test, y_train, y_test):
    '''
    Convert data arrays into numpy arrays
    '''
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


def getSignalDataFrame():
    '''
        read signal images present in the directory beat_write_dir
    and save them in a dataframe

        Returns:
                (dataframe): dataframe contatining image information 
        '''
    # get paths for where signals are present
    signal_path = get_write_dir('beat_write_dir', None)

    # create dataframe
    df = pd.DataFrame(columns=['Signal ID', 'Signal', 'Type'])

    arrhythmia_classes = getAllSubfoldersOfFolder(
        signal_path)

    image_paths = deque()
    image_ids = deque()
    class_types = deque()
    images = []

    # get path for each image in classification folders
    for classification in arrhythmia_classes:
        classification_path = ''.join([signal_path, classification])
        image_list = files_in_dir(
            '.png', classification_path)
        for beat_id in image_list:
            image_ids.append(remove_extension(beat_id))
            class_types.append(classification)
            image_paths.append(''.join([classification_path, '/', beat_id]))

    # read and save images in dataframe
    for path in image_paths:
        images.append(cv2.imread(path))

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
    classes_to_check = CLASSES_TO_CHECK
    images_available_in_class = IMAGES_TO_TRAIN

    # train + test data (signals and classes of signals respectively)
    X = []
    y = []

    for index, row in df.iterrows():
        # check if current row is one of the classes to classify
        if row['Type'] in classes_to_check:
            images_available_in_class = df['Type'].value_counts()[row['Type']]

            X.append(row['Signal'])
            y.append(classes_to_check.index(row['Type']))
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


def files_in_dir(extension, directory):
    '''
    return the list of files in the directory with a specific extension

    Args:
            extension (str): file type to get

            directory (str): path of where files are present

    Returns:
            l (list): list of file names with extension (in current directory)
    '''
    l = []

    # get all file names in current directory
    file_names = os.listdir(directory)

    for file in file_names:
        if file.endswith(extension):
            l.append(file)

    return l


def remove_extension(file):
    '''
    remove extension of file passed in as a string

    Args:
            file (str): name of file with extension

    Returns:
            (str): name of file without extension
    '''
    return os.path.splitext(file)[0]


def get_write_dir(directory_name, subdirectory_name):
    '''
    get path of directory name specified where information needs
    to be written to (subdirectory specification is optional)

    Args:
            directory_name (str): name of directory to read from

            subdirectory (str): subdirectory of directory specified

    Returns:
            wr_dir (str): path of directory to write data to
    '''

    if subdirectory_name is None:
        wr_dir = os.getcwd() + '/../' + directory_name + '/'
    else:
        if subdirectory_name == '/':
            wr_dir = os.getcwd() + '/../' + directory_name + '/' + '_' + '/'
        else:
            wr_dir = os.getcwd() + '/../' + directory_name + \
                '/' + subdirectory_name + '/'
    # if dir does not exist make new one
    if not os.path.exists(wr_dir):
        os.makedirs(wr_dir, exist_ok=True)
        return wr_dir
    else:
        # return directory specified
        return wr_dir


def get_num_from_file(file_name):
    '''
    get maximum number from file name passed in

    Args;
            file_name (str): string to extract number from

    Returns:
            (str): max number from file name 
    '''
    numbers = re.findall('\d+', file_name)
    numbers = list(map(int, numbers))
    return str(numbers[0])


def get_all_subfolders(parent_dir):
    '''
    get all folders in the path specified

    Args;
            parent_dir (str): path where files are

    Returns:
            (list): list of strings with all folder names 
    '''
    return [dI for dI in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, dI))]