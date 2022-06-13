# Implmentation of AlexNet model taken from
# https://www.mydatahack.com/building-alexnet-with-keras/
# script that reads data, creates model and trains it

import os
import resnet50

from constant import NUMBER_OF_CLASSES, IMAGE_HEIGHT, IMAGE_WIDTH
from utils import getSignalDataFrame, trainAndTestSplit
from model import ClassificationModel
# removing warning for tensorflow about AVX support
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':

    # (2) GET DATA
    df = getSignalDataFrame()

    X_train, X_test, y_train, y_test = trainAndTestSplit(df, 0.2)

    classification_model = ClassificationModel(
        X_train, y_train, X_test, y_test, batch_size=64, epochs=20, num_classes=NUMBER_OF_CLASSES)

    resnet50_model = resnet50.ResNet50(
        weights='imagenet', include_top=False, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))

    model = classification_model.resnet(resnet50_model, 'resnet50_model.h5')

    model.evaluate(X_test, y_test)
    