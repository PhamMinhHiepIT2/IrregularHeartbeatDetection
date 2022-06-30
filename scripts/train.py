import os

from model import ClassificationModel, Resnet, VGG
from utils import getSignalDataFrame, trainAndTestSplit
from constants import (
    RESNET50_WEIGHT,
    RESNET50_METRICS,
    RESNET152_WEIGHT,
    RESNET152_METRICS,
    VGG16_METRICS,
    VGG16_WEIGHT,
    VGG19_METRICS,
    VGG19_WEIGHT,
    NUMBER_OF_CLASSES
)


# removing warning for tensorflow about AVX support
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def train_model(model: ClassificationModel, X_train, y_train, X_test, y_test, batch_size, epochs, callbacks, metrics_name, model_name):
    '''
        train model with given data

        Args:
            X_train (list): list of training signals

            X_test (list): list of testing signals

            y_train (list): list of training classes

            y_test (list): list of testing classes
    '''
    model.fit(X_train, y_train, batch_size=batch_size,
              epochs=epochs, shuffle=True, validation_data=(X_test, y_test),
              callbacks=callbacks)

    score = model.evaluate(X_test, y_test)
    model.print_test_metrics(score)
    model.save_metrics_and_weight(score, model, metrics_name, model_name)


def train_resnet50(X_train, y_train, X_test, y_test, batch_size, epochs, metrics_name=RESNET50_METRICS, model_name=RESNET50_WEIGHT):
    '''
        train resnet50 model with given data

        Args:
            X_train (list): list of training signals

            X_test (list): list of testing signals

            y_train (list): list of training classes

            y_test (list): list of testing classes
    '''
    resnet = Resnet(num_classes=NUMBER_OF_CLASSES)
    model = resnet.restnet50()
    model_path = resnet.get_model_path(model_name)
    graph_path = resnet.get_graph_path(model_name)
    checkpoints, early, tbCallBack = resnet.get_callbacks(
        model_path=model_path, graph_path=graph_path)
    train_model(model, X_train, y_train, X_test, y_test, batch_size,
                epochs, callbacks=[checkpoints, early, tbCallBack], metrics_name=metrics_name, model_name=model_name)


def train_resnet152(X_train, y_train, X_test, y_test, batch_size, epochs, metrics_name=RESNET152_METRICS, model_name=RESNET152_WEIGHT):
    '''
        train resnet152 model with given data

        Args:
            X_train (list): list of training signals

            X_test (list): list of testing signals

            y_train (list): list of training classes

            y_test (list): list of testing classes
    '''
    resnet = Resnet(num_classes=NUMBER_OF_CLASSES)
    model = resnet.restnet152()
    model_path = resnet.get_model_path(model_name)
    graph_path = resnet.get_graph_path(model_name)
    checkpoints, early, tbCallBack = resnet.get_callbacks(
        model_path=model_path, graph_path=graph_path)
    train_model(model, X_train, y_train, X_test, y_test, batch_size,
                epochs, callbacks=[checkpoints, early, tbCallBack], metrics_name=metrics_name, model_name=model_name)


def train_vgg16(X_train, y_train, X_test, y_test, batch_size, epochs, metrics_name=VGG16_METRICS, model_name=VGG16_WEIGHT):
    '''
        train vgg16 model with given data

        Args:
            X_train (list): list of training signals

            X_test (list): list of testing signals

            y_train (list): list of training classes

            y_test (list): list of testing classes
    '''
    vgg = VGG(num_classes=NUMBER_OF_CLASSES)
    model = vgg.vgg16()
    model_path = vgg.get_model_path(model_name)
    graph_path = vgg.get_graph_path(model_name)
    checkpoints, early, tbCallBack = vgg.get_callbacks(
        model_path=model_path, graph_path=graph_path)
    train_model(model, X_train, y_train, X_test, y_test, batch_size,
                epochs, callbacks=[checkpoints, early, tbCallBack], metrics_name=metrics_name, model_name=model_name)


def train_vgg19(X_train, y_train, X_test, y_test, batch_size, epochs, metrics_name=VGG19_METRICS, model_name=VGG19_WEIGHT):
    '''
        train vgg19 model with given data

        Args:
            X_train (list): list of training signals

            X_test (list): list of testing signals

            y_train (list): list of training classes

            y_test (list): list of testing classes
    '''
    vgg = VGG(num_classes=NUMBER_OF_CLASSES)
    model = vgg.vgg19()
    model_path = vgg.get_model_path(model_name)
    graph_path = vgg.get_graph_path(model_name)
    checkpoints, early, tbCallBack = vgg.get_callbacks(
        model_path=model_path, graph_path=graph_path)
    train_model(model, X_train, y_train, X_test, y_test, batch_size,
                epochs, callbacks=[checkpoints, early, tbCallBack], metrics_name=metrics_name, model_name=model_name)


if __name__ == '__main__':

    df = getSignalDataFrame()
    X_train, X_test, y_train, y_test = trainAndTestSplit(df, 0.2)

    train_resnet50(X_train, y_train, X_test, y_test, batch_size=64, epochs=20)
    train_resnet152(X_train, y_train, X_test, y_test, batch_size=64, epochs=20)
    train_vgg16(X_train, y_train, X_test, y_test, batch_size=64, epochs=20)
    train_vgg19(X_train, y_train, X_test, y_test, batch_size=64, epochs=20)
