import os
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet import ResNet152
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras import optimizers
from keras.models import Model
from keras.layers import Flatten, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from constants import WEIGHT_FOLDER, METRICS_FOLDER


class ClassificationModel:
    def __init__(self, num_classes, weight='imagenet', image_weight=224, image_height=224, output_folder="../models", graph_folder="../graphs"):
        self.output_folder = output_folder
        self.graph_folder = graph_folder
        self.weight = weight
        self.image_weight = image_weight
        self.image_height = image_height
        self.num_classes = num_classes

    def get_model_path(self, model_name):
        return self.output_folder + "/" + model_name

    def get_graph_path(graph_folder, model_name):
        folder_path = graph_folder + "/" + os.path.splitext(model_name)[0]
        os.makedirs(folder_path, exist_ok=True)
        return folder_path

    def get_weight_path(self, model_name):
        weight_folder = os.path.join(self.output_folder, model_name, WEIGHT_FOLDER)
        os.makedirs(weight_folder, exist_ok=True)
        return weight_folder

    def get_metrics_folder(self, model_name):
        metrics_folder = os.path.join(self.output_folder, model_name, METRICS_FOLDER)
        os.makedirs(metrics_folder, exist_ok=True)
        return metrics_folder

    def evaluate(self, model, X_test, y_test):
        score = model.evaluate(X_test, y_test, verbose=0)
        return score


    def get_callbacks(self, model_path, graph_path):
        checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=1,
                                     save_best_only=True, save_weights_only=False, mode='auto', period=1)
        early = EarlyStopping(monitor='val_acc', min_delta=0,
                              patience=10, verbose=1, mode='auto')
        tbCallBack = TensorBoard(
            log_dir=graph_path, histogram_freq=0, write_graph=True, write_images=True)
        return [checkpoint, early, tbCallBack]

    def build_model(self, model):
        # Freeze the layers which you don't want to train. Here I am freezing the all layers.
        for layer in model.layers[:]:
            layer.trainable = False

        x = model.output
        x = Flatten()(x)
        predictions = Dense(self.num_classes, activation="softmax")(x)

        # creating the final model
        model_final = Model(inputs=model.input, outputs=predictions)

        # compile the model
        model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(
            lr=0.0001, momentum=0.9), metrics=["accuracy"])

        return model_final

    def print_test_metrics(self, score):
        '''
        print prediction score

        Args:
            score (list): list with test loss and test accuracy
        '''
        print()
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    
    def save_metrics_and_weight(self, score, model, metric_name, model_name):
        '''
        Save metric and weight data in specific folders

        Args:
            score (list): list containing loss value and accuracy of current model

            model (keras model): compiled model which has been trained with training data
            Returns:
                    (dataframe): dataframe contatining image information 
            '''
        current_acc = score[1]
        weight_folder = self.get_weight_path(model_name)
        metrics_folder = self.get_metrics_folder(model_name)
        metrics_path = os.path.join(metrics_folder, metric_name)
        weight_path = os.path.join(weight_folder, model_name)
        if (len(os.listdir(metrics_path)) == 0):
            # create text file with placeholder accuracy value (i.e 0)
            np.save(metrics_path + metric_name, [0])
            model.save(weight_path)
            del model
        else:
            highest_acc = np.load(metrics_path)[0]
            if (current_acc > highest_acc):
                np.save(metrics_path + metric_name, [current_acc])
                model.save(weight_path)
                del model
                print('\nAccuracy Increase: ' +
                    str((current_acc - highest_acc)*100) + '%')


class Resnet(ClassificationModel):
    def __init__(self, num_classes, weight='imagenet', image_weight=224, image_height=224, output_folder="../models", graph_folder="../graphs") -> None:
        super().__init__(num_classes, weight, image_weight,
                         image_height, output_folder, graph_folder)

    def build_model(self, model):
        # Freeze the layers which you don't want to train. Here I am freezing the all layers.
        for layer in model.layers[:]:
            layer.trainable = False

        x = model.output
        x = Flatten()(x)
        predictions = Dense(self.num_classes, activation="softmax")(x)

        # creating the final model
        model_final = Model(inputs=model.input, outputs=predictions)

        # compile the model
        model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(
            lr=0.0001, momentum=0.9), metrics=["accuracy"])

        return model_final

    def restnet50(self):
        model_init = ResNet50(include_top=False, weights=self.weight,
                              input_shape=(self.image_weight, self.image_height, 3))

        model = self.build_model(model_init)
        return model

    def restnet152(self):
        model_init = ResNet152(include_top=False, weights=self.weight,
                               input_shape=(self.image_weight, self.image_height, 3))

        model = self.build_model(model_init)
        return model


class VGG(ClassificationModel):
    def __init__(self, num_classes, weight='imagenet', image_weight=224, image_height=224, output_folder="../models", graph_folder="../graphs") -> None:
        super().__init__(num_classes, weight, image_weight,
                         image_height, output_folder, graph_folder)

    def vgg16(self):
        model_init = VGG16(include_top=False, weights=self.weight,
                           input_shape=(self.image_weight, self.image_height, 3))

        model = self.build_model(model_init)
        return model
        

    def vgg19(self):
        model_init = VGG19(include_top=False, weights=self.weight,
                           input_shape=(self.image_weight, self.image_height, 3))

        model = self.build_model(model_init)
        return model
        
        