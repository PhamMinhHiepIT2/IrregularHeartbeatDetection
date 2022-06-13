from tensorflow.keras import optimizers
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers import Dense, Activation, Dropout, Flatten,\
    Conv2D, MaxPooling2D

from tensorflow.keras.layers import BatchNormalization

from constant import NUMBER_OF_CLASSES, IMAGE_HEIGHT, IMAGE_WIDTH
from utils import printTestMetrics, saveMetricsAndWeights


class ClassificationModel:
    def __init__(self, train_data, train_label, test_data, test_label, batch_size, epochs, num_classes) -> None:
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_classes = num_classes
        self.model_output = None

        self.input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)

        self.check_point = ModelCheckpoint(self.model_output, monitor='val_acc', verbose=1,
                                           save_best_only=True, save_weights_only=False, mode='auto', period=1)

        self.early = EarlyStopping(monitor='val_acc', min_delta=0,
                                   patience=10, verbose=1, mode='auto')
        self.tbCallBack = TensorBoard(
            log_dir='../graph', histogram_freq=0, write_graph=True, write_images=True)

    def resnet(self, model, model_output):
        """
        Train and test ResNet model.

        Parameters:
        """

        self.model_output = model_output

        # Freeze the layers which you don't want to train. Here I am freezing the all layers.
        for layer in model.layers[:]:
            layer.trainable = False

        # Adding custom Layer
        # We only add
        x = model.output
        x = Flatten()(x)
        predictions = Dense(self.num_classes, activation="softmax")(x)

        # creating the final model
        model_final = Model(inputs=model.input, outputs=predictions)

        # compile the model
        model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(
            lr=0.0001, momentum=0.9), metrics=["accuracy"])

        model_final.fit(self.train_data, self.train_label, batch_size=self.batch_size,
                        epochs=self.epochs, shuffle=True, validation_data=(self.test_data, self.test_label),
                        callbacks=[self.check_point, self.early, self.tbCallBack])

        return model_final

    def alexnet(self, model=Sequential()):
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

        # (4) COMPILE MODEL
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        # (5) TRAIN
        model.fit(
            self.train_data,
            self.train_label,
            batch_size=64,
            epochs=150,
            verbose=1,
            validation_data=(self.test_data, self.test_label),
            shuffle=True,
            callbacks=[self.check_point, self.early, self.tbCallBack]
        )

    def evaluate(self, model):
        # (6) EVALUATE MODEL
        score = model.evaluate(self.test_data, self.test_label, verbose=0)

        printTestMetrics(score)

        saveMetricsAndWeights(score, model)
