from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet import ResNet152
from tensorflow.keras import optimizers
from keras.models import Model
from keras.layers import Flatten, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard


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

        self.input_shape = (224, 224, 3)

        self.check_point = ModelCheckpoint(self.model_output, monitor='val_acc', verbose=1,
                                           save_best_only=True, save_weights_only=False, mode='auto', period=1)

        self.early = EarlyStopping(monitor='val_acc', min_delta=0,
                                   patience=10, verbose=1, mode='auto')
        self.tbCallBack = TensorBoard(
            log_dir='../graph', histogram_freq=0, write_graph=True, write_images=True)

    def resnet50(self, model_output):
        """
        Train and test ResNet50 model.

        Parameters:
        """

        self.model_output = model_output

        model = ResNet50(include_top=False, weights='imagenet',
                         input_shape=self.input_shape)

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

        model_final.fit(self.train_data, self.train_labels, batch_size=self.batch_size,
                        epochs=self.epochs, shuffle=True, validation_data=(self.test_data, self.test_labels),
                        callbacks=[self.checkpoint, self.early, self.tbCallBack])

        return model_final
