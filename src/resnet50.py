from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet import ResNet152
from tensorflow.keras import optimizers
from keras.models import Model
from keras.layers import Flatten, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

img_width, img_height = 224, 224


def resnet50(train_data, train_labels, test_data, test_labels, batch_size, epochs,
             num_classes, output_file):
    """
    Train and test ResNet50 model.

    Parameters:
    """

    model = ResNet50(include_top=False, weights='imagenet',
                     input_shape=(img_width, img_height, 3))

    # Freeze the layers which you don't want to train. Here I am freezing the all layers.
    for layer in model.layers[:]:
        layer.trainable = False

    # Adding custom Layer
    # We only add
    x = model.output
    x = Flatten()(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    # creating the final model
    model_final = Model(inputs=model.input, outputs=predictions)

    # compile the model
    model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(
        lr=0.0001, momentum=0.9), metrics=["accuracy"])

    # Save the model according to the conditions
    checkpoint = ModelCheckpoint(output_file, monitor='val_acc', verbose=1,
                                 save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_acc', min_delta=0,
                          patience=10, verbose=1, mode='auto')
    tbCallBack = TensorBoard(
        log_dir='../graph', histogram_freq=0, write_graph=True, write_images=True)

    model_final.fit(train_data, train_labels, batch_size=batch_size,
                    epochs=epochs, shuffle=True, validation_data=(test_data, test_labels),
                    callbacks=[checkpoint, early, tbCallBack])

    return model_final


def resnet152(train_data, train_labels, test_data, test_labels, batch_size, epochs,
              num_classes, output_file):
    """
    Train and test ResNet50 model.

    Parameters:
    """

    model = ResNet152(include_top=False, weights='imagenet',
                      input_shape=(img_width, img_height, 3))

    # Freeze the layers which you don't want to train. Here I am freezing the all layers.
    for layer in model.layers[:]:
        layer.trainable = False

    x = model.output
    x = Flatten()(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    # creating the final model
    model_final = Model(inputs=model.input, outputs=predictions)

    # compile the model
    model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(
        lr=0.0001, momentum=0.9), metrics=["accuracy"])

    # Save the model according to the conditions
    checkpoint = ModelCheckpoint(output_file, monitor='val_acc', verbose=1,
                                 save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_acc', min_delta=0,
                          patience=5, verbose=1, mode='auto')

    tbCallBack = TensorBoard(
        log_dir='../graph', histogram_freq=0, write_graph=True, write_images=True)

    model_final.fit(train_data, train_labels, batch_size=batch_size,
                    epochs=epochs, shuffle=True, validation_data=(test_data, test_labels),
                    callbacks=[checkpoint, early, tbCallBack])

    return model_final
