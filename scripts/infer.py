import cv2
from tensorflow import keras

from constants import CLASSES_TO_CHECK


description = {
    "L": "Left bundle branch block beat",
    "N": "Normal beat",
    "V": "Ventricular premature beat",
    "A": "Atrial premature beat",
    "R": "Right bundle branch block beat"
}


def load_model(model_path: str):
    model = keras.models.load_model(model_path)
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy"])
    return model


def predict(model, image):
    img = cv2.imread(image)
    img = cv2.resize(img, (224, 224))
    img = img.reshape(1, 224, 224, 3)
    prob = model.predict(img)
    label_pred = prob.argmax(axis=-1)
    res = CLASSES_TO_CHECK[label_pred[0]]
    print("Probability: {}".format(prob))
    print("Result: {}".format(res))
    result_description = description[res]
    if res == "N":
        state = "Normal"
    else:
        state = "Abnormal"
    return_val = "{}.Result: {} with probability {}. {} means {} ".format(
        state,
        res,
        prob[0][label_pred[0]],
        res,
        result_description
    )
    return return_val
