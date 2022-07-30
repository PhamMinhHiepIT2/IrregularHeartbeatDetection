from tensorflow import keras
from fastapi import FastAPI
import uvicorn
import cv2

MODEL_PATH = "model/vgg16.h5"
app = FastAPI()

CLASSES_TO_CHECK = ['L', 'N', 'V', 'A', 'R']
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


@app.get("/")
def image_pred(image_path):
    model = load_model(MODEL_PATH)
    return predict(model, image_path)

def test_pred():
    import os
    image_folder = "/home/hieppm/hieppm/IrregularHeartbeatDetection/beat_write_dir/N"
    model_path = "/home/hieppm/hieppm/IrregularHeartbeatDetection/model/my_model.h5"
    model = load_model(model_path)
    for image in os.listdir(image_folder):
        try:
            img_path = os.path.join(image_folder, image)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))
            img = img.reshape(1, 224, 224, 3)
            prob = model.predict(img)
            label_pred = prob.argmax(axis=-1)
            res = CLASSES_TO_CHECK[label_pred[0]]
            # print("Probability: {}".format(prob))
            # print("Result: {}".format(res))
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
            print(return_val)
        except Exception as e:
            print(e)



if __name__ == "__main__":
    # uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
    test_pred()