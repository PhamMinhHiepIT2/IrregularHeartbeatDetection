from tensorflow import keras
from fastapi import FastAPI
import uvicorn
import cv2

MODEL_PATH = "model/vgg16.h5"
app = FastAPI()

CLASSES_TO_CHECK = ['L', 'N', 'V', 'A', 'R']


def load_model(model_path: str):
    model = keras.models.load_model(model_path)
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy"])
    return model


def predict(model, image):
    img = cv2.imread(image)
    img = cv2.resize(img, (224, 224))
    img = img.reshape(1,224,224,3)
    prob = model.predict(img)
    label_pred = prob.argmax(axis=-1)
    res = CLASSES_TO_CHECK[label_pred[0]]
    print("Probability: {}".format(prob))
    print("Result: {}".format(res))
    return_val = "Result: {} with probability {}".format(res, prob[0][label_pred[0]])
    return return_val


@app.get("/")
def image_pred(image_path):
    model = load_model(MODEL_PATH)
    return predict(model, image_path)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
