from tensorflow import keras
from fastapi import FastAPI
import uvicorn


MODEL_PATH = "model/vgg16.h5"
app = FastAPI()


def load_model(model_path: str):
    model = keras.models.load_model(model_path)
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy"])
    return model


def predict(model, image):
    return model.predict(image)


@app.get("/")
def image_pred(image_path):
    model = load_model(MODEL_PATH)
    return predict(model, image_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
