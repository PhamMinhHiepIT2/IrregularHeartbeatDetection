from fastapi import FastAPI
import uvicorn

from constants import MODEL_PATH
from infer import load_model, predict

app = FastAPI()


@app.get("/")
def image_pred(image_path):
    model = load_model(MODEL_PATH)
    return predict(model, image_path)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
