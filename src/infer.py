from tensorflow import keras


model = keras.models.load_model("model.h5")

print(model)
print(dir(model))
model.output
