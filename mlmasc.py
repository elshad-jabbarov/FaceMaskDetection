from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import os

# axım dövrünü qeyd et EPOCHS
INIT_LR = 1e-4
EPOCHS = 20  # number of iterations
BS = 32  # batch siz number of samples processed before the model is updated

DIRECTORY = r"C:\Users\V-COMP\Desktop\facemask\dataset"

CATEGORIES = ["with_mask", "without_mask"]

# loading data from dataset
print("[INFO] sekiller yuklenir...")

data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(250, 250))
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(category)

lb = LabelBinarizer()  # binary classifier using one hot encoding for better result
labels = lb.fit_transform(labels)
#: Converts a class vector (integers) to binary class matrix.
labels = to_categorical(labels)
# to nummy array
data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.20, stratify=labels, random_state=42)

# generation of more data for training
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# MobileNetV2, FC layerlerle isleme
baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(250, 250, 3)))

headModel = baseModel.output
# Average pooling operation for spatial data. Inherits From: Layer, Module
# Average pooling operation for spatial data
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
# Flattens the input. Does not affect the batch size.
headModel = Flatten(name="flatten")(headModel)
# Dense implements the operation
headModel = Dense(128, activation="relu")(headModel)
# The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False

#  compile
print("[INFO] compiling model...")
# Adam optimization is a stochastic gradient descent
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

print("[INFO] training head...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)

print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# For testing we need index of image and calculate max probability
predIdxs = np.argmax(predIdxs, axis=1)

# Report
# The classification_report function builds a text report showing the main classification metrics
print(classification_report(testY.argmax(axis=1), predIdxs,
                            target_names=lb.classes_))

# Creating model
print("[INFO] maska askarlama modelini yadda saxla...")
model.save("maska_modeli.model", save_format="h5")

# Piloting
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
