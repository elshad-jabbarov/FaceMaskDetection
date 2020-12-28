from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os


def maskadetect(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    # Opencv’s  blobFromImage function
    blob = cv2.dnn.blobFromImage(frame, 1.0, (250, 250),
                                 (104.0, 177.0, 123.0))
    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    askarlama = faceNet.forward()
    print(askarlama.shape)
    faces = []
    lock = []
    preds = []

    for i in range(0, askarlama.shape[2]):
        confidence = askarlama[0, 0, i, 2]
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = askarlama[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # For defining ROI we need to convert BGR to RGB
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (250, 250))
            # Converts a PIL Image instance to a Numpy array.
            face = img_to_array(face)
            # Preprocesses a tensor or Numpy array encoding a batch of images.
            face = preprocess_input(face)

            faces.append(face)
            lock.append((startX, startY, endX, endY))

    # at least one face
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
    return (lock, preds)


# Load the model
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# For detecting mask load the model
maskNet = load_model("maska_modeli.model")

# Starting Livestreaming
print("[INFO] video kamera aktiv edilir...")
vs = VgitideoStream(src=0).start()

# For the capturing every frame on live we need while loop
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=900)

    # mask detection
    (lock, preds) = maskadetect(frame, faceNet, maskNet)

    for (box, pred) in zip(lock, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # box and detection on the interface happens here
        label = "Maska" if mask > withoutMask else "Maska yoxdur"
        color = (0, 255, 0) if label == "Maska" else (0, 0, 255)

        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # show the output frame
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

    # if 'q' pressed stop
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
