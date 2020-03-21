from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2
import os
import numpy as np
import dlib
import pickle

print(tf.config.list_physical_devices('GPU'))

# Define paths
base_dir = os.path.dirname(__file__)
prototxt_path = os.path.join(base_dir + 'model/deploy.prototxt')
caffeemodel_path = os.path.join(
    base_dir + 'model/res10_300x300_ssd_iter_140000.caffemodel')
openface_path = os.path.join(base_dir + 'model/openface.nn4.small2.v1.t7')

# Read trained model
tf_model = load_model('./output/model.h5')
with open('./output/encoded_faces_with_labels.pickle', 'rb') as handle:
    encoded = pickle.load(handle)

# Read the model
model = cv2.dnn.readNetFromCaffe(prototxt_path, caffeemodel_path)
embedded = cv2.dnn.readNetFromTorch(openface_path)

# Detect face
cam = cv2.VideoCapture(0)
cam.set(3, 300)
cam.set(4, 300)

# Detect
while True:
    ret, frame = cam.read()

    temp = cv2.resize(frame, (300, 300))

    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(temp,
                                 1.0, (300, 300), (104.0, 177.0, 123.0),
                                 swapRB=False,
                                 crop=False)

    model.setInput(blob)

    detections = model.forward()

    for i in range(0, detections.shape[2]):
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype('int')

        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:

            roi = frame[startY:endY, startX:endX]
            try:
                faceBlob = cv2.dnn.blobFromImage(roi,
                                                 1.0 / 255, (96, 96),
                                                 (0, 0, 0),
                                                 swapRB=True,
                                                 crop=False)
            except Exception as e:
                print(str(e))

            embedded.setInput(faceBlob)
            vec = embedded.forward().reshape(-1, 1, 128)
            results = tf_model.predict(vec).flatten()
            best_index = results.argmax()
            name = encoded['code'][best_index]
            proba = results[best_index]
            cv2.putText(frame, name + f': {proba:.2f}%', (startX, startY),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) >= 0:
        break

cv2.destroyAllWindows()