import cv2
import os
import numpy as np
import argparse

# Construct argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-n',
                '--name',
                required=True,
                help='name of person to train on')
ap.add_argument('-c', '--count', required=True, help='number of images')
args = vars(ap.parse_args())

# Define model
base_dir = os.path.dirname(__file__)
prototxt_path = os.path.join(base_dir + 'model/deploy.prototxt')
caffeemodel_path = os.path.join(
    base_dir + 'model/res10_300x300_ssd_iter_140000.caffemodel')

# Read the model
model = cv2.dnn.readNetFromCaffe(prototxt_path, caffeemodel_path)

# Setup camera
cam = cv2.VideoCapture(0)
cam.set(3, 1920)
cam.set(4, 1080)

# Model setup
j = 0
running = True
name = args['name']
count = int(args['count'])
save_dir = base_dir + 'data/' + name

# Check path exits - if not, create one
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Detect face and take pictures
while running:
    ret, frame = cam.read()

    resized = cv2.resize(frame, (300, 300))

    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(resized,
                                 1.0, (300, 300), (104.0, 177.0, 123.0),
                                 swapRB=False,
                                 crop=False)

    model.setInput(blob)

    detections = model.forward()

    for i in range(0, detections.shape[2]):
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

        (startX, startY, endX, endY) = box.astype('int')

        confidence = detections[0, 0, i, 2]

        if confidence > 0.95:
            cv2.imwrite(f'{save_dir}/{j}.jpg', frame[startY:endY, startX:endX])
            j += 1

            print(f'Taking image: {j}/100')
            if j > count:
                print('Finished')
                running = False
    cv2.imshow('Face', frame)

    if cv2.waitKey(1) >= 0:
        break

cv2.destroyAllWindows()