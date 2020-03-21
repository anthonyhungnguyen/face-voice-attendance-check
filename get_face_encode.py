from imutils import paths
import cv2
import os
import numpy as np
import pickle

# Define paths
base_dir = os.path.dirname(__file__)
prototxt_path = os.path.join(base_dir + 'model/deploy.prototxt')
caffeemodel_path = os.path.join(
    base_dir + 'model/res10_300x300_ssd_iter_140000.caffemodel')
openface_path = os.path.join(base_dir + 'model/openface.nn4.small2.v1.t7')

# Define detector

embeddings = {}

# Get labels
for name in os.listdir('data'):
    embeddings[name] = []

# Read the model
model = cv2.dnn.readNetFromCaffe(prototxt_path, caffeemodel_path)
embedded = cv2.dnn.readNetFromTorch(openface_path)

# join all image paths
imagePaths = list(paths.list_images(base_dir + 'data'))
num_of_images = len(imagePaths)

for (i, data) in enumerate(imagePaths):

    print(f'Processing image {i}/{num_of_images}')

    label = data.split('\\')[-2]

    img = cv2.imread(data)

    faceBlob = cv2.dnn.blobFromImage(img,
                                     1.0 / 255, (96, 96), (0, 0, 0),
                                     swapRB=True,
                                     crop=False)
    embedded.setInput(faceBlob)
    vec = embedded.forward()

    embeddings[label].append(vec)

embeddings['code'] = list(embeddings.keys())

with open('./output/encoded_faces_with_labels.pickle', 'wb') as handle:
    pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)