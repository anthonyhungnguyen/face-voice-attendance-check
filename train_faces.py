import pickle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
import tensorflow as tf

with open('./output/encoded_faces_with_labels.pickle', 'rb') as handle:
    data = pickle.load(handle)


def transformToXy(data):
    X, y = [], []
    for name in data['code']:
        X.extend([i for i in data[name]])
        y.extend([name] * len(data[name]))

    return X, y


def encodeXy(X, y):
    X = np.array(X)
    y = np.array(y)

    labelEncoder, oneHotEncoder = LabelEncoder(), OneHotEncoder()

    y = labelEncoder.fit_transform(y).reshape(-1, 1)

    y = oneHotEncoder.fit_transform(y).toarray().reshape(
        -1, 1, len(data['code']))

    return X, y


X, y = transformToXy(data)
X, y = encodeXy(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential([
    Dense(128, input_shape=(1, 128), activation='relu'),
    Dense(256, activation='relu'),
    Dense(len(data['code']), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, validation_split=0.2)

print(model.evaluate(X_test, y_test))

model.save('./output/model.h5')
