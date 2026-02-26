import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import tensorflow_datasets as tfds

# direct load dataset
ds_train, ds_test = tfds.load('emnist/letters', split=['train', 'test'], as_supervised=True)

# convert func for numpy array
def preprocess(ds):
    x, y = [], []
    for image, label in tfds.as_numpy(ds):
        x.append(image)
        y.append(label)
    return np.array(x), np.array(y)

x_train, y_train = preprocess(ds_train)
x_test, y_test = preprocess(ds_test)

# fixing the orientation, rotate 270. & flip
x_train = np.array([np.fliplr(np.rot90(img, k=3)) for img in x_train])
x_test = np.array([np.fliplr(np.rot90(img, k=3)) for img in x_test])

# 0-25, not 1-26
y_train = y_train - 1
y_test = y_test - 1
x_train, x_test = x_train / 255.0, x_test / 255.0

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28, 1)), #28x28
    layers.Dense(250, activation='relu'), #250 nodes
    layers.Dropout(0.2), #20% drop
    layers.Dense(120, activation='relu'), #SECOND LAYER 120
    layers.Dense(26, activation='softmax') #26 possible outcomes
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=7)
model.save('letter_model.h5')