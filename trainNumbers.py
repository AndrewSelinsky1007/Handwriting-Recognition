import tensorflow as tf
from tensorflow.keras import layers, models

# Data Set
mnist = tf.keras.datasets.mnist

# X is the pictures, Y is the answer keys to them
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Restricting the pixels to be only pure white or black for reducing weight
x_train, x_test = x_train / 255.0, x_test / 255.0

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)), # Number images are 28x28
    layers.Dense(100, activation='relu'), # Node number, relu for ignoring empty space
    layers.Dropout(0.2),                  # Stops Memorizing somewhat 20% 
    layers.Dense(10, activation='softmax')# 10 possible outputs, softmax helps determine based on confidence %
])

# Adam = Adapts the brain based on model results. standard.
# sparse_categorical_crossentropy = Change based on results
# accuracy = % correct
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.save('digit_model.h5')
print("Finished training")