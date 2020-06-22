# Tensorflow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
from joblib import dump
from pathlib import Path
import matplotlib.pyplot as plt

print(tf.__version__)

# Import the dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print(train_images.shape, test_images.shape)

# Save the data
with open(Path(Path.cwd() / "train.pkl"), 'wb') as f:
    dump({"img": train_images, "label": train_labels}, f, compress=True)

with open(Path(Path.cwd() / "test.pkl"), 'wb') as f:
    dump({"img": test_images, "label": test_labels}, f, compress=True)

# Class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# The pixels range from 0 to 255. We will scale them to a range of 0 to 1 (makes the NN faster)
train_images = train_images / 255
test_images = test_images / 255

# Setup the layers for the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # Flatten transforms images from 28 by 28 to 784 pixels one-dim array
    keras.layers.Dense(128, activation='relu'),  # Neural layer with 128 neurons (nodes)
    keras.layers.Dense(10)  # Returns a logit array of len 10 in which each node contains score of one of the 10 clasess
])

# Compile the model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train/fit the model
model.fit(
    x=train_images,
    y=train_labels,
    epochs=10
)
model.to_json()
with open(Path(Path.cwd() / "model.json"), 'w') as f:
    f.write(model.to_json())

model.save_weights(str(Path(Path.cwd() / "model.h5")))

# Evaluate the model
test_loss, test_acc = model.evaluate(x=test_images, y=test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')

