# Tensorflow and tf.keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras import Sequential
from tensorflow import keras

# Helper libraries
from joblib import dump
from pathlib import Path
from datetime import datetime

# Import the dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print(train_images.shape, test_images.shape)

# Save the data
with open(Path(Path.cwd() / "data/train.pkl"), 'wb') as f:
    dump({"img": train_images, "label": train_labels}, f, compress=True)

with open(Path(Path.cwd() / "data/test.pkl"), 'wb') as f:
    dump({"img": test_images, "label": test_labels}, f, compress=True)

# Class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# The pixels range from 0 to 255. We will scale them to a range of 0 to 1 (makes the NN faster)
train_images = train_images / 255
test_images = test_images / 255

# Setup the layers for the model
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten transforms images from 28 by 28 to 784 pixels one-dim array
    Dense(128, activation='relu'),  # Neural layer with 128 neurons (nodes)
    Dense(10)  # Returns a logit array of len 10 in which each node contains score of one of the 10 clasess
])

# Compile the model
model.compile(
    optimizer='adam',
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# We would like to log our results and view with Tensorboard
log_dir = Path(Path.cwd() / "logs" / datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train/fit the model
model.fit(
    x=train_images,
    y=train_labels,
    epochs=10,
    validation_split=0.3,
    callbacks=[tensorboard_callback]
)

# Save the model
model.save(str(Path(Path.cwd() / "model/model.h5")))

# Evaluate the model
test_loss, test_acc = model.evaluate(x=test_images, y=test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')

