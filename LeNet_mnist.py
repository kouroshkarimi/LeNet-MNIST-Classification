# IMPORT PACKAGES
import tensorflow as tf
from LeNet import LeNet
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import SGD
import numpy as np
from tensorflow.keras import backend as K
import tensorflow.keras.utils as util



print("[INFO] Downloading mnist dataset.")
# DOWNLOAD MNIST DATA
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

if K.image_data_format() == "channels_first":
	train_images = train_images.reshape(train_images.shape[0], 1, 28, 28)
	test_images = test_images.reshape(test_images.shape[0], 1, 28, 28)

else:
	train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
	test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

# NORMALIZING INPUT DATA
train_images = train_images /255.0
test_images = test_images / 255.0

# ONE HOT ENCODING OUR OUTPUT
train_labels = util.to_categorical(train_labels, 10)
test_labels = util.to_categorical(test_labels, 10)

print("[INFO] Compiling...")

w_path = None
#w_path = "LeMnist_model_wights.h5" # IF WE WANT OUR SAVED WEIGHTS

# COMPILING OUR MODEL WITH SGD AND CATEGORICAL (ONE HOT) LOSS FUNCTION
optimizer = SGD(lr = 1e-3, momentum = 0.9)
ls = 'categorical_crossentropy'
model = LeNet.build(n_channels = 1, imgRows = 28, imgCols = 28, n_class = 10, w_path = w_path)
model.compile(loss = ls, optimizer = optimizer, metrics = 'accuracy')

if w_path is None:
	print('[INFO] Training images (weight not loaded)...')
	model.fit(train_images, train_labels, batch_size = 128, epochs = 20, verbose = 1)
	print("[INFO] Training Complete.")

# EVALUATING OUR MODEL WITH TEST SET
print("[INFO] Evaluating the model...")
loss, acc = model.evaluate(test_images, test_labels, batch_size = 128, verbose = 1)
print("[INFO] validation accuracy: {:.2f}%. ".format(acc * 100))
print("[INFO] validation loss: {:.2f}. ".format(loss))

# SAVE OUR MODEL AND WEIGHTS (TO TRAIN ONCE AND USE IT)
model.save("LeMnist_model.h5", overwrite = True)
model.save_weights("LeMnist_model_wights.h5", overwrite=True)