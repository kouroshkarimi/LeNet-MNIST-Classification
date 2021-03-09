# IMPORT PACKAGES
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
import cv2
from tensorflow.keras import backend as K
import numpy as np
import tensorflow.keras.utils as util

# PREPARING DATA
print("[INFO] Downloading mnist dataset.")
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

if K.image_data_format() == "channels_first":
	train_images = train_images.reshape(train_images.shape[0], 1, 28, 28)
	test_images = test_images.reshape(test_images.shape[0], 1, 28, 28)

else:
	train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
	test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)


train_images = train_images /255.0
test_images = test_images / 255.0
train_labels = util.to_categorical(train_labels, 10)
test_labels = util.to_categorical(test_labels, 10)

# LOAD MODEL SAVED AFTER TRAIN
model = keras.models.load_model("LeMnist_model.h5")

# A LOOP TO SHOW IMAGES AND OUR MODEL PREDICTION FOR THAT IMAGE
for i in np.random.choice(np.arange(0, len(test_labels)), size=(10,)):
	# classify the digit
	probs = model.predict(test_images[np.newaxis, i])
	prediction = probs.argmax(axis=1)
	# extract the image from the test_images if using "channels_first"
	# ordering
	if K.image_data_format() == "channels_first":
		image = (test_images[i][0] * 255).astype("uint8")
	# otherwise we are using "channels_last" ordering
	else:
		image = (test_images[i] * 255).astype("uint8")
	# merge the channels into one image
	image = cv2.merge([image] * 3)
	# resize the image from a 28 x 28 image to a 96 x 96 image so we
	# can better see it
	image = cv2.resize(image, (96 * 2, 96 * 2), interpolation=cv2.INTER_LINEAR)
	# show the image and prediction
	cv2.putText(image, str(prediction[0]), (5, 20),
				cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
	print("[INFO] Predicted: {}, Actual: {}".format(prediction[0],
		np.argmax(test_labels[i])))
	cv2.imshow("Digit", image)
	cv2.waitKey(0)