from GradCAM import GradCAM

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121

from PIL import Image
from skimage.transform import resize

def load_data(IMAGE_SIZE = (224, 224), NUM_CLASSES = 14, ZEROS = True):
	testCSV = np.loadtxt("./../CheXpert-v1.0-small/valid.csv", delimiter=",", dtype=str)
	testPaths = testCSV[1:, 0]
	test_labels = testCSV[1:, 5:]

	x_test = np.zeros((testPaths.shape[0], IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

	for i in range(testPaths.shape[0]):
		image = Image.open("./../" + testPaths[i])
		image_array = np.asarray(image.convert("RGB"))
		image_array = image_array / 255.
		x_test[i] = resize(image_array, IMAGE_SIZE)

		test_labels[test_labels == '1.0'] = '1'
		test_labels[test_labels == ''] = '0'
		test_labels[test_labels == '0.0'] = '0'
		if ZEROS:
			test_labels[test_labels == '-1.0'] = '0'
		else:
			test_labels[test_labels == '-1.0'] = '1'
		y_test = np.asarray(test_labels, dtype = int)

	return testPaths, x_test, y_test

def transparent_cmap(cmap, N=255):
	"Copy colormap and set alpha values"

	mycmap = cmap
	mycmap._init()
	mycmap._lut[:,-1] = np.linspace(0, 0.4, N+4)
	return mycmap

def plot_GradCAM(heatmap, x1, path):
	w, h = heatmap.shape
	y, x = np.mgrid[0:h, 0:w]   
	mycmap = transparent_cmap(plt.cm.Reds)
	fig, ax = plt.subplots(1, 1)
	ax.imshow(x1, cmap='gray')
	cb = ax.contourf(x, y, heatmap, 2, cmap=mycmap)
	plt.colorbar(cb)
	plt.savefig(path)
	#plt.show()


if __name__ == "__main__":
	num_classes = 14
	inp = layers.Input(shape=(224, 224, 3))
	model = tf.keras.applications.DenseNet121(include_top=True, weights="./Grad_CAM/weightsBest.h5", input_tensor=inp, input_shape=(224, 224, 3), classes=num_classes)
	#model.summary()

	grad_cam = GradCAM(model)

	testPaths, x_test, y_test = load_data()

	print(x_test.shape)

	for i in range(x_test.shape[0]):
		x1 = x_test[[i]]
		predictions = model.predict(x1)
		pred = np.argmax(predictions)

		heatmap = grad_cam.get_heatmap(pred, x1)
		path = "Grad_CAM/Results/" + testPaths[i].replace("/", "_").replace(".jpg", "") + "_heatmap_GradCAM.png"
		plot_GradCAM(heatmap, x1[0], path)