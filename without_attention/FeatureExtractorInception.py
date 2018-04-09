from os import listdir
from pickle import dump
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
import numpy as np


# extract features from each photo in the directory
def extract_features(directory):
	# load the model
	model = InceptionV3(weights='imagenet')
	# re-structure the model
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	# summarize
	#print(model.summary())
	# extract features from each photo
	features = dict()
	for name in listdir(directory):
		# load an image from file
		filename = directory + '/' + name
		image = load_img(filename, target_size=(299, 299))
		# convert the image pixels to a numpy array
		image = img_to_array(image)
		# reshape data for the model
		image = np.expand_dims(image, axis=0)
		# prepare the image for the Inception model
		image = preprocess_input(image)
		# get features
		feature = model.predict(image, verbose=0)
		# get image id
		image_id = name.split('.')[0]
		# store feature
		features[image_id] = feature
		#print('>%s' % name)
	return features

# extract features from all images
directory = 'Flicker8k_Dataset'
features = extract_features(directory)
print('Extracted Features: %d' % len(features))
# save to file
dump(features, open('features_inception_v3.pkl', 'wb'))
