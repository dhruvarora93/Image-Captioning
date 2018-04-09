from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session


# extract features from each photo in the directory
def extract_features(directory):
	# load the model
	model = VGG16()
	# re-structure the model
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	# summarize
	# print(model.summary())
	# extract features from each photo
	features = dict()
	for name in listdir(directory):
		# load an image from file
		filename = directory + '/' + name
		image = load_img(filename, target_size=(224, 224))
		# convert the image pixels to a numpy array
		image = img_to_array(image)
		# reshape data for the model
		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		# prepare the image for the VGG model
		image = preprocess_input(image)
		# get features
		feature = model.predict(image, verbose=0)
		# get image id
		image_id = name.split('.')[0]
		# store feature
		features[image_id] = feature
	# print('>%s' % name)
	return features


# def get_session():
# 	num_threads = 0
# 	config = tf.ConfigProto()
# 	config.gpu_options.per_process_gpu_memory_fraction = 0.6
# 	config.gpu_options.allocator_type = "BFC"
# 	config.log_device_placement = False
# 	config.allow_soft_placement = True
#
# 	if num_threads > 1:
# 		return tf.Session(config=config, intra_op_parallelism_threads=num_threads)
# 	else:
# 		return tf.Session(config=config)


# extract features from all images
directory = 'Flicker8k_Dataset'

# sess = get_session()
# set_session(sess)

features = extract_features(directory)
print('Extracted Features: %d' % len(features))
# save to file
dump(features, open('features_vgg.pkl', 'wb'))
# sess.close()