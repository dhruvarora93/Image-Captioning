from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.models import load_model
import os
import numpy as np

# extract features from each photo in the directory
def extract_features(filename):
	# load the model
	model = InceptionV3(weights='imagenet')
	# re-structure the model
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	# load the photo
	image = load_img(filename, target_size=(299, 299))
	# convert the image pixels to a numpy array
	image = img_to_array(image)
	# reshape data for the model
	image = np.expand_dims(image, axis=0)
	# prepare the image for the VGG model
	image = preprocess_input(image)
	# get features
	feature = model.predict(image, verbose=0)
	return feature

# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
	# seed the generation process
	in_text = 'startseq'
	# iterate over the whole length of the sequence
	for i in range(max_length):
		# integer encode input sequence
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		# pad input
		sequence = pad_sequences([sequence], maxlen=max_length)
		# predict next word
		yhat = model.predict([photo,sequence], verbose=0)
		# convert probability to integer
		yhat = argmax(yhat)
		# map integer to word
		word = word_for_id(yhat, tokenizer)
		# stop if we cannot map the word
		if word is None:
			break
		# append as input for generating the next word
		in_text += ' ' + word
		# stop if we predict the end of the sequence
		if word == 'endseq':
			break
	return in_text

# load the tokenizer
tokenizer = load(open('tokenizer.pkl', 'rb'))
# pre-define the max sequence length (from training)
max_length = 34
# load the model
model = load_model('h5-models/model-ep009-loss2.777-val_loss3.473.h5')

testDir = os.path.join(os.getcwd(), 'testing')
f = open(os.path.join(testDir, 'test_result_generated_caption.txt'), 'w')
for img in os.listdir(testDir):
	try:
		# load and prepare the photograph
		photo = extract_features(os.path.join(testDir, img))
		# generate description
		description = generate_desc(model, tokenizer, photo, max_length)
		print(img, description)
		f.write(img + ' : ' + description.strip('startseq').strip('endseq') + '\n')
	except Exception as e:
		print(e)
f.close()