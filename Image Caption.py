
## File used for generating Captions of images.

import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'gtk')

import pickle
from tqdm import tqdm
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.models import Model
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Merge, Activation, Flatten
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.layers import concatenate
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
import nltk


 

## Creating a dictionary containing all the captions of the images

def create_dictionary(captions):
    d = {}
    for i, row in enumerate(captions):
        row = row.split('\t')
        row[0] = row[0][:len(row[0])-2]
        if row[0] in d:
            d[row[0]].append(row[1])
        else:
            d[row[0]] = [row[1]]
    return d


def split_data(l):
    temp = []
    for i in img:
        if i[len(images):] in l:
            temp.append(i)
    return temp

## Feed these images to Inception V3 to get the encoded images by preprocessing them first.
def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def preprocess(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)
    return x

# encoding images
def encode(image):
    image = preprocess(image)
    temp_enc = model_new.predict(image)
    temp_enc = np.reshape(temp_enc, temp_enc.shape[1])
    return temp_enc


# ## Generator 
# 
## Use the encoding of an image and use a start word to predict the next word.
## After that, use the same image and use the previously predicted word to predict the next word.
## The image will be used at every iteration for the entire caption. 
## This is how the caption is generated for an image.
## Create a custom generator for that.

def data_generator(batch_size = 32):
        partial_caps = []
        next_words = []
        images = []
        
        df = pd.read_csv('flickr8k_training_dataset.txt', delimiter='\t')
        df = df.sample(frac=1)
        iter = df.iterrows()
        c = []
        imgs = []
        for i in range(df.shape[0]):
            x = next(iter)
            c.append(x[1][1])
            imgs.append(x[1][0])


        count = 0
        while True:
            for j, text in enumerate(c):
                current_image = encoding_train[imgs[j]]
                for i in range(len(text.split())-1):
                    count+=1
                    
                    partial = [word2idx[txt] for txt in text.split()[:i+1]]
                    partial_caps.append(partial)
                    
                    # Initializing with zeros to create a one-hot encoding matrix
                    # This is what must be predicted
                    # Initializing it with length=vocab_size
                    n = np.zeros(vocab_size)
                    # Setting the next word to 1 in the one-hot encoded matrix
                    n[word2idx[text.split()[i+1]]] = 1
                    next_words.append(n)
                    
                    images.append(current_image)

                    if count>=batch_size:
                        next_words = np.asarray(next_words)
                        images = np.asarray(images)
                        partial_caps = sequence.pad_sequences(partial_caps, maxlen=max_len, padding='post')
                        yield [[images, partial_caps], next_words]
                        partial_caps = []
                        next_words = []
                        images = []
                        count = 0



## Predict funtion


def predict_captions(image):
    start_word = ["<start>"]
    while True:
        par_caps = [word2idx[i] for i in start_word]
        par_caps = sequence.pad_sequences([par_caps], maxlen=max_len, padding='post')
        e = encoding_test[image[len(images):]]
        preds = final_model.predict([np.array([e]), np.array(par_caps)])
        word_pred = idx2word[np.argmax(preds[0])]
        start_word.append(word_pred)
        
        if word_pred == "<end>" or len(start_word) > max_len:
            break
            
    return ' '.join(start_word[1:-1])





def beam_search_predictions(image, beam_index = 3):
    start = [word2idx["<start>"]]
    
    start_word = [[start, 0.0]]
    
    while len(start_word[0][0]) < max_len:
        temp = []
        for s in start_word:
            par_caps = sequence.pad_sequences([s[0]], maxlen=max_len, padding='post')
            e = encoding_test[image[len(images):]]
            preds = final_model.predict([np.array([e]), np.array(par_caps)])
            
            word_preds = np.argsort(preds[0])[-beam_index:]
            
            # Getting the top <beam_index>(n) predictions and creating a 
            # new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
                    
        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]
    
    start_word = start_word[-1][0]
    intermediate_caption = [idx2word[i] for i in start_word]

    final_caption = []
    
    for i in intermediate_caption:
        if i != '<end>':
            final_caption.append(i)
        else:
            break
    
    final_caption = ' '.join(final_caption[1:])
    return final_caption


# Function to evaluate the normal Max search and beam searches
def evaluate_normal_and_beam_search_predictions(try_image):
    print ('Normal Max search:', predict_captions(try_image)) 
    print ('Beam Search, k=3:', beam_search_predictions(try_image, beam_index=3))
    print ('Beam Search, k=5:', beam_search_predictions(try_image, beam_index=5))
    print ('Beam Search, k=7:', beam_search_predictions(try_image, beam_index=7))
    Image.open(try_image)


##Main function
if __name__ == "__main__":

    print ("inside")
    ## Use the Flickr8K dataset
    token = 'Flickr8k_text/Flickr8k.token.txt'
    captions = open(token, 'r').read().strip().split('\n')

    d = create_dictionary(captions)
    
    images = 'Flicker8k_Dataset/'

    ## Contains all the images
    img = glob.glob(images+'*.jpg')

    #List of the training dataset
    train_images_file = 'Flickr8k_text/Flickr_8k.trainImages.txt'

    ##Read training images
    train_images = set(open(train_images_file, 'r').read().strip().split('\n'))

    # Getting the training images from all the images
    train_img = split_data(train_images)

    ## List of images used as the validation set
    val_images_file = 'Flickr8k_text/Flickr_8k.devImages.txt'
    val_images = set(open(val_images_file, 'r').read().strip().split('\n'))


    ## Getting the validation images from all the images
    val_img = split_data(val_images)

    ## List of images used as the validation set
    test_images_file = 'Flickr8k_text/Flickr_8k.testImages.txt'
    test_images = set(open(test_images_file, 'r').read().strip().split('\n'))

    # Getting the testing images from all the images
    test_img = split_data(test_images)
    len(test_img)

    Image.open(train_img[0])

    plt.imshow(np.squeeze(preprocess(train_img[0])))


    ## Create InceptionV3 keras model
    model = InceptionV3(weights='imagenet')
    new_input = model.input
    hidden_layer = model.layers[-2].output
    model_new = Model(new_input, hidden_layer)

    ## Define the predictor function
    ## tryi = model_new.predict(preprocess(train_img[0]))

    ## encode train images
    encoding_train = {}
    for img in tqdm(train_img):
        encoding_train[img[len(images):]] = encode(img)


    with open("encoded_images_inceptionV3.p", "wb") as encoded_pickle:
        pickle.dump(encoding_train, encoded_pickle) 


    encoding_train = pickle.load(open('encoded_images_inceptionV3.p', 'rb'))


    # encoding_train['3556792157_d09d42bef7.jpg'].shape

    ## encode test images
    encoding_test = {}
    for img in tqdm(test_img):
        encoding_test[img[len(images):]] = encode(img)





    with open("encoded_images_test_inceptionV3.p", "wb") as encoded_pickle:
        pickle.dump(encoding_test, encoded_pickle) 

    encoding_test = pickle.load(open('encoded_images_test_inceptionV3.p', 'rb'))


    # encoding_test[test_img[0][len(images):]].shape


    train_d = {}
    for i in train_img:
        if i[len(images):] in d:
            train_d[i] = d[i[len(images):]]

    val_d = {}
    for i in val_img:
        if i[len(images):] in d:
            val_d[i] = d[i[len(images):]]

    test_d = {}
    for i in test_img:
        if i[len(images):] in d:
            test_d[i] = d[i[len(images):]]


    # Calculating the unique words in the vocabulary.

    caps = []
    for key, val in train_d.items():
        for i in val:
            caps.append('<start> ' + i + ' <end>')

    words = [i.split() for i in caps]

    unique = []
    for i in words:
        unique.extend(i)
    unique = list(set(unique))
    unique = pickle.load(open('unique.p', 'rb'))


    # Mapping the unique words to indices and vice-versa

    word2idx = {val:index for index, val in enumerate(unique)}
    word2idx['<start>']
    idx2word = {index:val for index, val in enumerate(unique)}

    # Calculating the maximum length among all the captions

    max_len = 0
    for c in caps:
        c = c.split()
        if len(c) > max_len:
            max_len = len(c)
    max_len

    vocab_size = len(unique)

    # Adding <start> and <end> to all the captions to indicate the starting and ending of a sentence. This will be used while we predict the caption of an image
    f = open('flickr8k_training_dataset.txt', 'w')
    f.write("image_id\tcaptions\n")

    for key, val in train_d.items():
        for i in val:
            f.write(key[len(images):] + "\t" + "<start> " + i +" <end>" + "\n")

    f.close()

    df = pd.read_csv('flickr8k_training_dataset.txt', delimiter='\t')
    c = [i for i in df['captions']]
    imgs = [i for i in df['image_id']]

    a = c[-1]
    a, imgs[-1]

    for i in a.split():
        print (i, "=>", word2idx[i])

    ## Samples per epoch
    samples_per_epoch = 0
    for ca in caps:
        samples_per_epoch += len(ca.split())-1

    # ## Create the model

    embedding_size = 300


    # Input dimension is 4096 as the encoded version of the image is input.

    image_model = Sequential([
            Dense(embedding_size, input_shape=(2048,), activation='relu'),
            RepeatVector(max_len)
        ])


    # Since the next word is predicted using the previous words(length of previous word changes with every iteration over the caption), set return_sequences = True.

    caption_model = Sequential([
            Embedding(vocab_size, embedding_size, input_length=max_len),
            LSTM(256, return_sequences=True),
            TimeDistributed(Dense(300))
        ])


    # Merging the models and creating a softmax classifier


    final_model = Sequential([
            Merge([image_model, caption_model], mode='concat', concat_axis=1),
            Bidirectional(LSTM(256, return_sequences=False)),
            Dense(vocab_size),
            Activation('softmax')
        ])

    final_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    final_model.summary()



    ## Define steps_per_epoch, used by keras 2 instead of samples_per_epoch
    steps_per_epoch=samples_per_epoch/128

    final_model.fit_generator(data_generator(batch_size=128), steps_per_epoch=steps_per_epoch, epochs=1, 
                              verbose=2)

    final_model.fit_generator(data_generator(batch_size=128), steps_per_epoch=steps_per_epoch, epochs=1, 
                              verbose=2)

    final_model.fit_generator(data_generator(batch_size=128), steps_per_epoch=steps_per_epoch, epochs=1, 
                              verbose=2)

    final_model.fit_generator(data_generator(batch_size=128), steps_per_epoch=steps_per_epoch, epochs=1, 
                              verbose=2)

    final_model.fit_generator(data_generator(batch_size=128), steps_per_epoch=steps_per_epoch, epochs=1, 
                              verbose=2)

    final_model.fit_generator(data_generator(batch_size=128), steps_per_epoch=steps_per_epoch, epochs=1, 
                              verbose=2)

    final_model.optimizer.lr = 1e-4

    final_model.fit_generator(data_generator(batch_size=128), steps_per_epoch=steps_per_epoch, epochs=1, 
                              verbose=2)

    final_model.fit_generator(data_generator(batch_size=128), steps_per_epoch=steps_per_epoch, epochs=1, 
                              verbose=2)



    final_model.save_weights('time_inceptionV3_7_loss_3.2604.h5')
    final_model.load_weights('time_inceptionV3_7_loss_3.2604.h5')




    final_model.fit_generator(data_generator(batch_size=128), steps_per_epoch=steps_per_epoch, epochs=1, 
                              verbose=2)

    final_model.fit_generator(data_generator(batch_size=128), steps_per_epoch=steps_per_epoch, epochs=1, 
                              verbose=2)

    final_model.save_weights('time_inceptionV3_3.21_loss.h5')



    final_model.fit_generator(data_generator(batch_size=128), steps_per_epoch=steps_per_epoch, epochs=1, 
                              verbose=2)

    final_model.fit_generator(data_generator(batch_size=128), steps_per_epoch=steps_per_epoch, epochs=1, 
                              verbose=2)


    final_model.fit_generator(data_generator(batch_size=128), steps_per_epoch=steps_per_epoch, epochs=1, 
                              verbose=2)

    final_model.save_weights('time_inceptionV3_3.15_loss.h5')



    final_model.fit_generator(data_generator(batch_size=128), steps_per_epoch=steps_per_epoch, epochs=1, 
                              verbose=2)

    final_model.load_weights('time_inceptionV3_1.5987_loss.h5')


    test_img_random_number= int(np.random.randint(0, 1000, size=1))

    try_image = test_img[0]
    Image.open(try_image)

    try_images=[test_img[0],test_img[7],test_img[851],'Flickr8k_Dataset/Flicker8k_Dataset/136552115_6dc3e7231c.jpg',\
                                                 'Flickr8k_Dataset/Flicker8k_Dataset/1674612291_7154c5ab61.jpg', 'Flickr8k_Dataset/Flicker8k_Dataset/384577800_fc325af410.jpg',\
                'Flickr8k_Dataset/Flicker8k_Dataset/3631986552_944ea208fc.jpg', 'Flickr8k_Dataset/Flicker8k_Dataset/3320032226_63390d74a6.jpg',\
                'Flickr8k_Dataset/Flicker8k_Dataset/3316725440_9ccd9b5417.jpg', 'Flickr8k_Dataset/Flicker8k_Dataset/2306674172_dc07c7f847.jpg',\
                'Flickr8k_Dataset/Flicker8k_Dataset/2542662402_d781dd7f7c.jpg',test_img[test_img_random_number]]

    for i in range(len(try_images)):
        evaluate_normal_and_beam_search_predictions(try_images[i])
