import pandas as pd
import numpy as np
import os
import cPickle
from cnn_util import *

vgg_model = '/home/dalek/git-repos/show_attend_and_tell.tensorflow/vgg/VGG_ILSVRC_19_layers.caffemodel'
vgg_deploy = '/home/dalek/git-repos/show_attend_and_tell.tensorflow/vgg/VGG_ILSVRC_19_layers_deploy.prototxt'

annotation_path = '/home/dalek/git-repos/show_attend_and_tell.tensorflow/Flickr8k_text/Flickr8k.token.txt'
flickr_image_path = '/home/dalek/git-repos/show_attend_and_tell.tensorflow/Flickr8k_images/'
feat_path = '/media/dalek/Modus Operandi/Users/amit1/linux-files/data/feats.npy'
annotation_result_path = '/media/dalek/Modus Operandi/Users/amit1/linux-files/data/annotations.pickle'

cnn = CNN(model=vgg_model, deploy=vgg_deploy, width=224, height=224)

annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])
annotations['image_num'] = annotations['image'].map(lambda x: x.split('#')[1])
annotations['image'] = annotations['image'].map(lambda x: os.path.join(flickr_image_path,x.split('#')[0]))


unique_images = annotations['image'].unique()
final_images = []
for images in unique_images:
	if os.path.exists(images):
		final_images.append(images)
image_df = pd.DataFrame({'image':final_images, 'image_id':range(len(final_images))})
print(len(unique_images)-len(final_images))
annotations = pd.merge(annotations, image_df)
annotations.to_pickle(annotation_result_path)

if not os.path.exists(feat_path):
    feats = cnn.get_features(final_images, layers='conv5_3', layer_sizes=[512,14,14])
    np.save(feat_path, feats)

