import pandas as pd
import numpy as np
import os
import glob
import cPickle
from cnn_util import *

vgg_model = '/home/dalek/git-repos/show_attend_and_tell.tensorflow/vgg/VGG_ILSVRC_19_layers.caffemodel'
vgg_deploy = '/home/dalek/git-repos/show_attend_and_tell.tensorflow/vgg/VGG_ILSVRC_19_layers_deploy.prototxt'

i = 2
test_image_path = '/home/dalek/git-repos/show_attend_and_tell.tensorflow/test_images/test6.jpg'
feat_path = '/home/dalek/git-repos/show_attend_and_tell.tensorflow/test_feats5.npy'



cnn = CNN(model=vgg_model, deploy=vgg_deploy, width=224, height=224)

image_list = glob.glob(test_image_path)
np_img_list = np.array(image_list)
final_images = []
for images in np_img_list:
	if os.path.exists(images):
		final_images.append(images)


feats = cnn.get_features(np_img_list, layers='conv5_3', layer_sizes=[512,14,14])
np.save(feat_path, feats)

