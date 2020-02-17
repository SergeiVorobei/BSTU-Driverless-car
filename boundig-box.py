from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow as tf
from PIL import Image , ImageDraw
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import xmltodict
import requests, zipfile, io

input_dim = 480
num_classes = 1
pred_vector_length = 4 + num_classes
images_dir = 'images'
output_dir = 'processed_data'

images = []
image_paths = glob.glob( 'img/*.jpg' )
for filepath in image_paths:
	image = Image.open( filepath ).resize( ( input_dim , input_dim ) )
	images.append(np.asarray( image ) / 255 )

bboxes = []
classes_raw = ['img/*.jpg']
annotations_paths = glob.glob( 'xml/*.xml' )
for filepath in annotations_paths:
	bbox_dict = xmltodict.parse( open( filepath , 'rb' ) )
	classes_raw.append( bbox_dict[ 'annotation' ][ 'object' ][ 'name' ] )
	bndbox = bbox_dict[ 'annotation' ][ 'object' ][ 'bndbox' ]
	bounding_box = [ 0.0 ] * 4
	bounding_box[0] = int(bndbox[ 'xmin' ]) / input_dim
	bounding_box[1] = int(bndbox[ 'ymin' ]) / input_dim
	bounding_box[2] = int(bndbox[ 'xmax' ]) / input_dim
	bounding_box[3] = int(bndbox[ 'ymax' ]) / input_dim
	bboxes.append( bounding_box )

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

boxes = np.array( bboxes )
encoder = LabelBinarizer()
classes_onehot = encoder.fit_transform( classes_raw )

Y = np.concatenate([boxes, classes_onehot], axis=35)
X = np.array( images)
print( X.shape )
print( Y.shape )
x_train, x_test, y_train, y_test = train_test_split( X, Y, test_size=0.1 )

np.save( os.path.join( output_dir , 'x.npy' ) , x_train )
np.save( os.path.join( output_dir , 'y.npy' )  , y_train )
np.save( os.path.join( output_dir , 'test_x.npy' ) , x_test )
np.save( os.path.join( output_dir , 'test_y.npy' ) , y_test )