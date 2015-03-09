import cPickle as pickle
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
import numpy as np
from os import listdir
import pdb
import random
from scipy.misc import imread
import theano

from load_theano_data import AdjustVariable, plot_loss

TRAIN_SET_DIR = './cropped_images/train_set/'
TEST_SET_DIR = './cropped_images/test_set/'

IMAGE_SIZE = 64

random.seed(13131313)

def load_data(img_list):
	print 'LOADING IMAGE DATA...'

	X = np.zeros((len(img_list), NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
	y = np.zeros((len(img_list), 1), dtype=np.int8)

	for idx, dog_path in enumerate(img_list):
		img = imread(dog_path)
		img = img.transpose((2, 1, 0)) * 1.0 / 255

		X[idx,:,:,:] = img.astype(np.float32)
		y[idx,:] = int(dog_path.split('_')[3])

		if idx % 500 == 0: print '{} IMAGES LOADED...'.format(idx)

	return X, y

def train_conv_network(X, y):
	conv_net = NeuralNet(
		layers=[
			('input', layers.InputLayer),
			('conv1', layers.Conv2DLayer),
			('pool1', layers.MaxPool2DLayer),
			('dropout1', layers.DropoutLayer),
        	('conv2', layers.Conv2DLayer),
	        ('pool2', layers.MaxPool2DLayer),
	        ('dropout2', layers.DropoutLayer),
	        ('conv3', layers.Conv2DLayer),
	        ('pool3', layers.MaxPool2DLayer),
	        ('dropout3', layers.DropoutLayer),
	        ('hidden4', layers.DenseLayer),
	        ('dropout4', layers.DropoutLayer),
	        ('hidden5', layers.DenseLayer),
	        ('output', layers.DenseLayer),
		],

		input_shape=(None, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE),
	    conv1_num_filters=32, conv1_filter_size=(5, 5), pool1_ds=(2, 2), dropout1_p=0.5,
	    conv2_num_filters=64, conv2_filter_size=(5, 5), pool2_ds=(2, 2), dropout2_p=0.6,
	    conv3_num_filters=128, conv3_filter_size=(5, 5), pool3_ds=(2, 2), dropout3_p=0.7,
	    hidden4_num_units=1000, dropout4_p=0.7, hidden5_num_units=1000,
	    output_num_units=16, output_nonlinearity=None,

	    #batch_iterator_train=AugmentBatchIterator(batch_size=256),

	    update_learning_rate=theano.shared(np.cast['float32'](0.03)),
    	update_momentum=theano.shared(np.cast['float32'](0.9)),

    	on_epoch_finished=[
	        AdjustVariable('update_learning_rate', start=0.01, stop=0.0001),
	        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        ],

	    regression=False,
	    max_epochs=10,
	    verbose=1,
	)

	conv_net.fit(X, y)

	with open('breed_net.pk', 'wb') as out_file:
		pickle.dump(conv_net, out_file, protocol=pickle.HIGHEST_PROTOCOL)

	return conv_net

train_list = [TRAIN_SET_DIR + file_name for file_name in listdir(TRAIN_SET_DIR)]
test_list = [TEST_SET_DIR + file_name for file_name in listdir(TEST_SET_DIR)]

X_train, y_train = load_data(train_list)
X_test, y_test = load_data(test_list)

breed_net = train_conv_network(X_train, y_train)

y_pred = breed_net.predict(X_test)
accuracy = np.mean(y_pred == y_test)

print 'TEST ACCURACY: {}'.format(accuracy)

pdb.set_trace()