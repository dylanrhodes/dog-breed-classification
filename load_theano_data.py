import cPickle as pickle
from lasagne import layers
from lasagne.nonlinearities import rectify_leaky
from lasagne.updates import nesterov_momentum
import matplotlib.pyplot as plt
from nolearn.lasagne import NeuralNet, BatchIterator
import pdb
import random
from scipy.misc import imread, imresize
from sklearn.metrics import mean_squared_error
import theano

from extract_training_faces import *

IMAGE_PREFIX = '/home/ubuntu/dog-breed-classification/CU_Dogs/dogImages/{}.jpg'

IMAGE_SIZE = 128
NUM_CHANNELS = 3

PART_FLIP_IDXS = [
	[0, 1],
	[3, 7],
	[4, 6],
]

random.seed(13131313)

def load_data(img_list):
	print 'LOADING IMAGE DATA...'

	X = np.zeros((len(img_list), NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
	y = np.zeros((len(img_list), 16), dtype=np.float32)

	for idx, dog_path in enumerate(img_list):
		img = imread(IMAGE_PREFIX.format(dog_path))
		orig_size = img.shape
	
		try:
			img = imresize(img, (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)) * 1.0 / 255
			img = img.transpose((2, 1, 0))
			X[idx,:,:,:] = img.astype(np.float32)
		except ValueError:
			continue # Skip malformed files in dataset

		point_dict, point_arr = load_dog(dog_path)
		point_arr = point_arr.astype(np.float64)

		x_scale = IMAGE_SIZE * 1.0 / orig_size[1]
		y_scale = IMAGE_SIZE * 1.0 / orig_size[0]

		point_arr[:,0] = ((point_arr[:,0] * x_scale) - (IMAGE_SIZE / 2)) / (IMAGE_SIZE / 2)
		point_arr[:,1] = ((point_arr[:,1] * y_scale) - (IMAGE_SIZE / 2)) / (IMAGE_SIZE / 2)
		
		point_arr = np.reshape(point_arr, (1, point_arr.shape[0] * point_arr.shape[1]))
		y[idx,:] = point_arr.astype(np.float32)

		if idx % 500 == 0: print '{} IMAGES LOADED...'.format(idx)

	return X, y

def train_dense_network(X, y):
	dense_net = NeuralNet(
	    layers=[
	        ('input', layers.InputLayer),
	        ('hidden', layers.DenseLayer),
	        ('output', layers.DenseLayer),
	    ],
	    
	    input_shape=(None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
	    hidden_num_units=100,  
	    output_nonlinearity=None,
	    output_num_units=16,

	    update=nesterov_momentum,
	    update_learning_rate=0.01,
	    update_momentum=0.9,

	    regression=True,
	    max_epochs=50,
	    verbose=1,
	)

	dense_net.fit(X, y)
	return dense_net

class AugmentBatchIterator(BatchIterator):
	def transform(self, Xb, yb):
		Xb, yb = super(AugmentBatchIterator, self).transform(Xb, yb)

		batch_size = Xb.shape[0]
		flip_idx = np.random.choice(batch_size, batch_size / 2, replace=False)
		Xb[flip_idx] = Xb[flip_idx, :, ::-1, :]

		# Jitter contrast
		contrast_jitter = np.random.normal(1, 0.07, (Xb.shape[0], 1, 1, 1))
		Xb *= contrast_jitter

		# Jitter tint
		tint_jitter = np.random.uniform(0.0, 0.05, (Xb.shape[0], 3, 1, 1))
		Xb += tint_jitter

		if yb is not None:
			x_cols = np.array([i for i in xrange(yb.shape[1]) if i % 2 == 0])
			x_cols = np.tile(x_cols, len(flip_idx))

			yb[np.repeat(flip_idx, len(x_cols) / len(flip_idx)), x_cols] = yb[np.repeat(flip_idx, len(x_cols) / len(flip_idx)), x_cols] * -1

			# Swap left parts for right parts eg. LEFT EYE <-> RIGHT EYE
			for left, right in self.part_flips:
				tmp = yb[flip_idx, left]
				yb[flip_idx, left] = yb[flip_idx, right]
				yb[flip_idx, right] = tmp

		return Xb, yb

class AdjustVariable(object):
	def __init__(self, name, start=0.01, stop=0.0001):
		self.name = name
		self.start, self.stop = start, stop
		self.ls = None

	def __call__(self, nn, train_history):
		if self.ls is None:
			self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

		epoch = train_history[-1]['epoch']
		new_value = np.cast['float32'](self.ls[epoch - 1])
		getattr(nn, self.name).set_value(new_value)

def train_conv_network(X, y, flip_idxs, out_file_name):
	conv_net = NeuralNet(
		layers=[
			('input', layers.InputLayer),
			('conv1a', layers.Conv2DLayer),
			('conv1', layers.Conv2DLayer),
			('pool1', layers.MaxPool2DLayer),
			('dropout1', layers.DropoutLayer),
			('conv2a', layers.Conv2DLayer),
        	('conv2', layers.Conv2DLayer),
	        ('pool2', layers.MaxPool2DLayer),
	        ('dropout2', layers.DropoutLayer),
	        ('conv3a', layers.Conv2DLayer),
	        ('conv3', layers.Conv2DLayer),
	        ('pool3', layers.MaxPool2DLayer),
	        ('dropout3', layers.DropoutLayer),
	        ('hidden4', layers.DenseLayer),
	        ('dropout4', layers.DropoutLayer),
	        ('hidden5', layers.DenseLayer),
	        ('output', layers.DenseLayer),
		],

		input_shape=(None, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE),
		conv1a_num_filters=16, conv1a_filter_size=(7, 7), conv1a_nonlinearity=rectify_leaky, 
	    conv1_num_filters=32, conv1_filter_size=(5, 5), conv1_nonlinearity=rectify_leaky, pool1_ds=(2, 2), dropout1_p=0.2,
	    conv2a_num_filters=64, conv2a_filter_size=(5, 5), conv2a_nonlinearity=rectify_leaky,
	    conv2_num_filters=64, conv2_filter_size=(5, 5), conv2_nonlinearity=rectify_leaky, pool2_ds=(2, 2), dropout2_p=0.2,
	    conv3a_num_filters=128, conv3a_filter_size=(3, 3), conv3a_nonlinearity=rectify_leaky,
	    conv3_num_filters=256, conv3_filter_size=(3, 3), conv3_nonlinearity=rectify_leaky, pool3_ds=(2, 2), dropout3_p=0.3,
	    hidden4_num_units=1250, dropout4_p=0.7, hidden5_num_units=1000,
	    output_num_units=y.shape[1], output_nonlinearity=None,

	    batch_iterator_train=AugmentBatchIterator(batch_size=180),

	    update_learning_rate=theano.shared(np.cast['float32'](0.03)),
    	update_momentum=theano.shared(np.cast['float32'](0.9)),

    	on_epoch_finished=[
	        AdjustVariable('update_learning_rate', start=0.01, stop=0.0001),
	        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        ],

	    regression=True,
	    max_epochs=10,
	    verbose=1,
	)

	conv_net.batch_iterator_train.part_flips = flip_idxs

	conv_net.fit(X, y)

	with open(out_file_name, 'wb') as out_file:
		pickle.dump(conv_net, out_file, protocol=pickle.HIGHEST_PROTOCOL)

	return conv_net

def plot_loss(network):
	train_loss = np.array([i["train_loss"] for i in network.train_history_])
	valid_loss = np.array([i["valid_loss"] for i in network.train_history_])

	plt.plot(train_loss, linewidth=3, label="train")
	plt.plot(valid_loss, linewidth=3, label="valid")
	plt.grid()
	plt.legend()
	plt.xlabel("epoch")
	plt.ylabel("loss")
	plt.yscale("log")
	plt.show()

def plot_predictions(network, X, y):
	y_pred = network.predict(X)

	y = np.reshape(y, (y.shape[0], y.shape[1] / 2, 2))
	y_pred = np.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1] / 2, 2))

	fig = plt.figure(figsize=(6, 6))
	fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

	for i, idx in enumerate(np.random.choice(y.shape[0], 16, replace=False)):
		scale = IMAGE_SIZE / 2
		img = X[idx].transpose((2, 1, 0))

		ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
		ax.imshow(img)
		
		ax.scatter(y[idx, :, 0] * scale + scale, y[idx, :, 1] * scale + scale, marker='x', color='g', s=10)
		ax.scatter(y_pred[idx, :, 0] * scale + scale, y_pred[idx, :, 1] * scale + scale, marker='x', color='r', s=10)
		
	plt.show()

def plot_prediction(network, X, y, idx):
	img = X[idx].transpose((2,1,0))

	scale = IMAGE_SIZE / 2

	y_pred = network.predict(X)
	y_pred = y_pred[idx] * scale + scale
	y_pred = np.reshape(y_pred, (2, len(y_pred) / 2))

	y_test = y[idx] * scale + scale
	y_test = np.reshape(y_test, (2, len(y_test) / 2))

	plt.imshow(img)
	plt.plot(y_pred, 'rx')
	plt.plot(y_test, 'go')
	plt.show()
