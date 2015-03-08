import cPickle as pickle
from lasagne import layers
from lasagne.updates import nesterov_momentum
import matplotlib.pyplot as plt
from nolearn.lasagne import NeuralNet, BatchIterator
import random
from scipy.misc import imread, imresize
import theano

from extract_training_faces import *

IMAGE_PREFIX = '/home/ubuntu/dog-breed-classification/CU_Dogs/dogImages/{}.jpg'

IMAGE_SIZE = 128
NUM_CHANNELS = 3

random.seed(13131313)

def load_data(img_list):
	print 'LOADING IMAGE DATA...'

	X = np.zeros((len(img_list), NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
	y = np.zeros((len(img_list), 2), dtype=np.float32)

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

		x_scale = IMAGE_SIZE * 1.0 / orig_size[1]
		y_scale = IMAGE_SIZE * 1.0 / orig_size[0]

		x_loc = ((point_dict['NOSE'][0] * x_scale) - (IMAGE_SIZE / 2)) / (IMAGE_SIZE / 2)
		y_loc = ((point_dict['NOSE'][1] * y_scale) - (IMAGE_SIZE / 2)) / (IMAGE_SIZE / 2)
		y[idx,:] = np.array([x_loc, y_loc]).astype(np.float32)

		if idx % 100 == 0: print '{} IMAGES LOADED...'.format(idx)

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
	    output_num_units=2,

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

		if yb is not None:
			yb[flip_idx, 0] = yb[flip_idx, 0] * -1

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
	    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_ds=(2, 2), dropout1_p=0.3,
	    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_ds=(2, 2), dropout2_p=0.4,
	    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_ds=(2, 2), dropout3_p=0.5,
	    hidden4_num_units=500, dropout4_p=0.5, hidden5_num_units=500,
	    output_num_units=2, output_nonlinearity=None,

	    batch_iterator_train=AugmentBatchIterator(batch_size=256),

	    update_learning_rate=theano.shared(np.cast['float32'](0.03)),
    	update_momentum=theano.shared(np.cast['float32'](0.9)),

    	on_epoch_finished=[
	        AdjustVariable('update_learning_rate', start=0.01, stop=0.0001),
	        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        ],

	    regression=True,
	    max_epochs=100,
	    verbose=1,
	)

	conv_net.fit(X, y)
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

	fig = plt.figure(figsize=(6, 6))
	fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

	def plot_sample(img, y, y_pred, axis):
		scale = IMAGE_SIZE / 2
	    axis.imshow(img.transpose((2, 1, 0)))
	    axis.scatter(y[0] * scale + scale, y[1] * scale + scale, marker='x', color='g', s=10)
	    axis.scatter(y_pred[0] * scale + scale, y_pred[i] * scale + scale, marker='x', color='r', s=10)

	for i in range(16):
		ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
		plot_sample(X[i], y[i], y_pred[i], ax)

	plt.show()

train_list = get_training_list()
test_list = get_testing_list()

random.shuffle(train_list)
random.shuffle(test_list)

X_train, y_train = load_data(train_list)
X_test, y_test = load_data(test_list[:100])

conv_net = train_conv_network(X_train, y_train)

pickle.dump(conv_net, file('conv_net_dropout.pk', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

import pdb; pdb.set_trace()