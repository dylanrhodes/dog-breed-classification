from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import random
from scipy.misc import imread, imresize

from extract_training_faces import *

IMAGE_PREFIX = '/home/ubuntu/dog-breed-classification/CU_Dogs/dogImages/{}.jpg'

IMAGE_SIZE = 256
NUM_CHANNELS = 3

random.seed(13131313)

def load_data(img_list):
	X = np.zeros((len(img_list), IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
	y = np.zeros((len(img_list), 2))

	for idx, dog_path in enumerate(img_list):
		img = imread(IMAGE_PREFIX.format(dog_path))
		orig_size = img.shape
		img = imresize(img, (IMAGE_SIZE, IMAGE_SIZE))

		point_dict, point_arr = load_dog(dog_path)

		X[idx,:,:,:] = img

		y_scale = IMAGE_SIZE * 1.0 / orig_size[0]
		x_scale = IMAGE_SIZE * 1.0 / orig_size[1]

		scaled_loc = np.array([point_dict['NOSE'][0] * y_scale, point_dict['NOSE'][1] * x_scale])
		y[idx,:] = scaled_loc

		import pdb; pdb.set_trace()

	return X, y

dense_net = NeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('hidden', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, IMAGE_SIZE * IMAGE_SIZE),  # 96x96 input pixels per batch
    hidden_num_units=100,  
    output_nonlinearity=None,  # output layer uses identity function
    output_num_units=30,  # 30 target values

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,  # flag to indicate we're dealing with regression problem
    max_epochs=400,  # we want to train this many epochs
    verbose=1,
    )

train_list = get_training_list()
test_list = get_testing_list()

X_train, y_train = load_data(train_list)
X_test, y_test = load_data(test_list)

dense_net.fit(X_train, y_train)

