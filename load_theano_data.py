from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import random
from scipy.misc import imread, imresize

from extract_training_faces import *

IMAGE_PREFIX = '/home/ubuntu/dog-breed-classification/CU_Dogs/dogImages/{}.jpg'

IMAGE_SIZE = 128
NUM_CHANNELS = 3

random.seed(13131313)

def load_data(img_list):
	print 'LOADING IMAGE DATA...'

	X = np.zeros((len(img_list), IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
	y = np.zeros((len(img_list), 2))

	for idx, dog_path in enumerate(img_list):
		img = imread(IMAGE_PREFIX.format(dog_path))
		orig_size = img.shape
	
		try:
			img = imresize(img, (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)) * 1.0 / 255
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

	import pdb; pdb.set_trace()
	return X, y

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

train_list = get_training_list()
test_list = get_testing_list()

X_train, y_train = load_data(train_list)
#X_test, y_test = load_data(test_list)

dense_net.fit(X_train, y_train)

