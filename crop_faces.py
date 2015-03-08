import cPickle as pickle
import numpy as np
from scipy.misc import imrotate, imresize, imsave

from extract_training_faces import *
from load_theano_data import *

CURRENT_MODEL = 'conv_net_dropout_large.pk'
FACE_DIR = './cropped_images/{}.png'

CROP_SIZE = 64

def crop_box(img, bounding_box, slope):
	theta = np.atan2(slope[0], slope[1])
	theta_deg = theta * 180 / np.pi
	rotation_mat = np.array([[np.cos(theta), -1 * np.sin(theta)], [np.sin(theta), np.cos(theta)]])

	img_rotate = imrotate(img, theta_deg, interp='bicubic')
	
	box_rotate = rotation_mat * bounding_box.T
	import pdb; pdb.set_trace()	

	return img_rotate[:, :50, :50]

def load_model(filename):
	return pickle.load(open(filename, 'rb'))

def write_cropped_faces(file_list, X):
	network = load_model(CURRENT_MODEL)
	y_pred = network.predict(X)
	y_pred = np.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1] / 2, 2))

	for i, dog_file in enumerate(file_list):
		img = X[i].transpose((2,1,0))

		pred_points = {
			'RIGHT_EYE': y_pred[i, 0, :],
			'LEFT_EYE': y_pred[i, 1, :],
			'NOSE': y_pred[i, 2, :],
		}

		corners, slope, distance = get_face_box(pred_points)
		cropped_img = crop_box(img, corners, slope)

		crop_file = 'crop_' + str(int(dog_file[:3])) + '_' + dog_file.split('/')[1]
		imwrite(FACE_DIR.format(crop_file), cropped_img)

train_list = get_training_list()
test_list = get_testing_list()

X_train, y_train = load_data(train_list + test_list[1000:])
X_test, y_test = load_data(test_list[:1000])

write_cropped_faces(test_list[:100], X_test[:100,:,:,:])