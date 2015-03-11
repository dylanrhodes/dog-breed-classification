import cPickle as pickle
import numpy as np
from scipy.misc import imrotate, imresize, imsave

from extract_training_faces import *
from load_theano_data import *

CURRENT_MODEL = 'conv_net_dropout_large.pk'
TRAIN_FACE_DIR = './rand_crops/train_set/{}.png'
TEST_FACE_DIR = './rand_crops/test_set/{}.png'

CROP_SIZE = 64
NUM_CHANNELS = 3

NUM_RAND_CROPS = 5

def crop_box(dog_path, bounding_box, slope):
	img = imread(IMAGE_PREFIX.format(dog_path))

	x_scale = img.shape[1] * 1.0 / IMAGE_SIZE
	y_scale = img.shape[0] * 1.0 / IMAGE_SIZE

	bounding_box[0,:] *= x_scale
	bounding_box[1,:] *= y_scale

	resized_imgs = np.zeros((NUM_RAND_CROPS, CROP_SIZE, CROP_SIZE, NUM_CHANNELS))

	for i in xrange(NUM_RAND_CROPS):
		theta = (np.arctan2(slope[0], slope[1]) + np.pi / 2)
		theta = np.random.normal(theta, 0.05)

		theta_deg = theta * 180 / np.pi * -1
		rotation_mat = np.array([[np.cos(theta), -1 * np.sin(theta)], [np.sin(theta), np.cos(theta)]])

		img_rotate = imrotate(img, theta_deg, interp='bicubic')
		
		box_rotate = rotation_mat.dot(bounding_box.T)

		rand_shift = np.random.normal(0.0, 8.0)

		x_min = max(round((box_rotate[0,0] + box_rotate[0,1]) / 2 + rand_shift), 0.0)
		x_max = min(round((box_rotate[0,2] + box_rotate[0,3]) / 2 + rand_shift), img_rotate.shape[1])
		y_min = max(round((box_rotate[1,0] + box_rotate[1,3]) / 2 + rand_shift), 0.0)
		y_max = min(round((box_rotate[1,1] + box_rotate[1,2]) / 2 + rand_shift), img_rotate.shape[0])

		try:
			cropped_img = img_rotate[y_min:y_max, x_min:x_max, :]
			resized_imgs[i] = imresize(cropped_img, (CROP_SIZE, CROP_SIZE, NUM_CHANNELS), interp='bicubic')
		except:
			import pdb; pdb.set_trace()

	return resized_imgs

def load_model(filename):
	return pickle.load(open(filename, 'rb'))

def write_cropped_faces(file_list, X, output_dir):
	network = load_model(CURRENT_MODEL)
	y_pred = network.predict(X)
	y_pred = np.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1] / 2, 2))

	for i, dog_file in enumerate(file_list):
		img = X[i].transpose((2,1,0))

		scale = IMAGE_SIZE / 2
		pred_points = {
			'RIGHT_EYE': y_pred[i, 0, :] * scale + scale,
			'LEFT_EYE': y_pred[i, 1, :] * scale + scale,
			'NOSE': y_pred[i, 2, :] * scale + scale,
		}

		corners, slope, distance = get_face_box(pred_points)
		cropped_imgs = crop_box(dog_file, corners, slope)

		for i in xrange(NUM_RAND_CROPS):
			crop_file = 'c{}_'.format(i) + str(int(dog_file[:3])) + '_' + dog_file.split('/')[1]
			imsave(output_dir.format(crop_file), cropped_img)

train_list = get_training_list()
test_list = get_testing_list()

X_train, y_train = load_data(train_list)
X_test, y_test = load_data(test_list)

write_cropped_faces(train_list, X_train, TRAIN_FACE_DIR)
write_cropped_faces(test_list, X_test, TEST_FACE_DIR)
