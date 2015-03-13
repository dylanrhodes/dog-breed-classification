import pdb
import random
from sklearn.metrics import mean_squared_error

from extract_training_faces import *
from load_theano_data import *

MODEL_MASKS = [
	[0, 1, 2, 3],
	[4, 5],
	[6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
]

PART_FLIPS = [
	[
		(0, 2),
		(1, 3),
	],
	[],
	[
		(0, 8),
		(1, 9),
		(2, 6),
		(3, 7),
	]
]

FULL_FLIP_SET = [
	(0, 2),
	(1, 3),
	(6, 14),
	(7, 15),
	(8, 12),
	(9, 13),
]

random.seed(13131313)

train_list = get_training_list()
test_list = get_testing_list()

random.shuffle(train_list)
random.shuffle(test_list)

X_train, y_train = load_data(train_list + test_list[800:])
X_test, y_test = load_data(test_list[:800])

conv_net = train_conv_network(X_train, y_train, FULL_FLIP_SET, 'keypoint_net.pk')
print "NET MSE: {}".format(mean_squared_error(conv_net.predict(X_test), y_test))

"""
eye_net = train_conv_network(X_train, y_train[:, MODEL_MASKS[0]], PART_FLIPS[0], 'eye_net.pk')
print "EYE MEAN SQ. ERROR: {}".format(mean_squared_error(eye_net.predict(X_test), y_test[:, MODEL_MASKS[0]]))

nose_net = train_conv_network(X_train, y_train[:, MODEL_MASKS[1]], PART_FLIPS[1], 'nose_net.pk')
print "NOSE MEAN SQ. ERROR: {}".format(mean_squared_error(nose_net.predict(X_test), y_test[:, MODEL_MASKS[1]]))

ear_net = train_conv_network(X_train, y_train[:, MODEL_MASKS[2]], PART_FLIPS[2], 'ear_net.pk')
print "EAR MEAN SQ. ERROR: {}".format(mean_squared_error(ear_net.predict(X_test), y_test[:, MODEL_MASKS[2]]))
"""

pdb.set_trace()