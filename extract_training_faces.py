#import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import pdb
import random
from scipy import misc
from sklearn import svm
from sklearn.metrics import confusion_matrix

IMAGE_PREFIX = '../CU_Dogs/dogImages/{}.jpg'
POINT_PREFIX = '../CU_Dogs/dogParts/{}.txt'

PART_FILE_KEY = [
	('RIGHT_EYE', 0),
	('LEFT_EYE', 1),
	('NOSE', 2),
	('RIGHT_EAR_TIP', 3),
	('RIGHT_EAR_BASE', 4),
	('HEAD_TOP', 5),
	('LEFT_EAR_BASE', 6),
	('LEFT_EAR_TIP', 7),
]

FACE_BOX_SCALE = 2.5

NUM_NEGATIVE_TRAIN_SAMPLES = 4000
NUM_NEGATIVE_TEST_SAMPLES = 3000

def load_dog(path_to_dog):	
	if '.jpg' in path_to_dog:
		path_to_dog = path_to_dog.replace('.jpg', '')

	part_locations = []
	with open(POINT_PREFIX.format(path_to_dog), 'rb') as parts:
		for line in parts.readlines():
			x, y = line.split()
			part_locations.append(np.array([int(x),int(y)]))

	part_dict = {}

	for part, index in PART_FILE_KEY:
		part_dict[part] = part_locations[index]

	return part_dict, np.array(part_locations)

def get_center_point(parts):
	center_point = np.zeros(2)
	
	center_point += parts['LEFT_EYE']
	center_point += parts['RIGHT_EYE']
	center_point += parts['NOSE']
	center_point /= 3

	return center_point

def get_center_alt(parts):
	center_point = np.zeros(2)
	
	center_point += parts['LEFT_EYE']
	center_point += parts['RIGHT_EYE']
	center_point /= 2

	center_point[1] += parts['NOSE'][1]
	center_point[1] /= 2

	return center_point

def point_in_box(point, box):
	if point[0] < np.min(box[:,0]) or point[0] > np.max(box[:,0]):
		return False
	elif point[1] < np.min(box[:,1]) or point[1] > np.max(box[:,1]):
		return False

	return True

def get_face_box(parts):
	center = get_center_point(parts)

	left_eye = parts['LEFT_EYE']
	right_eye = parts['RIGHT_EYE']

	eye_slope = np.array([right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]])
	eye_slope = eye_slope / np.linalg.norm(eye_slope)
	eye_norm = np.array([eye_slope[1] * -1, eye_slope[0]])

	inter_eye_dist = np.sqrt((left_eye[0] - right_eye[0]) ** 2 + (left_eye[1] - right_eye[1]) ** 2)
	dist = inter_eye_dist * FACE_BOX_SCALE / 2

	box_corners = [
		center + (eye_slope * dist) + (eye_norm * dist),
		center + (eye_slope * dist) - (eye_norm * dist),
		center - (eye_slope * dist) - (eye_norm * dist),
		center - (eye_slope * dist) + (eye_norm * dist),
	]

	return np.array(box_corners), eye_slope, inter_eye_dist

def get_random_box(image, parts):
	face_box, slope, dist = get_face_box(parts)

	while(True):
		center = np.array([random.randrange(image.shape[0]), random.randrange(image.shape[1])])
		slope = np.array([random.randrange(100), random.randrange(100)])
		slope = slope / np.linalg.norm(slope)
		norm = np.array([slope[1] * -1, slope[0]])
		dist = random.randrange(64, 128)

		if not point_in_box(center, face_box):
			return np.array([
				center + (slope * dist) + (norm * dist),
				center + (slope * dist) - (norm * dist),
				center - (slope * dist) - (norm * dist),
				center - (slope * dist) + (norm * dist),
			]), slope, dist

def get_training_list():
	train_images = []
	with open('/root/CU_Dogs/training.txt', 'rb') as in_file:
		for line in in_file.readlines():
			train_images.append(line.strip().replace('.jpg', ''))

	return train_images

def get_testing_list():
	test_images = []

	with open('/root/CU_Dogs/testing.txt', 'rb') as in_file:
		for line in in_file.readlines():
			test_images.append(line.strip().replace('.jpg', ''))

	return test_images

def display_dog(dog_file):
	part_dict, part_loc = load_dog(dog_file)
	center = get_center_point(part_dict)
	box, slope, dist = get_face_box(part_dict)
	box_plot = np.vstack((box, box[0,:]))

	image = misc.imread(IMAGE_PREFIX.format(dog_file))
	random_box, slope, dist = get_random_box(image, part_dict)
	random_box_plot = np.vstack((random_box, random_box[0,:]))

	plt.imshow(image)
	plt.plot(part_loc[:,0], part_loc[:,1], 'ro')
	plt.plot(center[0], center[1], 'bo')
	plt.plot(box_plot[:,0], box_plot[:,1], 'w-', linewidth=5.0)
	plt.plot(random_box_plot[:,0], random_box_plot[:,1], 'r-', linewidth=5.0)
	plt.show()

def get_keypoints(image, box, slope, dist):
	keypoints = []

	norm = np.array([slope[1] * -1, slope[0]])

	center = np.sum(box, axis=0) / 4

	nose = center - (norm * dist / 2)
	forehead = center + (norm * dist / 3)
	left_eye = forehead - (slope * dist / FACE_BOX_SCALE)
	right_eye = forehead + (slope * dist / FACE_BOX_SCALE)

	nose_scale = dist
	eye_scale = dist / FACE_BOX_SCALE

	angle = (180 - math.atan2(norm[0], norm[1]) * 180 / math.pi) % 360

	keypoints.append(cv2.KeyPoint(x=nose[0], y=nose[1], _size=nose_scale, _angle=angle))
	keypoints.append(cv2.KeyPoint(x=forehead[0], y=forehead[1], _size=eye_scale, _angle=angle))
	keypoints.append(cv2.KeyPoint(x=left_eye[0], y=left_eye[1], _size=eye_scale, _angle=angle))
	keypoints.append(cv2.KeyPoint(x=right_eye[0], y=right_eye[1], _size=eye_scale, _angle=angle))
	
	return keypoints

def extract_features(image, box, slope, dist):
	grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	sift = cv2.SIFT()

	keypoints = get_keypoints(image, box, slope, dist)
	features = sift.compute(grayscale, keypoints)
	
	kpimg = cv2.drawKeypoints(grayscale, keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	cv2.imwrite('keypoint_test.jpg', kpimg)

	return np.reshape(features[1], features[1].shape[0] * features[1].shape[1])
