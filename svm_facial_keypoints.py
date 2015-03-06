import cv2

from extract_training_faces import *

random.seed(13131313)


train_list = get_training_list()
test_list = get_testing_list()

X_train = []
y_train = []
i = 0

print "BUILDING POSITIVE TRAINING EXAMPLES..."

for dog_file in train_list:
	image = cv2.imread(IMAGE_PREFIX.format(dog_file))
	part_dict, part_loc = load_dog(dog_file)
	center = get_center_point(part_dict)
	box, slope, dist = get_face_box(part_dict)

	X_train.append(extract_features(image, box, slope, dist))
	y_train.append(1)

	i += 1
	if i % 500 == 0: print '{}...'.format(i)

print "BUILDING NEGATIVE TRAINING EXAMPLES..."

for i in xrange(NUM_NEGATIVE_TRAIN_SAMPLES):
	img_file = random.choice(train_list)

	image = cv2.imread(IMAGE_PREFIX.format(img_file))
	part_dict, part_loc = load_dog(img_file)
	box, slope, dist = get_random_box(image, part_dict)

	X_train.append(extract_features(image, box, slope, dist))
	y_train.append(0)

	if i % 500 == 0: print '{}...'.format(i)

print "FITTING MODEL..."

X_train = np.array(X_train)
X_train = np.squeeze(X_train)
y_train = np.array(y_train)

model = svm.SVC(probability=True)
model.fit(X_train, y_train)

X_test = []
y_test = []
i = 0

print "BUILDING POSITIVE TESTING EXAMPLES..."

for dog_file in test_list:
	image = cv2.imread(IMAGE_PREFIX.format(dog_file))
	part_dict, part_loc = load_dog(dog_file)
	center = get_center_point(part_dict)
	box, slope, dist = get_face_box(part_dict)

	X_test.append(extract_features(image, box, slope, dist))
	y_test.append(1)

	i += 1
	if i % 500 == 0: print '{}...'.format(i)

print "BUILDING NEGATIVE TESTING EXAMPLES..."

for i in xrange(NUM_NEGATIVE_TEST_SAMPLES):
	img_file = random.choice(train_list)

	image = cv2.imread(IMAGE_PREFIX.format(img_file))
	part_dict, part_loc = load_dog(img_file)
	box, slope, dist = get_random_box(image, part_dict)

	X_test.append(extract_features(image, box, slope, dist))
	y_test.append(0)

	if i % 500 == 0: print '{}...'.format(i)

print "EVALUATING MODEL..."

X_test = np.array(X_test)
X_test = np.squeeze(X_test)
y_test = np.array(y_test)

prediction = model.predict(X_test)

print "MODEL ACCURACY: " + str(np.mean(prediction == y_test))
print "CONFUSION MATRIX: "
print confusion_matrix(y_test, prediction)

pred_probs = model.predict_proba(X_test)

pdb.set_trace()


"""
index_to_display = int(raw_input('Enter index of dog to display: '))
train_list = get_training_list()

while index_to_display != -1:
	dog_file = train_list[index_to_display]

	image = misc.imread(IMAGE_PREFIX.format(dog_file))
	part_dict, part_loc = load_dog(dog_file)
	center = get_center_point(part_dict)
	box, slope, dist = get_face_box(part_dict)

	features = extract_features(image, box, slope, dist)
	display_dog(dog_file)

	index_to_display = int(raw_input('Enter index of dog to display: '))
"""
