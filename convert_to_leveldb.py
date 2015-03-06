import caffe
import numpy as np
import plyvel
import random
from scipy.misc import imresize

from extract_training_faces import *

DATA_PREFIX = '/root/CU_Dogs/dogImages/'
NUM_BATCHES = 100

random.seed(13131313)

train_list = get_training_list()
random.shuffle(train_list)

test_list = get_testing_list()

data_db = plyvel.DB('/tmp/data_db', create_if_missing=True)
test_db = plyvel.DB('/tmp/test_db', create_if_missing=True)

def fill_in_db(img_list, database):
	batch_size = len(img_list) / NUM_BATCHES

	mean = np.zeros(3)
	num_images = 0

	for i in xrange(NUM_BATCHES):
		wb = database.write_batch()

		for j in range(i * batch_size, min((i+1) * batch_size, len(img_list))):
			curr_dog = img_list[j]
			img_file = DATA_PREFIX + curr_dog + '.jpg'
			try:
				img = caffe.io.load_image(img_file)
			except:
				continue

			img = imresize(img, size=(256, 256), interp='bicubic')
			mean += np.mean(img, axis=(0, 1))
			num_images += 1
			
			if num_images % 100 == 0: 
				print '{} images processed...'.format(num_images)
				print 'SHAPE: {}, LABEL: {}, TAG: {}'.format(img.shape, int(curr_dog[0:3]), curr_dog[3:23])

			img = img[:, :, (2, 1, 0)]
			img = img.transpose((2, 0, 1))
			img = img.astype(np.uint8, copy=False)

			label_dict, label_arr = load_dog(img_list[j])
			label = int(curr_dog[0:3])

			img_datum= caffe.io.array_to_datum(img, label)

			wb.put('%08d_%s'.format(i, img_list[j]), img_datum.SerializeToString())

		wb.write()

	mean /= num_images
	print 'MEAN OF CHANNELS: ' + str(mean)

	database.close()

print 'PROCESSING TRAINING DATA...\n'
fill_in_db(train_list, data_db)

print 'PROCESSING TEST DATA...\n'
fill_in_db(train_list, test_db)