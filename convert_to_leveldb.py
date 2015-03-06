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

batch_size = len(train_list) / NUM_BATCHES

mean = np.zeros(3,1)

for i in xrange(NUM_BATCHES):
	dwb = data_db.write_batch()

	for j in range(i * batch_size, min((i+1) * batch_size, len(train_list))):
		curr_dog = train_list[j]
		img_file = DATA_PREFIX + curr_dog + '.jpg'
		try:
			img = caffe.io.load_image(img_file)
		except:
			continue

		img = imresize(img, size=(256, 256), interp='bicubic')
		import pdb; pdb.set_trace()
		mean += np.mean(img, axis=3)

		img = img[:, :, (2, 1, 0)]
		img = img.transpose((2, 0, 1))
		img = img.astype(np.uint8, copy=False)

		label_dict, label_arr = load_dog(train_list[j])
		label = int(curr_dog[0:3])

		img_datum= caffe.io.array_to_datum(img, label)

		dwb.put('%08d_%s'.format(i, train_list[j]), img_datum.SerializeToString())

	dwb.write()

data_db.close()
