import caffe
#import cv2
import numpy as np
import plyvel
import random

from extract_training_faces import *

DATA_PREFIX = '/root/CU_Dogs/dogImages/'
NUM_BATCHES = 100

random.seed(13131313)

train_list = get_training_list()
random.shuffle(train_list)

test_list = get_testing_list()

data_db = plyvel.DB('/tmp/data_db', create_if_missing=True)
#label_db = plyvel.DB('/tmp/label_db', create_if_missing=True)

batch_size = len(train_list) / NUM_BATCHES

for i in xrange(NUM_BATCHES):
	dwb = data_db.write_batch()
	#lwb = label_db.write_batch()

	for j in range(i * batch_size, min((i+1) * batch_size, len(train_list))):
		curr_dog = train_list[j]
		img_file = DATA_PREFIX + curr_dog + '.jpg'
	 	print img_file
		try:
			img = caffe.io.load_image(img_file)
		except:
			continue
		#img = cv2.imread(img_file)

		img = img[:, :, (2, 1, 0)]
		img = img.transpose((2, 0, 1))
	   	img = img.astype(np.uint8, copy=False)

	   	label_dict, label_arr = load_dog(train_list[j])
	   	#nose_label = np.array(label_dict['NOSE']).reshape(2,1,1)
    		label = int(curr_dog[0:3])

    		img_datum= caffe.io.array_to_datum(img, label)
  	 	#label_datum = caffe.io.array_to_datum(nose_label)

    		dwb.put('%08d_%s'.format(i, train_list[j]), img_datum.SerializeToString())
    		#lwb.put('%08d_%s'.format(i, train_list[j], label_datum))

	dwb.write()
	#lwb.write()

data_db.close()
#label_db.close()
