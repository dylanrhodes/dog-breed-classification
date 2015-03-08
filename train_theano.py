import pdb
import random
from sklearn.metrics import mean_squared_error

from extract_training_faces import *
from load_theano_data import *

random.seed(13131313)

train_list = get_training_list()
test_list = get_testing_list()

train_list = train_list[:1000]

random.shuffle(train_list)
random.shuffle(test_list)

X_train, y_train = load_data(train_list + test_list[1000:])
X_test, y_test = load_data(test_list[:1000])

conv_net = train_conv_network(X_train, y_train)

print "MEAN SQUARED ERROR: {}".format(mean_squared_error(conv_net.predict(X_test), y_test))

pdb.set_trace()