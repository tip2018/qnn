
import numpy as np
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.datasets import mnist

def load_dataset(dataset):
	if (dataset == "CIFAR-10"):

		print('Loading CIFAR-10 dataset...')
		
		train_set_size = 45000
		(x_train, y_train), (x_test, y_test) = cifar10.load_data();
		x_valid = x_train[train_set_size:,:,:,:]
		y_valid = y_train[train_set_size:,:]
		x_train = x_train[0:train_set_size,:,:,:]
		y_train = y_train[0:train_set_size:,:]

		x_train = np.subtract(np.multiply(2. / 255., x_train), 1.)
		x_valid = np.subtract(np.multiply(2. / 255., x_valid), 1.)
		x_test = np.subtract(np.multiply(2. / 255., x_test), 1.)

		# flatten targets
		y_train = np.hstack(y_train)
		y_valid = np.hstack(y_valid)
		y_test = np.hstack(y_test)

		# Onehot the targets
		y_train = np.float32(np.eye(10)[y_train])
		y_valid = np.float32(np.eye(10)[y_valid])
		y_test = np.float32(np.eye(10)[y_test])

		# for hinge loss
		y_train = 2 * y_train - 1.
		y_valid = 2 * y_valid - 1.
		y_test = 2 * y_test - 1.

		# enlarge train data set by mirrroring
		x_train_flip = x_train[:, :, ::-1, :]
		y_train_flip = y_train
		x_train = np.concatenate((x_train, x_train_flip), axis=0)
		y_train = np.concatenate((y_train, y_train_flip), axis=0)
				
	elif (dataset == "CIFAR-100"):

		print('Loading CIFAR-100 dataset...')
		
		train_set_size = 45000
		(x_train, y_train), (x_test, y_test) = cifar100.load_data();
		x_valid = x_train[train_set_size:,:,:,:]
		y_valid = y_train[train_set_size:,:]
		x_train = x_train[0:train_set_size,:,:,:]
		y_train = y_train[0:train_set_size:,:]

		x_train = np.subtract(np.multiply(2. / 255., x_train), 1.)
		x_valid = np.subtract(np.multiply(2. / 255., x_valid), 1.)
		x_test = np.subtract(np.multiply(2. / 255., x_test), 1.)

		# flatten targets
		y_train = np.hstack(y_train)
		y_valid = np.hstack(y_valid)
		y_test = np.hstack(y_test)

		# Onehot the targets
		y_train = np.float32(np.eye(100)[y_train])
		y_valid = np.float32(np.eye(100)[y_valid])
		y_test = np.float32(np.eye(100)[y_test])

		# for hinge loss
		y_train = 2 * y_train - 1.
		y_valid = 2 * y_valid - 1.
		y_test = 2 * y_test - 1.

		# enlarge train data set by mirrroring
		x_train_flip = x_train[:, :, ::-1, :]
		y_train_flip = y_train
		x_train = np.concatenate((x_train, x_train_flip), axis=0)
		y_train = np.concatenate((y_train, y_train_flip), axis=0)

	elif (dataset == "MNIST"):

		print('Loading MNIST dataset...')

		
		train_set_size = 50000
		(x_train, y_train), (x_test, y_test) = mnist.load_data();
		x_valid = x_train[train_set_size:,:,:,:]
		y_valid = y_train[train_set_size:,:]
		x_train = x_train[0:train_set_size,:,:,:]
		y_train = y_train[0:train_set_size:,:]
		

		x_train = np.transpose(np.reshape(np.subtract(np.multiply(2. / 255., x_train), 1.), (-1, 1, 28, 28)),(0,2,3,1))
		x_valid = np.transpose(np.reshape(np.subtract(np.multiply(2. / 255., x_valid), 1.), (-1, 1,  28, 28)),(0,2,3,1))
		x_test = np.transpose(np.reshape(np.subtract(np.multiply(2. / 255., x_test), 1.), (-1, 1,  28, 28)),(0,2,3,1))
		# flatten targets
		y_train = np.hstack(y_train)
		y_valid = np.hstack(y_valid)
		y_test = np.hstack(y_test)

		# Onehot the targets
		y_train = np.float32(np.eye(10)[y_train])
		y_valid = np.float32(np.eye(10)[y_valid])
		y_test = np.float32(np.eye(10)[y_test])

		# for hinge loss
		y_train = 2 * y_train - 1.
		y_valid = 2 * y_valid - 1.
		y_test = 2 * y_test - 1.
		# enlarge train data set by mirrroring
		x_train_flip = x_train[:, :, ::-1, :]
		y_train_flip = y_train
		x_train = np.concatenate((x_train, x_train_flip), axis=0)
		y_train = np.concatenate((y_train, y_train_flip), axis=0)

	else:
		print("wrong dataset given")

	return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)
