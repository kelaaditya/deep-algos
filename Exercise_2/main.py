import theano
import theano.tensor as T
import lasagne
import numpy as np

from load_mnist import *
from batchify import *
from mlp_sgd import *
from mlp_nesterov import *
from mlp_nesterov_l1 import *
from mlp_nesterov_l2 import *

import matplotlib.pyplot as plt


if __name__ == "__main__":
	X_train, y_train, X_test, y_test = load_data()
	epochs = 1
	batch_size  = 500

	training_loss_sgd, test_accuracy_sgd = mlp_sgd(X_train, y_train, X_test, y_test, epochs, 500)
	training_loss_nes, test_accuracy_nes = mlp_nesterov(X_train, y_train, X_test, y_test, epochs, 500)
	training_loss_l1, test_accuracy_l1 = mlp_nesterov_l1(X_train, y_train, X_test, y_test, epochs, 500)
	training_loss_l2, test_accuracy_l2 = mlp_nesterov_l2(X_train, y_train, X_test, y_test, epochs, 500)

	'''
	Training loss plotted as a function of batches fed
	'''
        plt.title('Training Loss vs Batch')
	plt.plot(np.arange(len(training_loss_sgd)), training_loss_sgd, label='SGD')
	plt.plot(np.arange(len(training_loss_nes)), training_loss_nes, label='Nesterov Momentum')
	plt.plot(np.arange(len(training_loss_l1)), training_loss_l1, label='Nesterov with l1 reg')
	plt.plot(np.arange(len(training_loss_l2)), training_loss_l2, label='Nesterov with l2 reg')
        plt.legend()
	plt.show()

        plt.title('Test Accuracy vs Batch')
	plt.plot(np.arange(len(test_accuracy_sgd)), test_accuracy_sgd, label='SGD')
	plt.plot(np.arange(len(test_accuracy_nes)), test_accuracy_nes, label='Nesterov Momentum')
	plt.plot(np.arange(len(test_accuracy_l1)), test_accuracy_l1, label='Nesterov with l1 reg')
	plt.plot(np.arange(len(test_accuracy_l2)), test_accuracy_l2, label='Nesterov with l2 reg')
        plt.legend()
	plt.show()
