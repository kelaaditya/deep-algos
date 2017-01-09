import theano
import theano.tensor as T
import lasagne
import numpy as np

from batchify import *
from load_ORL_faces import *
from cnn_orl import *

import matplotlib.pyplot as plt


if __name__ == "__main__":
	epochs = 100
	batch_size  = 10

	'''
	Training Loss and Test Accuracy calculated for different learning rates: {0.01, 0.02, 0.03, 0.04, 0.05}
	'''
	training_loss_01, test_accuracy_01, W, b = cnn_orl(trainX, trainY, testX, testY, epochs, batch_size, 0.01)
	training_loss_02, test_accuracy_02, _, _ = cnn_orl(trainX, trainY, testX, testY, epochs, batch_size, 0.02)
	training_loss_03, test_accuracy_03, _, _ = cnn_orl(trainX, trainY, testX, testY, epochs, batch_size, 0.03)
	training_loss_04, test_accuracy_04, _, _ = cnn_orl(trainX, trainY, testX, testY, epochs, batch_size, 0.04)
	training_loss_05, test_accuracy_05, _, _ = cnn_orl(trainX, trainY, testX, testY, epochs, batch_size, 0.05)


	plt.title('Training Loss vs Batch')
	plt.plot(np.arange(len(training_loss_01)), training_loss_01, label='eta = 0.01')
	plt.plot(np.arange(len(training_loss_02)), training_loss_02, label='eta = 0.02')
	plt.plot(np.arange(len(training_loss_03)), training_loss_03, label='eta = 0.03')
	plt.plot(np.arange(len(training_loss_04)), training_loss_04, label='eta = 0.04')
	plt.plot(np.arange(len(training_loss_05)), training_loss_05, label='eta = 0.05')
	plt.legend()
	plt.show()

	plt.title('Test Accuracy vs Batch')
	plt.plot(np.arange(len(test_accuracy_01)), test_accuracy_01, label='eta = 0.01')
	plt.plot(np.arange(len(test_accuracy_02)), test_accuracy_02, label='eta = 0.02')
	plt.plot(np.arange(len(test_accuracy_03)), test_accuracy_03, label='eta = 0.03')
	plt.plot(np.arange(len(test_accuracy_04)), test_accuracy_04, label='eta = 0.04')
	plt.plot(np.arange(len(test_accuracy_05)), test_accuracy_05, label='eta = 0.05')
	plt.legend()
	plt.show()

	#Visualization of the first CNN layer
	#Weights and biases added together
	#Each filter (30) displayed in the grid
	first_layer_weights = [weights[0]+bias for weights, bias in zip(W.get_value(), b.get_value())]
	gs = gridspec.GridSpec(10, 10)
	for i in range(len(test)):
		ax = plt.subplot(gs[i])
		ax.grid()
		ax.set_xticks([])
		ax.set_yticks([])
		ax.imshow(test[i], cmap='gray')
		
