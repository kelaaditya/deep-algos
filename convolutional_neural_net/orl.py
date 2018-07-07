import theano
import theano.tensor as T
import lasagne
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


# load data
data = np.load('ORL_faces.npz')
trainX = np.reshape(data['trainX'], (240, 1, 92, 112))/256
trainY = data['trainY']
testX = np.reshape(data['testX'], (160, 1, 92, 112))/256
testY = data['testY']


def batchify(X, y, batch_size):
    for i in range(0, len(X)-batch_size+1, batch_size):
        yield(X[i: i+batch_size], y[i: i+batch_size])


def cnn_orl(X_tr, y_tr, X_t, y_t, epochs, batch_size, learning_rate=0.01):
    X = T.tensor4('X')
    y = T.ivector('y')
    
    #Channel size is 1
    input_layer = lasagne.layers.InputLayer(shape=(None, 1, 112, 92), input_var=X)
    #convolution filter size is (20, 20)
    #Conv -> RelU -> Max Pool
    #pool_size for Max Pooling is (2, 2)
    cnn_layer_1 = lasagne.layers.Conv2DLayer(input_layer, num_filters=30, 
                                         filter_size=(10, 10),
                                         nonlinearity=lasagne.nonlinearities.rectify,
                                         W=lasagne.init.GlorotUniform()
                                        )
    max_pooling_1 = lasagne.layers.MaxPool2DLayer(cnn_layer_1, pool_size=(2, 2))
    
    #Visualization of the first convolutional layer
    #Will add W and the bias 
    W_1, b_1 = lasagne.layers.get_all_params(cnn_layer_1, trainable=True)
    

    #Another convolution layer followed by ReLU and Max Pooling
    cnn_layer_2 = lasagne.layers.Conv2DLayer(max_pooling_1, num_filters=30, 
                                         filter_size=(10, 10),
                                         nonlinearity=lasagne.nonlinearities.rectify,
                                         W=lasagne.init.GlorotUniform()
                                        )
    max_pooling_2 = lasagne.layers.MaxPool2DLayer(cnn_layer_2, pool_size=(2, 2))
    
    #A fully connected layer after the convolution layers
    fully_connected_layer_1 = lasagne.layers.DenseLayer(max_pooling_2, 
                                               num_units=256, 
                                               nonlinearity=lasagne.nonlinearities.rectify, 
                                               W=lasagne.init.GlorotUniform()
                                              )
    #fully_connected_layer_1 = lasagne.layers.DropoutLayer(fully_connected_layer_1, p=0.5)

    #output layer makes use of softmax
    output_layer = lasagne.layers.DenseLayer(fully_connected_layer_1, 
                                             num_units=20, 
                                             nonlinearity=lasagne.nonlinearities.softmax
                                            )
    
    
    prediction = lasagne.layers.get_output(output_layer)
    loss = lasagne.objectives.categorical_crossentropy(prediction, y)
    loss = loss.mean()

    #Used Stochastic Gradient Descent here without momentum
    params = lasagne.layers.get_all_params(output_layer, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate, momentum=0.9)
        
    #Disable dropout for validation
    validation_prediction = lasagne.layers.get_output(output_layer, deterministic=True)
    validation_loss = lasagne.objectives.categorical_crossentropy(validation_prediction, y)
    validation_loss = validation_loss.mean()
    validation_accuracy = T.mean(T.eq(T.argmax(validation_prediction, axis=1), y), 
                          dtype=theano.config.floatX)
    
    train_cnn = theano.function([X, y], loss, updates=updates)
    validation_cnn = theano.function([X, y], [validation_loss, validation_accuracy])

    #training_loss_list is a collection of the training loss for each batch for each epoch
    #The 'i'th epoch will have a contribution of len(training_set)/batch_size 
    # in the training loss list at the 'i'th place
    training_loss_list = []
    test_accuracy_list = []
    for i in range(epochs):
        for batch in batchify(X_tr, y_tr, batch_size):
            X_batch, y_batch = batch
            training_loss_list.append(train_cnn(X_batch, y_batch))
        
        for batch in batchify(X_t, y_t, batch_size):
            X_batch, y_batch = batch
            _, test_accuracy = validation_cnn(X_batch, y_batch)
            test_accuracy_list.append(test_accuracy*100)

    return(training_loss_list, test_accuracy_list, W_1, b_1)



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
	for i in range(len(first_layer_weights)):
		ax = plt.subplot(gs[i])
		ax.grid()
		ax.set_xticks([])
		ax.set_yticks([])
		ax.imshow(first_layer_weights[i], cmap='gray')
	plt.show()
		
