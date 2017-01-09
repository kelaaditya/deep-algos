import theano
import theano.tensor as T
import lasagne
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from batchify import *
from load_ORL_faces import *


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

    return(training_loss_list, test_accuracy_list)


if __name__ == "__main__":
    training_loss, test_accuracy = cnn_orl(trainX, trainY, testX, testY, 1, 10, 0.01)
    print(np.array(training_loss))
    print(test_accuracy)
