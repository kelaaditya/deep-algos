import theano
import theano.tensor as T
import lasagne
import numpy as np

from load_mnist import *
from batchify import *

import matplotlib.pyplot as plt



X_train, y_train, X_test, y_test = load_data()

def mlp_nesterov_l1(X_tr, y_tr, X_t, y_t, epochs, batch_size):
    X = T.tensor4('X')
    y = T.ivector('y')
    
    input_layer = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=X)
    #channel = 1 as images are black-white
    input_layer_drop = lasagne.layers.DropoutLayer(input_layer, p=0.2)
        
    hidden_layer_1 = lasagne.layers.DenseLayer(input_layer_drop, 
                                               num_units=800, 
                                               nonlinearity=lasagne.nonlinearities.rectify, 
                                               W=lasagne.init.GlorotUniform()
                                              )
    hidden_layer_1_drop = lasagne.layers.DropoutLayer(hidden_layer_1, p=0.5)
        
    hidden_layer_2 = lasagne.layers.DenseLayer(hidden_layer_1_drop, 
                                               num_units=800, 
                                               nonlinearity=lasagne.nonlinearities.rectify, 
                                               W=lasagne.init.GlorotUniform()
                                              )
    hidden_layer_2_drop = lasagne.layers.DropoutLayer(hidden_layer_2, p=0.5)
        
    output_layer = lasagne.layers.DenseLayer(hidden_layer_2_drop, 
                                             num_units=10, 
                                             nonlinearity=lasagne.nonlinearities.softmax
                                            )
        
    prediction = lasagne.layers.get_output(output_layer)
    loss = lasagne.objectives.categorical_crossentropy(prediction, y)
    l1_regularization = lasagne.regularization.regularize_layer_params(output_layer, lasagne.regularization.l1)
    loss = loss + l1_regularization
    loss = loss.mean()

    params = lasagne.layers.get_all_params(output_layer, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)
    #Used Stochastic Gradient Descent here without momentum
        
    validation_prediction = lasagne.layers.get_output(output_layer, deterministic=True)
    #We disable dropout for validation
    validation_loss = lasagne.objectives.categorical_crossentropy(validation_prediction, y)
    validation_loss = validation_loss.mean()
        
    validation_accuracy = T.mean(T.eq(T.argmax(validation_prediction, axis=1), y), 
                          dtype=theano.config.floatX)
    
    train_mlp = theano.function([X, y], loss, updates=updates)
    validation_mlp = theano.function([X, y], [validation_loss, validation_accuracy])
    #No updates performed in validation_mlp
    


    training_loss_list = []
    test_accuracy_list = []
    for i in range(epochs):
        for batch in batchify(X_tr, y_tr, batch_size):
            X_batch, y_batch = batch
            training_loss_list.append(train_mlp(X_batch, y_batch))
        
        test_accuracy_list = []
        for batch in batchify(X_t, y_t, batch_size):
            X_batch, y_batch = batch
            _, test_accuracy = validation_mlp(X_batch, y_batch)
            test_accuracy_list.append(test_accuracy*100)

    return(training_loss_list, test_accuracy_list)


if __name__ == "__main__":
    training_loss, test_accuracy = mlp_nesterov_l1(X_train, y_train, X_test, y_test, 2, 500)
    print(np.array(training_loss))
    print(np.array(test_accuracy))

