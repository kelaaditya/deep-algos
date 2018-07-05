import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


    ###########################
    #         batchify        #
    ###########################

def batchify(X, y, batch_size):
    for i in range(0, len(X)-batch_size+1, batch_size):
        yield(X[i: i+batch_size], y[i: i+batch_size])



    ###########################
    #       load MNIST        #
    ###########################

"""
Created on Tue Oct 25 11:40:10 2016

Partly taken from https://github.com/tqchen/ML-SGHMC
"""

import os, struct
from array import array as pyarray
from numpy import  array, zeros
import numpy as np

def load_mnist(dataset="training", digits=np.arange(10), path="."):
    """
    Loads MNIST files into 3D numpy arrays 
    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows, cols) )
    labels = zeros((N ) )
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels


def load_data():
    X_train, y_train = load_mnist('training')
    X_test, y_test = load_mnist('testing')
    X_train = np.reshape(X_train, (len(X_train), 1, 28, 28))
    y_train = y_train.astype('int32')
    X_test = np.reshape(X_test, (len(X_test), 1, 28, 28))
    y_test = y_test.astype('int32')
    return X_train/np.float32(256), y_train, X_test/np.float32(256), y_test



    ###########################
    #      MLP for MNIST      #
    ###########################

def mlp(batch_size, epochs, learning_rate=0.01):
    X = tf.placeholder(tf.float32, shape=(batch_size, 784), name='X')
    Y = tf.placeholder(tf.float32, shape=(batch_size, 10), name='Y')
    
    #Hidden Layer 1 has 800 hidden units
    hidden_layer_1_weights = tf.Variable(tf.random_normal([784, 800]), name='w_1')
    hidden_layer_1_bias = tf.Variable(tf.random_normal([800]), name='b_1')
    
    #Hidden Layer 2 has 800 hidden units
    hidden_layer_2_weights = tf.Variable(tf.random_normal([800, 800]), name='w_2')
    hidden_layer_2_bias = tf.Variable(tf.random_normal([800]), name='b_2')
    
    #Output Layer has 800 hidden units and 10 output classes
    output_layer_weights = tf.Variable(tf.random_normal([800, 10]), name='w_o')
    output_layer_bias = tf.Variable(tf.random_normal([10]), name='b_o')
    
    #Hidden Layer activated with a RELU
    hidden_layer_1 = tf.add(tf.matmul(X, hidden_layer_1_weights), hidden_layer_1_bias)
    hidden_layer_1 = tf.nn.relu(hidden_layer_1)
    
    #Hidden Layer 2 activated with a RELU
    hidden_layer_2 = tf.add(tf.matmul(hidden_layer_1, hidden_layer_2_weights), hidden_layer_2_bias)
    hidden_layer_2 = tf.nn.relu(hidden_layer_2)
    
    output_layer = tf.add(tf.matmul(hidden_layer_2, output_layer_weights), output_layer_bias)
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        number_batches = int(mnist.train.num_examples/batch_size)
        training_loss_list = []
        for epoch in range(epochs):
            for i in range(number_batches):
                X_batch, Y_batch = mnist.train.next_batch(batch_size)
                _, training_loss_value = sess.run([optimizer, loss], feed_dict={X: X_batch, Y: Y_batch})
                training_loss_list.append(training_loss_value)

    return(training_loss_list)

if __name__ == "__main__":
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    
    training_loss = mlp(batch_size=50, 
                        epochs=100,
                        learning_rate=0.01)

    plt.plot(np.arange(len(training_loss)), training_loss)
    plt.show()
    
