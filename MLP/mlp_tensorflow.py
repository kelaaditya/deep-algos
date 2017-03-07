import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from load_mnist import *
from batchify import *

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
                        epochs=1,
                        learning_rate=0.01)
    
