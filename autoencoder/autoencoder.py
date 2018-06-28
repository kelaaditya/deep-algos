import tensorflow as tf
import numpy as np


class Autoencoder:
    def __init__(self, input_dimension, hidden_dimension):

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, input_dimension])

        with tf.name_scope('encode'):
            weight = tf.Variable(tf.random_normal(shape=[input_dimension, hidden_dimension], dtype=tf.float32), name='weight')
            bias = tf.Variable(tf.zeros(shape=[hidden_dimension]), name='bias')
            self.encode = tf.nn.tanh(tf.add(tf.matmul(self.x, weight), bias))

        with tf.name_scope('decode'):
            weight = tf.Variable(tf.random_normal(shape=[hidden_dimension, input_dimension], dtype=tf.float32), name='weight')
            bias = tf.Variable(tf.zeros(shape=[input_dimension]), name='bias')
            self.decode = tf.add(tf.matmul(self.encode, weight), bias)

            self.saver = tf.train.Saver()


    def train(self, data, learning_rate, epochs):
        self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.x, self.decode))))

        self.train = tf.train.RMSPropOptimizer(learning_rate).minimize(self.loss)

        number_samples = len(data)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(epochs):
                for j in range(number_samples):
                    current_loss, _ = sess.run([self.loss, self.train], feed_dict={self.x: [data[j]]})
                if i % 20 == 0:
                    print('Current training epoch: {0}'.format(i))
            self.saver.save(sess, './autoencoder.ckpt')

    def test(self, data):
        with tf.Session() as sess:
            self.saver.restore(sess, './autoencoder.ckpt')
            decoded_input = sess.run([self.decode], feed_dict={self.x: data})
        return(decoded_input)





