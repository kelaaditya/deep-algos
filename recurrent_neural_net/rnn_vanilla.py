import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

num_classes = 2
length_step = 10
length_train = 100000
length_echo = 5
batch_size = 10
state_size = 10
learning_rate = 0.001


def generate_data(length_train, length_echo, batch_size, length_step):
    assert length_train % (batch_size * length_step) == 0

    x_train = np.random.randint(0, 2, size=length_train)
    y_train = np.roll(x_train, length_echo)
    y_train[0 : length_echo] = 0

    x_train = np.reshape(x_train, newshape=(-1, batch_size, length_step))
    y_train = np.reshape(y_train, newshape=(-1, batch_size, length_step))

    return(x_train, y_train)


def RNN_cell(state, input_vector):
    with tf.variable_scope('RNN_cell', reuse=tf.AUTO_REUSE):
        W = tf.get_variable('W', [num_classes + state_size, state_size], initializer=tf.random_normal_initializer())
        b = tf.get_variable('b', [state_size], initializer=tf.zeros_initializer())
    concatenation = tf.concat([input_vector, state], axis=1)
    nonlinear_output = tf.tanh(tf.add(tf.matmul(concatenation, W), b))
    return(nonlinear_output)

def RNN_cell_outputs(initial_state, list_of_inputs):
    cell_outputs = []
    state = initial_state
    for input_vector in list_of_inputs:
        state = RNN_cell(state, input_vector)
        cell_outputs.append(state)
    return(cell_outputs)


def RNN_logits(initial_state, list_of_inputs):
    with tf.variable_scope('softmax', reuse=tf.AUTO_REUSE):
        W = tf.get_variable('W', [state_size, num_classes], initializer=tf.random_normal_initializer())
        b = tf.get_variable('b', [num_classes], initializer=tf.zeros_initializer())
                                    
    cell_outputs = RNN_cell_outputs(initial_state, list_of_inputs)
                                            
    logits = [tf.add(tf.matmul(cell_output, W), b) for cell_output in cell_outputs]
                                                        
    return(logits)


def RNN_train(num_epochs):
    x = tf.placeholder(dtype=tf.int32, shape=[batch_size, length_step])
    y = tf.placeholder(dtype=tf.int32, shape=[batch_size, length_step])
    init_state = tf.placeholder(dtype=tf.float32, shape=[batch_size, state_size])

    x_one_hot = tf.one_hot(x, depth=2)
    x_one_hot_stack = tf.unstack(x_one_hot, axis=1)

    logits = RNN_logits(init_state, x_one_hot_stack)
    
    y_unstack = tf.unstack(y, axis=1)
    
    loss = tf.reduce_mean([tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logit) for logit, label in zip(logits, y_unstack)])
    
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        loss_list = []
        for epoch in range(num_epochs):
            print("epoch #: ", epoch)
            x_train, y_train = generate_data(length_train, length_echo, batch_size, length_step)
            training_state = np.zeros((batch_size, state_size))
            for x_data, y_data in zip(x_train, y_train):
                training_loss, _ = sess.run([loss, train_op], feed_dict={x: x_data, y: y_data, init_state: training_state})
                loss_list.append(training_loss)
    return(loss_list)

if __name__=="__main__":
    loss_list = RNN_train(num_epochs=10)
    plt.plot(loss_list)
    plt.show()
