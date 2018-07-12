import numpy as np
import tensorflow as tf

num_classes = 2
length_step = 25
hidden_size = 100
length_train = 10000
length_echo = 5
batch_size = 1
state_size = 100

def generate_data(length_train, length_echo, batch_size):
    assert length_train % batch_size == 0

    x_train = np.random.randint(0, 2, size=length_train)
    y_train = np.roll(x_train, length_echo)
    y_train[0 : length_echo] = 0

    x_train = np.reshape(x_train, newshape=(batch_size, -1))
    y_train = np.reshape(y_train, newshape=(batch_size, -1))

    return(x_train, y_train)

x = tf.placeholder(dtype=tf.int64, shape=[batch_size, length_step])
y = tf.placeholder(dtype=tf.int64, shape=[batch_size, length_step])
init_state = tf.placeholder(dtype=tf.float32, shape=[batch_size, state_size])

x_one_hot = tf.one_hot(x, num_classes)
x_one_hot_stack = tf.unstack(x_one_hot, axis=1)

def RNN_cell(rnn_input, state):
    with tf.variable_scope('RNN_cell', reuse=tf.AUTO_REUSE):
        W = tf.get_variable('W', shape=[state_size + num_classes, state_size], initializer=tf.random_normal_initializer())
        b = tf.get_variable('b', shape=[state_size], initializer=tf.constant_initializer(0.0))
    concatenation = tf.concat([rnn_input, state], 1)
    output = tf.add(tf.matmul(concatenation, W), b)
    output_nonlinear = tf.tanh(output)
    return(output_nonlinear)

def RNN_cell_outputs(init_state,
    cell_outputs = []
    state = init_state
    for input_vector in x_one_hot_stack:
        state = RNN_cell(input_vector, state)
        cell_outputs.append(state)

with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [state_size, num_classes], initializer=tf.random_normal_initializer())
    b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
logits = [tf.add(tf.matmul(cell_output, W), b) for cell_output in cell_outputs]
predictions = [tf.nn.softmax(logit) for logit in logits]

x_train, y_train = generate_data(length_train, length_echo, batch_size)
x_train = np.reshape(x_train, newshape=(length_train//length_step, batch_size, length_step))
x_train
