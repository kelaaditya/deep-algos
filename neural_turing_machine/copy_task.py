import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from feedforward_controller import FeedForwardController
from lstm_controller import LSTMController
from memory import Memory
from neural import NTM



def generate_data(batch_size, size_input_sequence, size_input):
    """Generate input_sequential_data and the target_output
    
    Parameters:
    -----------
    size: 3-tuple
        shape in the format: (batch_size, size_input_sequence, input_size)
    """
    size_copy_sequence = int(size_input_sequence / 2)
    
    input_sequential_data = np.zeros(shape=(batch_size, size_input_sequence, size_input))
    input_sequential_data[:, 0, 0] = 1  # delimiter at the start of the sequential input 
    input_sequential_data[:, size_copy_sequence, -1] = 1  # delimiter at the end of the sequential input

    target_output = np.zeros(shape=(batch_size, size_input_sequence, size_input))

    for index in range(size_copy_sequence - 1):
        binomial = np.random.binomial(1, 0.5, size=(batch_size, size_input - 2))
        input_sequential_data[:, index + 1, 1:-1] = binomial
        target_output[:, size_copy_sequence + (index + 1), 1:-1] = binomial
    
    return input_sequential_data, target_output


def train(model='lstm', # or 'feedforward'
          learning_rate=0.001,
          momentum=0.9,
          decay=0.95,
          iterations=100000,
          save_location='./checkpoints/ntm/',
          restore_training=False):
    
    tf.reset_default_graph()
    
    ntm = NTM(model,
              size_input,
              size_input_sequence,
              size_output,
              num_memory_vectors,
              size_memory_vector,
              num_read_heads,
              num_write_heads,
              size_conv_shift,
              batch_size
             )
    
    # ntm.output contains predictions by the NTM
    # and has shape: (batch_size, size_input_sequence, size_input)
    ntm_output = ntm.output

    eps = 1e-8
    loss = -1 * tf.reduce_mean(
        ntm.target_output * tf.log(ntm.output + eps) + (1 - ntm.target_output) * tf.log(1 - ntm.output + eps)
    )

    with tf.variable_scope('optimizer', reuse=tf.AUTO_REUSE):
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=momentum, decay=decay)
        gvs = optimizer.compute_gradients(loss)
        capped_gvs = [(tf.clip_by_value(grad, -10.0, 10.0), var) for grad, var in gvs]

        train_op = optimizer.apply_gradients(capped_gvs)

    with tf.Session() as sess:
        if restore_training:
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(save_location)
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('restored')
        else:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())

        loss_list = []
        for iteration in range(iterations):
            input_data, target_output_data = generate_data(batch_size, size_input_sequence, size_input)
            current_loss, _ = sess.run([loss, train_op], feed_dict={ntm.input_sequential_data: input_data, ntm.target_output: target_output_data})

            loss_list.append(current_loss)
            
            if iteration % 100 == 0:
                saver.save(sess, save_location + model, global_step=iteration)
                
    return loss_list