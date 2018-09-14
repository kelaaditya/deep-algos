import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

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


def train(model, # 'lstm' or 'feedforward'
          learning_rate,
          momentum,
          decay,
          iterations,
          save_location,
          restore_training,
          size_input,
          size_input_sequence,
          size_output,
          num_memory_vectors,
          size_memory_vector,
          num_read_heads,
          num_write_heads,
          size_conv_shift,
          batch_size):

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

    # loss function
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
        
        # check if restore training
        # if True, restore checkpoints
        # and train over the restored
        if restore_training:
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(save_location)
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('restored')
        else:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())

        loss_list = []
        memory_tensor_list = []
        
        for iteration in range(iterations):
            input_data, target_output_data = generate_data(batch_size, size_input_sequence, size_input)
            
            current_loss, _, memory_tensor = sess.run([loss, train_op, ntm.memory_tensor],
                                                      feed_dict={
                                                          ntm.input_sequential_data: input_data,
                                                          ntm.target_output: target_output_data
                                                      })
            
            loss_list.append(current_loss)
            memory_tensor_list.append(memory_tensor)
            
            if iteration % 1000 == 0:
                saver.save(sess, save_location + model, global_step=iteration)

    return loss_list, memory_tensor_list


def test(model, # or 'feedforward'
         save_location,
         size_input,
         size_input_sequence,
         size_output,
         num_memory_vectors,
         size_memory_vector,
         num_read_heads,
         num_write_heads,
         size_conv_shift,
         batch_size):
    
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
    
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(save_location)
    
    with tf.Session() as sess:
        saver.restore(sess, ckpt.model_checkpoint_path)

        input_data, target_output = generate_data(batch_size, size_input_sequence, size_input)
        ntm_output, memory_tensor = ntm.generate_output(ntm.input_sequential_data)
        ntm_output, memory_tensor = sess.run([ntm_output, memory_tensor],
                                             feed_dict={ntm.input_sequential_data: input_data})
        
    return input_data, ntm_output, target_output, memory_tensor


def main():
    parser = argparse.ArgumentParser(description='NTM - Copy Task')
    # mode: training vs testing
    parser.add_argument('--mode',               type=str,   default='train',    help='training vs testing')

    # model: LSTMController vs FeedForwardController
    parser.add_argument('--model',              type=str,   default='lstm',     help='controller: LSTM vs Feedforward')

    # training parameters
    parser.add_argument('--iterations',         type=int,   default=100000,     help='total training iterations')
    parser.add_argument('--restore_training',   type=str,   default=False,      help='restore training from checkpoint')
    parser.add_argument('--learning_rate',      type=float, default=0.001)
    parser.add_argument('--momentum',           type=float, default=0.9)
    parser.add_argument('--decay',              type=float, default=0.95)
    parser.add_argument('--save_location',      type=str,   default='./checkpoints/')

    # model parameters
    parser.add_argument('--size_input',         type=int,   default=10)
    parser.add_argument('--size_input_sequence',type=int,   default=10)
    parser.add_argument('--size_output',        type=int,   default=10)
    parser.add_argument('--num_memory_vectors', type=int,   default=128)
    parser.add_argument('--size_memory_vector', type=int,   default=64)
    parser.add_argument('--num_read_heads',     type=int,   default=1)
    parser.add_argument('--num_write_heads',    type=int,   default=1)
    parser.add_argument('--size_conv_shift',    type=int,   default=1)
    parser.add_argument('--batch_size',         type=int,   default=1)

    args = parser.parse_args()

    if args.mode == 'train':
        loss_list, _ = train(model=args.model,
                             learning_rate=args.learning_rate,
                             momentum=args.momentum,
                             decay=args.decay,
                             iterations=args.iterations,
                             save_location=args.save_location,
                             restore_training=args.restore_training,
                             size_input=args.size_input,
                             size_input_sequence=args.size_input_sequence,
                             size_output=args.size_output,
                             num_memory_vectors=args.num_memory_vectors,
                             size_memory_vector=args.size_memory_vector,
                             num_read_heads=args.num_read_heads,
                             num_write_heads=args.num_write_heads,
                             size_conv_shift=args.size_conv_shift,
                             batch_size=args.batch_size)

    else:
        input_data, ntm_output, target_output, memory_tensor = test(model=args.model,
                                                                    save_location=args.save_location,
                                                                    size_input=args.size_input,
                                                                    size_input_sequence=args.size_input_sequence,
                                                                    size_output=args.size_output,
                                                                    num_memory_vectors=args.num_memory_vectors,
                                                                    size_memory_vector=args.size_memory_vector,
                                                                    num_read_heads=args.num_read_heads,
                                                                    num_write_heads=args.num_write_heads,
                                                                    size_conv_shift=args.size_conv_shift,
                                                                    batch_size=args.batch_size)



if __name__ == "__main__":
    main()
