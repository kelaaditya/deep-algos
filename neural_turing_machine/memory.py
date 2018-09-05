import numpy as np
import tensorflow as tf


class Memory:
    def __init__(self,
                 num_memory_vectors=128,
                 size_memory_vector=20,
                 num_read_heads=1,
                 num_write_heads=1,
                 batch_size=1
                ):
        # num_memory_vectors is 'N' from the paper
        self.num_memory_vectors = num_memory_vectors
        
        # size_memory_vector is 'M' from the paper
        self.size_memory_vector = size_memory_vector
        
        # number of read heads of the controller
        self.num_read_heads = num_read_heads
        
        # number of write heads of the controller
        self.num_write_heads = num_write_heads
        
        self.batch_size = batch_size
        
        
    def content_addressing(self, memory, read_and_write_key_vectors, read_and_write_strengths):
        """content-based addressing from memory
        
        num_total_keys == num_read_and_write_heads
        
        Parameters:
        -----------
        
        memory: tf.Tensor 
            shape: (batch_size, num_memory_vectors, size_memory_vector)
            the memory matrix
        read_and_write_key_vectors: tf.Tensor
            shape: (batch_size, num_read_and_write_heads, size_memory_vector)
            concatenated read and write key vectors
        read_and_write_strengths: tf.Tensor
            shape: (batch_size, num_read_and_write_heads)
            tensor of read and write strengths
            
        Returns:
        --------
        tf.Tensor
            shape: (batch_size, num_memory_vectors, num_read_and_write_heads)
        """
        normalized_memory = tf.nn.l2_normalize(memory, axis=2)
        
        transpose_read_and_write_key_vectors = tf.transpose(read_and_write_key_vectors, perm=[0, 2, 1])
        normalized_read_and_write_key_vectors = tf.nn.l2_normalize(transpose_read_and_write_key_vectors, axis=1)
        
        conv_matrix = tf.matmul(normalized_memory, normalized_read_and_write_key_vectors)
        
        read_and_write_strengths = tf.expand_dims(read_and_write_strengths, 1)
        
        # multiply instead of matmul as each strength gets multiplied
        # with each similarity vector to get out a num_total_key
        # number of addresses
        return tf.nn.softmax(conv_matrix * read_and_write_strengths, axis=1)
        
    
    def interpolation(self,
                      read_and_write_interpolation_gates,
                      read_and_write_prev_weights,
                      content_addressing_weights):
        """interpolates content_addressing with previous weightings
        using the interpolation gates
        
        num_total_gates == num_read_and_write_heads == num_total_keys
        
        Parameters:
        -----------
        read_and_write_interpolation_gates: tf.Tensor
            shape: (batch_size, num_read_and_write_heads)
        read_and_write_prev_weights: tf.Tensor
            shape: (batch_size, num_memory_vectors, num_read_and_write_heads)
        content_addressing_weights: tf.Tensor
            shape: (batch_size, num_memory_vectors, num_read_and_write_heads)
        """
        
        read_and_write_interpolation_gates = tf.expand_dims(read_and_write_interpolation_gates, 1)
        
        gated_weightings = read_and_write_interpolation_gates * content_addressing_weights + \
                           (1 - read_and_write_interpolation_gates) * read_and_write_prev_weights
        
        return gated_weightings
        
        
    def convolutional_shift(self, gated_weightings, shift_weightings):
        """focussing by location by applying convolutional shift
        
        Parameters:
        -----------
        gated_weightings: tf.Tensor
            shape: (batch_size, num_memory_vectors, num_read_and_write_heads)
        shift_weightings: tf.Tensor
            shape: (batch_size, 2 * size_conv_shift + 1, num_read_and_write_heads)
        """
        
        concatenated_gated_weightings = tf.concat([gated_weightings, gated_weightings, gated_weightings], axis=1)
        
        #
        single_channel_weightings = tf.expand_dims(tf.expand_dims(concatenated_gated_weightings, axis=1), axis=-1)
        
        # unstacked for getting batch_size number of weightings, 
        # each of size: (batch_size=1, 2 * size_conv_shift + 1, num_read_and_write_heads, channels=1)
        unstacked_weightings = tf.unstack(single_channel_weightings, axis=0)
        
        single_channel_kernels = tf.expand_dims(tf.expand_dims(shift_weightings, axis=-1), axis=-1)
        unstacked_kernels = tf.unstack(single_channel_kernels, axis=0)
        
        conv_shift_array = []
        for i in range(self.batch_size):
            conv = tf.nn.conv2d(input=unstacked_weightings[i],
                                filter=unstacked_kernels[i],
                                strides=[1, 1, 1, 1],
                                padding="SAME"
                               )
            conv = tf.squeeze(conv, axis=[0, -1])
            conv = conv[self.num_memory_vectors: 2 * self.num_memory_vectors]
            conv_shift_array.append(conv)
            
        conv_weights = tf.stack(conv_shift_array, axis=0)
        return conv_weights