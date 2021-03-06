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
            shape: (batch_size, size_memory_vector, num_read_and_write_heads)
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
        
        normalized_read_and_write_key_vectors = tf.nn.l2_normalize(read_and_write_key_vectors, axis=1)
        
        conv_matrix = tf.matmul(normalized_memory, normalized_read_and_write_key_vectors)
        
        read_and_write_strengths = tf.expand_dims(read_and_write_strengths, 1)
        
        # multiply instead of matmul as each strength gets multiplied
        # with each similarity vector to get out a num_total_key
        # number of addresses
        return tf.nn.softmax(conv_matrix * read_and_write_strengths, axis=1)
        
    
    def interpolation(self,
                      read_and_write_prev_weights,
                      content_addressing_weights,
                      read_and_write_interpolation_gates,
                     ):
        """interpolates content_addressing with previous weightings
        using the interpolation gates
        
        num_total_gates == num_read_and_write_heads == num_total_keys
        
        Parameters:
        -----------
        read_and_write_prev_weights: tf.Tensor
            shape: (batch_size, num_memory_vectors, num_read_and_write_heads)
        content_addressing_weights: tf.Tensor
            shape: (batch_size, num_memory_vectors, num_read_and_write_heads)
        read_and_write_interpolation_gates: tf.Tensor
            shape: (batch_size, num_read_and_write_heads)
        """
        
        read_and_write_interpolation_gates = tf.expand_dims(read_and_write_interpolation_gates, 1)
        
        gated_weightings = read_and_write_interpolation_gates * content_addressing_weights + \
                           (1 - read_and_write_interpolation_gates) * read_and_write_prev_weights
        
        return gated_weightings
        
        
    def convolutional_shift(self, gated_weightings, shift_weightings, read_and_write_gammas):
        """focussing by location by applying convolutional shift
        
        Parameters:
        -----------
        gated_weightings: tf.Tensor
            shape: (batch_size, num_memory_vectors, num_read_and_write_heads)
        shift_weightings: tf.Tensor
            shape: (batch_size, 2 * size_conv_shift + 1, num_read_and_write_heads)
        read_and_write_gammas: tf.Tensor
            shape: (batch_size, num_read_and_write_heads)
        """
        
        convoluted_weights = tf.zeros_like(gated_weightings)
        
        size_conv_shift = int(shift_weightings.shape[1])
        
        for i in range(size_conv_shift):
            shift_index = size_conv_shift // 2 - i
            convoluted_weights += tf.manip.roll(gated_weightings, shift=shift_index, axis=1) * shift_weightings[:, i, :]
                
        read_and_write_gammas = tf.expand_dims(read_and_write_gammas, axis=1)
        
        pow_conv_weights = tf.pow(convoluted_weights, read_and_write_gammas)
        
        sharp_conv_weights = pow_conv_weights / tf.expand_dims(tf.reduce_sum(pow_conv_weights, 1), 1)
        
        return sharp_conv_weights
        
    
    def update_memory(self, memory, write_weightings, add_vector, erase_vector):
        """Updates memory according to
        M'_t(i) = M_{t-1}(i)[1 - w_t(i)e_t] and 
        M_t(i) = M'_t(i) + w_t(i)a_t
        
        Parameters:
        -----------
        memory: tf.Tensor
            shape: (batch_size, num_memory_vectors, size_memory_vector)
        write_weightings: tf.Tensor
            shape: (batch_size, num_memory_vectors, num_write_heads)
        add_vector: tf.Tensor
            shape: (batch_size, size_memory_vector)
        erase_vector: tf.Tensor
            shape: (batch_size, size_memory_vector)
        """
        
        # elementwise multiplication
        expand_add_vector = tf.expand_dims(add_vector, 1)
        expand_erase_vector = tf.expand_dims(erase_vector, 1)

        memory = memory * (1 - tf.matmul(write_weightings, expand_erase_vector)) + \
                        tf.matmul(write_weightings, expand_add_vector)
        
        return memory
    
        
    def update_read_vectors(self, memory, read_weightings):
        """Updates read vectors according to:
        r_t = \sum_i w_t(i) * M_t(i)
        
        Parameters:
        -----------
        memory: tf.Tensor
            shape: (batch_size, num_memory_vectors, size_memory_vector)
        read_weightings: tf.Tensor
            shape: (batch_size, num_memory_vectors, num_read_reads)
        """
        
        transpose_read_weigthings = tf.transpose(read_weightings, perm=[0, 2, 1])
        read_vectors = tf.matmul(transpose_read_weigthings, memory)
        transpose_read_vectors = tf.transpose(read_vectors, perm=[0, 2, 1])
        
        return transpose_read_vectors
        
    
    def read_operation(self,
                       memory,
                       read_weightings,
                       read_keys,
                       read_strengths,
                       read_gates,
                       read_shift_weightings,
                       read_gammas):
        """Performs a read-from_memory operation
        
        Parameters:
        -----------
        memory: tf.Tensor
            shape: (batch_size, num_memory_vectors, size_memory_vector)
        read_weightings: tf.Tensor
            shape: (batch_size, num_memory_vectors, num_read_heads)
        read_keys: tf.Tensor
            shape: (batch_size, size_memory_vector, num_read_heads)
        read_strengths: tf.Tensor
            shape: (batch_size, num_read_heads)
        read_gates: tf.Tensor
            shape: (batch_size, num_read_heads)
        read_shift_weightings: tf.Tensor
            shape: (batch_size, 2 * size_conv_shift + 1, num_read_heads)
        read_gammas: tf.Tensor
            shape: (batch_size, num_read_heads)
            
        Returns:
        --------
        (updated_read_weightings, updated_read_vectors)
        
        updated_read_weightings:
            shape: (batch_size, num_memory_vectors, num_read_heads)
        updated_read_vectors: 
            shape: (batch_size, size_memory_vectors, num_read_heads)
        """
        
        content_addressed_weightings = self.content_addressing(memory, read_keys, read_strengths)
        gated_weightings = self.interpolation(read_weightings, content_addressed_weightings, read_gates)
        updated_read_weightings = self.convolutional_shift(gated_weightings, read_shift_weightings, read_gammas)
        updated_read_vectors = self.update_read_vectors(memory, updated_read_weightings)
        
        return updated_read_weightings, updated_read_vectors
    
    
    def write_operation(self,
                        memory,
                        write_weightings,
                        write_keys,
                        write_strengths,
                        write_gates,
                        write_shift_weightings,
                        write_gammas,
                        add_vector,
                        erase_vector):
        """Performs a write-to-memory operation
        
        Parameters:
        -----------
        memory: tf.Tensor
            shape: (batch_size, num_memory_vectors, size_memory_vector)
        write_weightings: tf.Tensor
            shape: (batch_size, num_memory_vectors, num_write_heads)
        write_keys: tf.Tensor
            shape: (batch_size, size_memory_vector, num_write_heads)
        write_strengths: tf.Tensor
            shape: (batch_size, num_write_heads)
        write_gates: tf.Tensor
            shape: (batch_size, num_write_heads)
        write_shift_weightings: tf.Tensor
            shape: (batch_size, 2 * size_conv_shift + 1, num_write_heads)
        write_gammas: tf.Tensor
            shape: (batch_size, num_write_heads)
        add_vector: tf.Tensor
            shape: (batch_size, size_memory_vector)
        erase_vector: tf.Tensor
            shape: (batch_size, size_memory_vector)
            
        Returns:
        --------
        (updated_write_weightings, updated_memory)
        
        updated_write_weightings:
            shape: (batch_size, num_memory_vectors, num_write_heads)
        updated_memory: 
            shape: (batch_size, num_memory_vectors, size_memory_vector)
        """

        content_addressed_weightings = self.content_addressing(memory, write_keys, write_strengths)
        gated_weightings = self.interpolation(write_weightings, content_addressed_weightings, write_gates)
        updated_write_weightings = self.convolutional_shift(gated_weightings, write_shift_weightings, write_gammas)
        updated_memory = self.update_memory(memory, updated_write_weightings, add_vector, erase_vector)
        
        return updated_write_weightings, updated_memory
    
    
    def initialize(self):
        """Initialize the memory matrix
        
        Returns:
        --------
        4-tuple
        memory_matrix[0]: actual memory matrix
            shape: (batch_size, num_memory_vectors, size_memory_vector)
        memory_matrix[1]: write_weightings
            shape: (batch_size, num_memory_vectors, num_write_heads)
        memory_matrix[2]: read_weightings
            shape: (batch_size, num_memory_vectors, num_read_heads)
        memory_matrix[3]: last_read_vector
            shape: (batch_size, size_memory_vector, num_read_heads)
        """
        
        memory_matrix = (
            tf.truncated_normal(shape=[self.batch_size, self.num_memory_vectors, self.size_memory_vector],
                                mean=0.4,
                                stddev=0.1),
            tf.fill(dims=[self.batch_size, self.num_memory_vectors, self.num_write_heads], value=1e-6),
            tf.truncated_normal(shape=[self.batch_size, self.num_memory_vectors, self.num_read_heads],
                                mean=0.4,
                                stddev=0.1),
            tf.fill(dims=[self.batch_size, self.size_memory_vector, self.num_read_heads], value=1e-6)
        )
        return memory_matrix