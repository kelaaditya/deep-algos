import abc
import numpy as np
import tensorflow as tf


class Controller(abc.ABC):    
    def __init__(self,
                 size_input,
                 size_output,
                 size_memory_vector,
                 size_conv_shift,
                 num_read_heads,
                 num_write_heads=1,
                 batch_size=1
                ):
        self.size_input = size_input
        self.size_output = size_output
        
        # size_memory_vector is 'M' from the paper
        self.size_memory_vector = size_memory_vector
        
        self.size_conv_shift = size_conv_shift
        
        # number of read heads of controller
        self.num_read_heads = num_read_heads
        
        # the number of write_heads = 1 by default
        self.num_write_heads = num_write_heads
        
        self.batch_size = batch_size
        
        # size of input vector of shape (batch_size, size_input)
        # concatenated with vectors read by controller
        # each memory vector read by controller is of 
        # shape (batch_size, size_memory_vector)
        self.size_concatenated_input = self.size_memory_vector * self.num_read_heads + self.size_input
        
        """
        read_interface_size = size_of_key_vector + 
                              size_of_key_strength (beta_t) +
                              size_of_interpolation_gate (g_t) + 
                              size_of_gamma (gamma_t) + 
                              size_of_conv_shift_vector
        Here:                      
        size_of_conv_shift_vector=5 if size_conv_shift=2,
        i.e., if size_conv_shift=2, 
        then conv_shift_vector = [p_shift(-2), p_shift(-1), p_shift(0), p_shift(1), p_shift(2)]
        and size_conv_shift_vector=2*2+1=5
        """
        size_read_interface = self.size_memory_vector * self.num_read_heads + \
                              1 * self.num_read_heads + \
                              1 * self.num_read_heads + \
                              1 * self.num_read_heads + \
                              (2 * self.size_conv_shift + 1) * self.num_read_heads
        size_write_interface = self.size_memory_vector * self.num_write_heads + \
                               1 * self.num_write_heads + \
                               1 * self.num_write_heads + \
                               1 * self.num_write_heads + \
                               (2 * self.size_conv_shift + 1) * self.num_write_heads
        size_erase_vector = self.size_memory_vector
        size_add_vector = self.size_memory_vector
        self.size_interface_vector = size_read_interface + size_write_interface + size_erase_vector + size_add_vector
        
    
    def parse_interface_vector(self, interface_vector):
        """Parse interface vector into components
        
        Parameters:
        -----------
        interface_vector: tf.Tensor (batch_size, size_interface_vector)
        
        returns: dict
            a dictionary with the parsed components
        --------
        """
        
        read_key_vector = self.size_memory_vector * self.num_read_heads
        read_key_strength = read_key_vector + 1 * self.num_read_heads
        read_interpolation_gate = read_key_strength + 1 * self.num_read_heads
        read_gamma = read_interpolation_gate + 1 * self.num_read_heads
        read_conv_shift_vector = read_gamma + (2 * self.size_conv_shift + 1) * self.num_read_heads
        
        write_key_vector = read_conv_shift_vector + self.size_memory_vector * self.num_write_heads
        write_key_strength = write_key_vector + 1 * self.num_write_heads
        write_interpolation_gate = write_key_strength + 1 * self.num_write_heads
        write_gamma = write_interpolation_gate + 1 * self.num_write_heads
        write_conv_shift_vector = write_gamma + (2 * self.size_conv_shift + 1) * self.num_write_heads
        
        erase_vector = write_conv_shift_vector + self.size_memory_vector
        add_vector = erase_vector + self.size_memory_vector
        
        shape_read_key_vector = (self.batch_size, self.size_memory_vector, self.num_read_heads)
        shape_read_key_strength = (self.batch_size, self.num_read_heads)
        shape_read_interpolation_gate = (self.batch_size, self.num_read_heads)
        shape_read_gamma = (self.batch_size, self.num_read_heads)
        shape_read_conv_shift_vector = (self.batch_size, 2 * self.size_conv_shift + 1, self.num_read_heads)
        
        shape_write_key_vector = (self.batch_size, self.size_memory_vector, self.num_write_heads)
        shape_write_key_strength = (self.batch_size, self.num_write_heads)
        shape_write_interpolation_gate = (self.batch_size, self.num_write_heads)
        shape_write_gamma = (self.batch_size, self.num_write_heads)
        shape_write_conv_shift_vector = (self.batch_size, 2 * self.size_conv_shift + 1, self.num_write_heads)
        
        shape_erase_vector = (self.batch_size, self.size_memory_vector)
        shape_add_vector = (self.batch_size, self.size_memory_vector)
        
        # the parsing begins... 
        parsed = {}
        
        parsed['read_keys'] = tf.reshape(interface_vector[:, :read_key_vector], shape_read_key_vector)
        
        # the key_strength should be >= 0
        # hence, we apply softplus
        parsed['read_strengths'] = tf.nn.softplus(tf.reshape(interface_vector[:, read_key_vector:read_key_strength], shape_read_key_strength))
        
        # the interpolation_gate lies between [0, 1]
        # hence, we apply sigmoid
        parsed['read_gates'] = tf.nn.sigmoid(tf.reshape(interface_vector[:, read_key_strength:read_interpolation_gate], shape_read_interpolation_gate))
        
        # gamma_t >= 1 always
        # hence, we apply (softplus + 1)
        parsed['read_gammas'] = 1 + tf.nn.softplus(tf.reshape(interface_vector[:, read_interpolation_gate:read_gamma], shape_read_gamma))
        
        # conv_shift vector is a vector of probabilities
        # hence, we apply softmax
        parsed['read_shifts'] = tf.nn.softmax(tf.reshape(interface_vector[:, read_gamma:read_conv_shift_vector], shape_read_conv_shift_vector))
        
        # similar shapes and wrapping for the write head 
        parsed['write_keys'] = tf.reshape(interface_vector[:, read_conv_shift_vector:write_key_vector], shape_write_key_vector)
        parsed['write_strengths'] = tf.nn.softplus(tf.reshape(interface_vector[:, write_key_vector:write_key_strength], shape_read_key_strength))
        parsed['write_gates'] = tf.nn.sigmoid(tf.reshape(interface_vector[:, write_key_strength:write_interpolation_gate], shape_write_interpolation_gate))
        parsed['write_gammas'] = 1 + tf.nn.softplus(tf.reshape(interface_vector[:, write_interpolation_gate: write_gamma], shape_write_gamma))
        parsed['write_shifts'] = tf.nn.softmax(tf.reshape(interface_vector[:, write_gamma:write_conv_shift_vector], shape_write_conv_shift_vector))
        
        # each element of erase_vector lies between [0, 1]
        # hence, we apply sigmoid
        parsed['erase_vector'] = tf.nn.sigmoid(tf.reshape(interface_vector[:, write_conv_shift_vector:erase_vector], shape_erase_vector))
        parsed['add_vector'] = tf.reshape(interface_vector[:, erase_vector:add_vector], shape_add_vector)
        
        return parsed
    
        
    @abc.abstractmethod
    def weights_initial(self):
        """Defines the initial weights for the controller
        """
    
    
    @abc.abstractmethod
    def network_variables(self):
        """Defines the variables of the neural network model
        inside the controller
        """
        
    
    @abc.abstractmethod
    def network_operation(self, concatenated_input):
        """Defines the internal operation of the neural 
        network model inside the controller
        """
        
    
    @abc.abstractmethod
    def network_output(self):
        """Pushes through the input_data to get out the
        parsed interface vector and the pre_output
        """
