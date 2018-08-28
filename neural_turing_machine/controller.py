import numpy as np
import tensorflow as tf


class Controller:
    
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
                              (2 * self.size_conv_shift + 1)
        size_write_interface = self.size_memory_vector * self.num_write_heads + \
                               1 * self.num_write_heads + \
                               1 * self.num_write_heads + \
                               1 * self.num_write_heads + \
                               (2 * self.size_conv_shift + 1)
        size_erase_vector = self.size_memory_vector
        size_add_vector = self.size_memory_vector
        self.size_interface_vector = size_read_interface + size_write_interface + size_erase_vector + size_add_vector
        