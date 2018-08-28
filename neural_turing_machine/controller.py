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