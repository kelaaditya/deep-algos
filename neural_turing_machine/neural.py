import tensorflow as tf
import numpy as np

from feedforward_controller import FeedForwardController
from lstm_controller import LSTMController
from memory import Memory



class NTM:
    
    def __init__(self,
                 controller,  # either 'lstm' or 'feedforward'
                 size_input,
                 size_output,
                 num_memory_vectors=256,
                 size_memory_vector=64,
                 num_read_heads=4,
                 num_write_heads=1,
                 size_conv_shift=1,
                 batch_size=1
                ):
        """The Neural Turing Machine class
        
        controller argument choices: {'lstm','feedforward'}
        """
        
        
        self.size_input = size_input
        self.size_output = size_output
        self.num_memory_vectors = num_memory_vectors
        self.size_memory_vector = size_memory_vector
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads
        self.size_conv_shift = size_conv_shift
        self.batch_size = batch_size
        
        if controller == 'lstm':
            self.controller = LSTMController(self.size_input,
                                             self.size_output,
                                             self.size_memory_vector,
                                             self.size_conv_shift,
                                             self.num_read_heads,
                                             self.num_write_heads,
                                             batch_size
                                            )
        elif controller == 'feedforward':
            self.controller = FeedForwardController(self.size_input,
                                                    self.size_output,
                                                    self.size_memory_vector,
                                                    self.size_conv_shift,
                                                    self.num_read_heads,
                                                    self.num_write_heads,
                                                    batch_size
                                                   )

        self.memory = Memory(self.num_memory_vectors,
                             self.size_memory_vector,
                             self.num_read_heads,
                             self.num_write_heads,
                             self.batch_size
                            )
