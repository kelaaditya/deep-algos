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