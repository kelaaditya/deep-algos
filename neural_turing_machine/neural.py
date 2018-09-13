import tensorflow as tf
import numpy as np

from feedforward_controller import FeedForwardController
from lstm_controller import LSTMController
from memory import Memory



class NTM:
    
    def __init__(self,
                 controller,  # either 'lstm' or 'feedforward'
                 size_input,
                 size_input_sequence,
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
        self.size_input_sequence = size_input_sequence
        self.size_output = size_output
        self.num_memory_vectors = num_memory_vectors
        self.size_memory_vector = size_memory_vector
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads
        self.size_conv_shift = size_conv_shift
        self.batch_size = batch_size
        
        if controller == 'lstm':
            self.controller_network = 'lstm'
            self.controller = LSTMController(self.size_input,
                                             self.size_output,
                                             self.size_memory_vector,
                                             self.size_conv_shift,
                                             self.num_read_heads,
                                             self.num_write_heads,
                                             batch_size
                                            )
        elif controller == 'feedforward':
            self.controller_network = 'feedforward'
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
        
        # builds graph and concatenates all the pre-outputs
        # from the sequence of inputs into a tensor of shape
        # (batch_size, size_output * size_sequence)
        self.build_graph()
        
        
    def operation(self, input_data, memory_state, controller_state=None):
        """single pass through input to output
        
        Parameters:
        -----------
        memory_state: 4-tuple of tf.Tensors
        
        controller_state: if controller_network == 'lstm':
                                controller_state is 2-tuple
                          else:
                                controller_state is None
                                
        Returns:
        --------
        updated_memory_state: 4-tuple of tf.Tensors
        pre_output: tf.Tensor
            shape: (batch_size, size_output)
        prased_interface_vector: dict
        network_state: LSTMTuple if controller_state else -1
        """
        
        last_read_vectors = memory_state[3]
        
        if controller_state:
            pre_output, parsed_interface_vector, network_state = self.controller.network_output(input_data,
                                                                                               last_read_vectors,
                                                                                               controller_state)
        else:
            pre_output, parsed_interface_vector = self.controller.network_output(input_data,
                                                                                 last_read_vectors)
            
        updated_write_weightings, updated_memory = self.memory.write_operation(
            memory_state[0],
            memory_state[1],
            parsed_interface_vector['write_keys'],
            parsed_interface_vector['write_strengths'],
            parsed_interface_vector['write_gates'],
            parsed_interface_vector['write_shifts'],
            parsed_interface_vector['write_gammas'],
            parsed_interface_vector['add_vector'],
            parsed_interface_vector['erase_vector']
        )
        
        updated_read_weightings, updated_read_vectors = self.memory.read_operation(
            memory_state[0],
            memory_state[2],
            parsed_interface_vector['read_keys'],
            parsed_interface_vector['read_strengths'],
            parsed_interface_vector['read_gates'],
            parsed_interface_vector['read_shifts'],
            parsed_interface_vector['read_gammas']
        )
        
        updated_memory_state = (updated_memory,
                                updated_write_weightings,
                                updated_read_weightings,
                                updated_read_vectors)
        
        return [
            updated_memory_state,
            pre_output,
            network_state if controller_state else None
        ]
    
    
    def generate_output(self, input_sequential_data):
        """Generates output
        
        Parameters:
        -----------
        input_sequential_data: tf.Tensor
            shape: (batch_size, sequence_length, size_input)
        """
        
        # initialize the memory state
        memory_state = self.memory.initialize()
        
        # initialize the controller state
        if self.controller_network == 'lstm':
            controller_state = self.controller.init_state
        elif self.controller_network == 'feedforward':
            controller_state = None
        
        output_list = []
        for index in range(self.size_input_sequence):
            operation_list = self.operation(input_sequential_data[:, index, :], memory_state, controller_state)
            
            memory_state            = operation_list[0]
            controller_state        = operation_list[2]
            pre_output              = operation_list[1]
            
            output_list.append(pre_output)
            
        output = tf.sigmoid(tf.transpose(output_list, perm=[1, 0, 2]))
        return output, memory_state[0]
    
    
    def build_graph(self):
        """Builds the NTM graph
        """
        
        self.input_sequential_data = tf.placeholder(dtype=tf.float32,
                                                    shape=[self.batch_size, self.size_input_sequence, self.size_input],
                                                    name='sequential_input')
        self.target_output = tf.placeholder(dtype=tf.float32,
                                            shape=[self.batch_size, self.size_input_sequence, self.size_input],
                                            name='target_output')
        
        self.output, self.memory_tensor = self.generate_output(self.input_sequential_data)