import numpy as np
import tensorflow as tf

from controller import Controller


class LSTMController(Controller):
        
    def variables_for_network(self):
        """Defines the variables of the neural network model
        inside the controller
        """
        
        self.size_lstm_units = 100
        self.size_network_output = self.size_lstm_units
        
        self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.size_lstm_units)
        self.init_cell_state = tf.zeros(shape=[self.batch_size, self.size_lstm_units])
        self.init_hidden_state = tf.zeros(shape=[self.batch_size, self.size_lstm_units])
        self.init_state = tf.contrib.rnn.LSTMStateTuple(self.init_cell_state, self.init_hidden_state)

            
    def variables_for_network_output(self):
        """Defines the initial weights for the controller
        """
        
        # lambda for std dev bounding
        self.std_dev = lambda length: np.min([1e-2, np.sqrt(2.0 / length)])


        with tf.variable_scope('controller_network_weights', reuse=tf.AUTO_REUSE):
            self.interface_weights = tf.get_variable('interface_weights',
                                                     initializer=tf.truncated_normal(
                                                         shape=[self.size_network_output, self.size_interface_vector],
                                                         stddev=self.std_dev(self.size_network_output)
                                                     )
                                                    )
            self.network_output_weights = tf.get_variable('network_output_weights',
                                                          initializer=tf.truncated_normal(
                                                              shape=[self.size_network_output, self.size_output],
                                                              stddev=self.std_dev(self.size_network_output)
                                                          )
                                                         )
            
            
    def network_operation(self, concatenated_input_vector, state):
        """Defines the internal operation of the neural 
        network model inside the controller
        """
        
        return self.lstm_cell(concatenated_input_vector, state)
    
    
    def network_output(self, input_data, last_read_vectors, state=None):
        """Pushes through the input_data to get out the
        parsed interface vector and the pre_output
        
        Parameters:
        -----------
        input_data: tf.Tensor (self.batch_size, self.size_input)
        
        last_read_vectors: tf.Tensor (self.batch_size, self.size_memory_vector, self.num_read_heads)
            The last read vectors from memory
        """
        
        flat_last_read_vectors = tf.reshape(last_read_vectors, shape=[self.batch_size, self.size_memory_vector * self.num_read_heads])
        concatenated_input_vector = tf.concat([input_data, flat_last_read_vectors], axis=1)
        
        network_output, network_state = self.network_operation(concatenated_input_vector, state)
        
        pre_output = tf.matmul(network_output, self.network_output_weights)
        interface_vector = tf.matmul(network_output, self.interface_weights)
        parsed_interface_vector = self.parse_interface_vector(interface_vector)
        
        return pre_output, parsed_interface_vector, network_state
        