import numpy as np
import tensorflow as tf

from controller import Controller


class FeedForwardController(Controller):
    
    def variables_for_network(self):
        """Defines the variables of the neural network model
        inside the controller
        """
        
        self.size_network_output = 100
        
        with tf.variable_scope('controller_network_variables', reuse=tf.AUTO_REUSE):
            self.W = tf.get_variable('W', shape=[self.size_concatenated_input,
                                                 self.size_network_output],
                                                 initializer=tf.random_normal_initializer())
            self.b = tf.get_variable('b',
                                     shape=[self.size_network_output],
                                     initializer=tf.zeros_initializer())
            
    
    def variables_for_network_output(self):
        """Defines the initial weights for the controller
        """
        
        with tf.variable_scope('controller_network_weights', reuse=tf.AUTO_REUSE):
            self.interface_weights = tf.get_variable('interface_weights',
                                                     shape=[self.size_network_output, self.size_interface_vector],
                                                     initializer=tf.random_normal_initializer())
            self.network_output_weights = tf.get_variable('network_output_weights',
                                                          shape=[self.size_network_output, self.size_output],
                                                          initializer=tf.random_normal_initializer())
            
            
    def network_operation(self, concatenated_input_vector):
        """Defines the internal operation of the neural 
        network model inside the controller
        """
        
        network_output = tf.add(tf.matmul(concatenated_input_vector, self.W), self.b)
        network_output = tf.nn.relu(network_output)
        
        return network_output
    
    
    def network_output(self, input_data, last_read_vectors):
        """Pushes through the input_data to get out the
        parsed interface vector and the pre_output
        
        Parameters:
        -----------
        input_data: tf.Tensor (self.batch_size, self.size_input_data)
        
        last_read_vectors: tf.Tensor (self.batch_size, self.size_memory_vector, self.num_read_heads)
            The last read vectors from memory
        """
        
        flat_last_read_vectors = tf.reshape(last_read_vectors, shape=[self.batch_size, self.size_memory_vector * self.num_read_heads])
        concatenated_input_vector = tf.concat([input_data, flat_last_read_vectors], axis=1)
        
        network_output = self.network_operation(concatenated_input_vector)
        
        pre_output = tf.matmul(network_output, self.network_output_weights)
        interface_vector = tf.matmul(network_output, self.interface_weights)
        parsed_interface_vector = self.parse_interface_vector(interface_vector)
        
        return pre_output, parsed_interface_vector
