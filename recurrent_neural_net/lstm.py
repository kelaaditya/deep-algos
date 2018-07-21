import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request



def generate_char_data(file_url, file_name, batch_size, num_steps):
    if not os.path.exists(file_name):
        urllib.request.urlretrieve(file_url, file_name)

    with open(file_name, 'r') as f:
        raw_data = f.read()

    characters = set(raw_data)
    index_to_character = dict(enumerate(characters))
    character_to_index = dict(zip(index_to_character.values(), index_to_character.keys()))
    x_data = [character_to_index[char] for char in raw_data]

    # gives the number of characters in the vocabulary set
    num_classes= len(characters)

    # !! y_data is the x_data shifted by one position to the left
    y_data = np.roll(x_data, shift=-1)

    length_data = np.size(x_data)
    length_batch = length_data // batch_size

    # total sets = total number of data sets with sizes (batch_size, num_steps) each
    total_sets = length_batch // num_steps

    x_train = np.reshape(x_data[0 : total_sets * batch_size * num_steps], newshape=(total_sets, batch_size, num_steps))
    y_train = np.reshape(y_data[0 : total_sets * batch_size * num_steps], newshape=(total_sets, batch_size, num_steps))

    return(num_classes, index_to_character, character_to_index, x_train, y_train)

class LSTM:
    
    def __init__(self, state_size, num_classes, batch_size, num_steps, learning_rate=0.001):
        self.state_size = state_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.learning_rate = learning_rate

    def _forget_gate(self, hidden_state, input_vector):
        with tf.variable_scope('f_gate', reuse=tf.AUTO_REUSE):
            concatenation = tf.concat([hidden_state, input_vector], axis=1)
            
            # forget gate layer variable initialization
            self.W_f = tf.get_variable('W_f', [self.state_size + self.state_size, self.state_size], initializer=tf.random_normal_initializer())
            self.b_f = tf.get_variable('b_f', [self.state_size], initializer=tf.zeros_initializer())
            f_t = tf.sigmoid(tf.nn.xw_plus_b(concatenation, self.W_f, self.b_f))
        return(f_t)
    
    
    def _input_gate(self, hidden_state, input_vector):
        with tf.variable_scope('i_gate', reuse=tf.AUTO_REUSE):
            concatenation = tf.concat([hidden_state, input_vector], axis=1)
            
            # input gate layer variable initialization
            self.W_i = tf.get_variable('W_i', [self.state_size + self.state_size, self.state_size], initializer=tf.random_normal_initializer())
            self.b_i = tf.get_variable('b_i', [self.state_size], initializer=tf.zeros_initializer())
            i_t = tf.sigmoid(tf.nn.xw_plus_b(concatenation, self.W_i, self.b_i))
            
            self.W_c = tf.get_variable('W_c', [self.state_size + self.state_size, self.state_size], initializer=tf.random_normal_initializer())
            self.b_c = tf.get_variable('b_c', [self.state_size], initializer=tf.random_normal_initializer())
            ctilde_t = tf.tanh(tf.nn.xw_plus_b(concatenation, self.W_c, self.b_c))
        return(i_t, ctilde_t)
    
    
    def _cell_state_update(self, cell_state, f_t, i_t, ctilde_t):
        C_t = tf.add(tf.multiply(f_t, cell_state), tf.multiply(i_t, ctilde_t))
        return(C_t)
    
    
    def _output_gate(self, hidden_state, input_vector, C_t):
        with tf.variable_scope('o_gate', reuse=tf.AUTO_REUSE):
            concatenation = tf.concat([hidden_state, input_vector], axis=1)
            
            # output layer variable initialization
            self.W_o = tf.get_variable('W_o', [self.state_size + self.state_size, self.state_size], initializer=tf.random_normal_initializer())
            self.b_o = tf.get_variable('b_o', [self.state_size])
            o_t = tf.sigmoid(tf.nn.xw_plus_b(concatenation, self.W_o, self.b_o))
            
            
            h_t = tf.multiply(o_t, tf.tanh(C_t))
        return(o_t, h_t)
    
    def _LSTM_output(self, list_of_input_vectors, init_hidden_state, init_cell_state):
        list_of_outputs = []
        hidden_state = init_hidden_state
        cell_state = init_cell_state
        
        for input_vector in list_of_input_vectors:
            f_t = self._forget_gate(hidden_state, input_vector)
            i_t, ctilde_t = self._input_gate(hidden_state, input_vector)
            C_t = self._cell_state_update(cell_state, f_t, i_t, ctilde_t)
            o_t, h_t = self._output_gate(hidden_state, input_vector, C_t)
            
            list_of_outputs.append(o_t)
            
            cell_state = C_t
            hidden_state = h_t
        
        return(list_of_outputs)
