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
            
            hidden_state = h_t
            cell_state = C_t

        final_hidden_state = hidden_state
        final_cell_state = cell_state
        
        return(list_of_outputs, final_hidden_state, final_cell_state)


    # function to build graph with the initial hidden state as a zero tensor
    # and the hidden cell state as the zero tensor
    def build_graph(self):

        tf.reset_default_graph()

        x = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.num_steps])
        y = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.num_steps])

        with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
            embeddings = tf.Variable(tf.random_uniform([self.num_classes, self.state_size], -1.0, 1.0))
            lookup = tf.nn.embedding_lookup(embeddings, x)
            # the shape of 'list_of_input_vectors' is (self.num_steps, self.batch_size, self.state_size)
            # it is a list of length = self.num_steps of tensors of shape (self.batch_size, self.state_size)
            list_of_input_vectors = tf.unstack(lookup, axis=1)

        with tf.variable_scope('init_states', reuse=tf.AUTO_REUSE):
            init_hidden_state = tf.get_variable('h_0', [self.batch_size, self.state_size], initializer=tf.zeros_initializer())
            init_cell_state = tf.get_variable('C_0', [self.batch_size, self.state_size], initializer=tf.zeros_initializer())

        list_of_outputs, final_hidden_state, final_cell_state = self._LSTM_output(list_of_input_vectors, init_hidden_state, init_cell_state)
        output = tf.reshape(tf.concat(list_of_outputs, axis=1), [-1, self.state_size])

        with tf.variable_scope('softmax', reuse=tf.AUTO_REUSE):
            W = tf.get_variable('w', shape=[self.state_size, self.num_classes], dtype=tf.float32)
            b = tf.get_variable('b', shape=[self.num_classes], dtype=tf.float32)
            logits = tf.nn.xw_plus_b(output, W, b)
            logits = tf.reshape(logits, [self.batch_size, self.num_steps, self.num_classes])

            # softmax predictions for text generation
            prediction = tf.nn.softmax(logits)

        # sequence_loss takes logits in the shape of [batch_size, sequence_length, num_decoder_symbols]
        # and targets (here 'y') in the shape of [batch_size, sequence_length]
        # calculates the weighted cross-entropy loss for a sequence of logits
        # Alternative: pick from top 'n' most likely characters
        loss = tf.contrib.seq2seq.sequence_loss(logits, y, weights=tf.ones([self.batch_size, self.num_steps]))

        # train with Adam Optimizer
        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        train_op = optimizer.minimize(loss)

        graph = {
            "x": x,
            "y": y,
            "init_hidden_state": init_hidden_state,
            "init_cell_state": init_cell_state,
            "final_hidden_state": final_hidden_state,
            "final_cell_state" : final_cell_state,
            "loss": loss,
            "prediction": prediction,
            "train_op": train_op,
        }
        return(graph)




    def train_graph(self, graph, num_epochs, save_location, x_train, y_train):

        saver = tf.train.Saver()

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            loss_list = []

            sess.run(init)
            for epoch in range(num_epochs):
                for x_data, y_data in zip(x_train, y_train):
                    feed_dict = {graph["x"]: x_data, graph["y"]: y_data}
                    current_loss, _ = sess.run([graph["loss"], graph["train_op"]], feed_dict=feed_dict)
                    loss_list.append(current_loss)
            saver.save(sess, save_location)
        return(loss_list)


if __name__=="__main__":
    file_url = 'https://raw.githubusercontent.com/jcjohnson/torch-rnn/master/data/tiny-shakespeare.txt'
    file_name = 'tinyshakespeare.txt'

    num_classes, index_to_character, character_to_index, x_train, y_train = generate_char_data(file_url, file_name, batch_size=32, num_steps=200)
    char_lstm = LSTM(state_size=100, num_classes=num_classes, batch_size=32, num_steps=200)
    char_lstm_graph = char_lstm.build_graph()
    loss_list = char_lstm.train_graph(graph=char_lstm_graph, num_epochs=1, save_location='./checkpoints/lstm_shakespeare', x_train=x_train, y_train=y_train)
