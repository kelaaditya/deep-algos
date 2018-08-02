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





def generate_characters(graph, num_classes, load_checkpoint, start_letter="T", num_output_characters=200):
    saver = tf.train.Saver()
            
    current_char = character_to_index[start_letter]
    current_hidden_state = None
    char_list = [current_char]

    with tf.Session() as sess:
        saver.restore(sess, load_checkpoint)
       
        # start generating "num_output_characters" number of characters:
        for i in range(num_output_characters):
            if current_hidden_state is not None:
                feed_dict = {graph['x'] : [[current_char]], graph['init_hidden_state'] : current_hidden_state}
            else:
                feed_dict = {graph['x'] : [[current_char]]}
                                
            prediction, current_hidden_state = sess.run([graph['prediction'], graph['final_hidden_state']], feed_dict)
                                                                                                                        
            # We get the probability distribution of the predicted character via the
            # softmax calculation over logits. We use this probability distribution to 
            # select the next character by sampling over the distribution
            current_char = np.random.choice(num_classes, 1, p=np.squeeze(prediction))[0]
                                                                                                                                                                                    
            char_list.append(current_char)
                                                                                                                                                                                                        
    characters = map(lambda x: index_to_character[x], char_list)
    string_characters = "".join(characters)
    print(string_characters)
    
    return(string_characters)




class GRU:
    
    def __init__(self, state_size, num_classes, batch_size, num_steps, learning_rate=0.001):
        self.state_size = state_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        
    def _reset_gate(self, hidden_state, input_vector):
        with tf.variable_scope('reset_gate', reuse=tf.AUTO_REUSE):
            self.W_r = tf.get_variable('W_r', [self.state_size, self.state_size], initializer=tf.random_normal_initializer())
            self.U_r = tf.get_variable('U_r', [self.state_size, self.state_size], initializer=tf.random_normal_initializer())
            self.b_r = tf.get_variable('b_r', [self.state_size], initializer=tf.zeros_initializer())
            r_t = tf.sigmoid(tf.matmul(input_vector, self.W_r) + tf.matmul(hidden_state, self.U_r) + self.b_r)
        return(r_t)
    
    def _update_gate(self, hidden_state, input_vector):
        with tf.variable_scope('update_gate', reuse=tf.AUTO_REUSE):
            self.W_z = tf.get_variable('W_z', [self.state_size, self.state_size], initializer=tf.random_normal_initializer())
            self.U_z = tf.get_variable('U_z', [self.state_size, self.state_size], initializer=tf.random_normal_initializer())
            self.b_z = tf.get_variable('b_z', [self.state_size], initializer=tf.zeros_initializer())
            z_t = tf.sigmoid(tf.matmul(input_vector, self.W_z) + tf.matmul(hidden_state, self.U_z) + self.b_z)
        return(z_t)
    
    def _hidden_tilde(self, hidden_state, r_t, input_vector):
        with tf.variable_scope('hidden_tilde', reuse=tf.AUTO_REUSE):
            self.W_h = tf.get_variable('W_h', [self.state_size, self.state_size], initializer=tf.random_normal_initializer())
            self.U_h = tf.get_variable('U_h', [self.state_size, self.state_size], initializer=tf.random_normal_initializer())
            self.b_h = tf.get_variable('b_h', [self.state_size], initializer=tf.zeros_initializer())
            h_tilde_t = tf.tanh(tf.matmul(input_vector, self.W_h) + tf.matmul(tf.multiply(r_t, hidden_state), self.U_h) + self.b_h)
        return(h_tilde_t)
    
    def _hidden_update(self, hidden_state, z_t, h_tilde_t):
        h_t = tf.multiply(z_t, hidden_state) + tf.multiply((1 - z_t), h_tilde_t)
        return(h_t)
    
    def _output(self, h_t):
        with tf.variable_scope('output', reuse=tf.AUTO_REUSE):
            self.W_o = tf.get_variable('W_o', [self.state_size, self.state_size], initializer=tf.random_normal_initializer())
            self.b_o = tf.get_variable('b_o', [self.state_size], initializer=tf.random_normal_initializer())
            o_t = tf.sigmoid(tf.matmul(h_t, self.W_o) + self.b_o)
        return(o_t)
    
    def gru_output(self, list_of_input_vectors, init_hidden_state):
        list_of_outputs = []
        hidden_state = init_hidden_state
        
        for input_vector in list_of_input_vectors:
            r_t = self._reset_gate(hidden_state, input_vector)
            z_t = self._update_gate(hidden_state, input_vector)
            h_tilde_t = self._hidden_tilde(hidden_state, r_t, input_vector)
            h_t = self._hidden_update(hidden_state, z_t, h_tilde_t)
            o_t = self._output(h_t)
            
            list_of_outputs.append(o_t)
            
            hidden_state = h_t
            
        final_hidden_state = hidden_state
        
        return(list_of_outputs, final_hidden_state)
            
     
    def build_graph(self):
        tf.reset_default_graph()
        
        x = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.num_steps])
        y = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.num_steps])
        
        with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
            embeddings = tf.Variable(tf.random_uniform([self.num_classes, self.state_size], -1.0, 1.0))
            lookup = tf.nn.embedding_lookup(embeddings, x)
            list_of_input_vectors = tf.unstack(lookup, axis=1)
            
        init_hidden_state = tf.zeros(shape=[self.batch_size, self.state_size], dtype=tf.float32, name='h_0')
        
        list_of_outputs, final_hidden_state = self.gru_output(list_of_input_vectors, init_hidden_state)
        output = tf.reshape(tf.concat(list_of_outputs, axis=1), [-1, self.state_size])
        
        with tf.variable_scope('softmax', reuse=tf.AUTO_REUSE):
            W = tf.get_variable('W', shape=[self.state_size, self.num_classes], dtype=tf.float32)
            b = tf.get_variable('b', shape=[self.num_classes], dtype=tf.float32)
            logits = tf.nn.xw_plus_b(output, W, b)
            logits = tf.reshape(logits, [self.batch_size, self.num_steps, self.num_classes])
        
        prediction = tf.nn.softmax(logits)
        
        loss = tf.contrib.seq2seq.sequence_loss(logits, y, weights=tf.ones([self.batch_size, self.num_steps]))
        
        optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
        
        train_op = optimizer.minimize(loss)
        
        graph = {
            "x": x,
            "y": y,
            "init_hidden_state": init_hidden_state,
            "final_hidden_state": final_hidden_state,
            "loss": loss,
            "prediction": prediction,
            "train_op": train_op
        }
        return(graph)




def train_graph(graph, num_epochs, save_location, x_train, y_train):
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        loss_list = []
        sess.run(init)
        for epoch in range(num_epochs):
            for x_data, y_data in zip(x_train, y_train):
                feed_dict = {graph["x"]: x_data, graph["y"]: y_data}
                current_loss, _ = sess.run([graph["loss"], graph["train_op"]], feed_dict)
                loss_list.append(current_loss)
        saver.save(sess, save_location)
    return(loss_list)




if __name__=="__main__":
	file_url = 'https://raw.githubusercontent.com/jcjohnson/torch-rnn/master/data/tiny-shakespeare.txt'
	file_name = 'tinyshakespeare.txt'

	num_classes, index_to_character, character_to_index, x_train, y_train = generate_char_data(file_url, file_name, batch_size=32, 	num_steps=50)

	char_gru = GRU(state_size=30, num_classes=num_classes, batch_size=32, num_steps=50)
	char_gru_graph = char_gru.build_graph()
	loss_list = train_graph(graph=char_gru_graph, num_epochs=1, save_location="./checkpoints/gru_shakespeare", x_train=x_train, y_train=y_train)

	generate_char_gru = GRU(state_size=30, num_classes=num_classes, batch_size=1, num_steps=1)
	generate_char_gru_graph = generate_char_gru.build_graph()
	generate_characters(graph=generate_char_gru_graph, num_classes=num_classes, load_checkpoint="./checkpoints/gru_shakespeare")





