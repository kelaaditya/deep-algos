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




def generate_model_graph(state_size, num_rnn_layers, num_classes, batch_size, num_steps, learning_rate=0.001):

    tf.reset_default_graph()

    x = tf.placeholder(dtype=tf.int32, shape=[batch_size, num_steps])
    y = tf.placeholder(dtype=tf.int32, shape=[batch_size, num_steps])

    embeddings = tf.Variable(tf.random_uniform([num_classes, state_size], -1.0, 1.0))
    lookup = tf.nn.embedding_lookup(embeddings, x)
    input_vector = tf.unstack(lookup, axis=1)

    # initialize a Basic RNN Cell with state size number of units
    cell = tf.contrib.rnn.BasicRNNCell(num_units=state_size)

    # initialize a multi RNN Cell with "num_rnn_layers" layers
    stacked_rnn_cell = tf.contrib.rnn.MultiRNNCell([cell] * num_rnn_layers, state_is_tuple=True)

    # the initial state for the multi RNN cell has zero entries
    init_state = stacked_rnn_cell.zero_state(batch_size, dtype=tf.float32)
    state = init_state

    output_list = []
    with tf.variable_scope('RNN', reuse=tf.AUTO_REUSE):
        for input_set in input_vector:
            cell_output, state = stacked_rnn_cell(input_set, state)
            output_list.append(cell_output)
    final_state = state
    output = tf.reshape(tf.concat(output_list, axis=1), [-1, state_size])

    with tf.variable_scope('softmax', reuse=tf.AUTO_REUSE):
        w = tf.get_variable('w', shape=[state_size, num_classes], dtype=tf.float32)
        b = tf.get_variable('b', shape=[num_classes], dtype=tf.float32)
        logits = tf.nn.xw_plus_b(output, w, b)
        logits = tf.reshape(logits, [batch_size, num_steps, num_classes])

        # softmax predictions for text generation
        prediction = tf.nn.softmax(logits)

    # sequence_loss takes logits in the shape of [batch_size, sequence_length, num_decoder_symbols]
    # and targets (here 'y') in the shape of [batch_size, sequence_length]
    # calculates the weighted cross-entropy loss for a sequence of logits
    # Alternative: pick from top 'n' most likely characters
    loss = tf.contrib.seq2seq.sequence_loss(logits, y, weights=tf.ones([batch_size, num_steps]))

    # train with Adam Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)

    train_op = optimizer.minimize(loss)

    graph = {
        "x": x,
        "y": y,
        "init_state": init_state,
        "final_state": final_state,
        "loss": loss,
        "prediction": prediction,
        "train_op": train_op,
    }
    return(graph)




def train_graph(graph, num_epochs, save_location):

    saver = tf.train.Saver()
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        loss_list = []
        
        sess.run(init)
        for epoch in range(num_epochs):
            for x_data, y_data in zip(x_train, y_train):
                feed_dict = {graph["x"]: x_data, graph["y"]: y_data}
                final_state, current_loss, _ = sess.run([graph["final_state"], graph["loss"], graph["train_op"]], feed_dict=feed_dict)
                loss_list.append(current_loss)
        saver.save(sess, save_location)
    return(loss_list, final_state)




def generate_characters(graph, num_classes, load_checkpoint, start_letter='T', num_output_characters=2000):
    saver = tf.train.Saver()

    current_char = character_to_index[start_letter]
    current_state = None
    char_list = [current_char]

    with tf.Session() as sess:
        saver.restore(sess, load_checkpoint)

        # start generating "num_output_characters" number of characters:
        for i in range(num_output_characters):
            # current state starts out as None
            if current_state:
                feed_dict = {graph["x"] : [[current_char]], graph["init_state"] : current_state}
            else:
                feed_dict = {graph["x"] : [[current_char]]}

            prediction, current_state = sess.run([graph["prediction"], graph["final_state"]], feed_dict=feed_dict)

            # We get the probability distribution of the predicted character via the
            # softmax calculation over logits. We use this probability distribution to
            # select the next character by sampling over this distribution
            current_char = np.random.choice(num_classes, 1, p=np.squeeze(prediction))[0]

            char_list.append(current_char)

    characters = map(lambda x: index_to_character[x], char_list)
    string_characters = "".join(characters)
    print(string_characters)

    return(string_characters)





if __name__ == "__main__":

    file_url = 'https://raw.githubusercontent.com/jcjohnson/torch-rnn/master/data/tiny-shakespeare.txt'
    file_name = 'tinyshakespeare.txt'

    num_classes, index_to_character, character_to_index, x_train, y_train = generate_char_data(file_url, file_name, batch_size=32, num_steps=200)

    multi_rnn_graph = generate_model_graph(state_size=100, num_rnn_layers=3, num_classes=num_classes, batch_size=32, num_steps=200, learning_rate=0.0001)

    loss_list, _ = train_graph(multi_rnn_graph, num_epochs=50, save_location="./checkpoints/shakespeare.ckpt")
    #plt.plot(loss_list)
    #plt.show()

    prediction_graph = generate_model_graph(state_size=100, num_rnn_layers=3, num_classes=num_classes, batch_size=1, num_steps=1, learning_rate=0.001)
    predicted_characters = generate_characters(prediction_graph, num_classes, load_checkpoint="./checkpoints/shakespeare.ckpt")

