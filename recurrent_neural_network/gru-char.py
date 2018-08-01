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
    #current_cell_state = None
    char_list = [current_char]
       
    with tf.Session() as sess:
        saver.restore(sess, load_checkpoint)
       
        # start generating "num_output_characters" number of characters:
        for i in range(num_output_characters):
            if current_hidden_state is not None:
                feed_dict = {graph['x'] : [[current_char]], graph['init_hidden_state'] : current_hidden_state, graph['init_cell_state'] : current_cell_state}
            else:
                feed_dict = {graph['x'] : [[current_char]]}
                                
            prediction, current_hidden_state, current_cell_state = sess.run([graph['prediction'], graph['final_hidden_state'], graph['final_cell_state']], feed_dict)
                                                                                                                        
            # We get the probability distribution of the predicted character via the
            # softmax calculation over logits. We use this probability distribution to 
            # select the next character by sampling over the distribution
            current_char = np.random.choice(num_classes, 1, p=np.squeeze(prediction))[0]
                                                                                                                                                                                    
            char_list.append(current_char)
                                                                                                                                                                                                        
    characters = map(lambda x: index_to_character[x], char_list)
    string_characters = "".join(characters)
    print(string_characters)
    
    return(string_characters)
