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
