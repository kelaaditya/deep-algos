import numpy as np
import re

def clean(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = string.lower().strip()
    return(string)


def get_data(positive_examples, negative_examples, batch_size):
    with open(positive_examples) as pos_file:
        pos_lines = pos_file.readlines()
        pos_x = [clean(pos_line.strip()) for pos_line in pos_lines]
    with open(negative_examples) as neg_file:
        neg_lines = neg_file.readlines()
        neg_x = [clean(neg_line.strip()) for neg_line in neg_lines]
    x_data = pos_x + neg_x
    
    data_size = len(x_data)

    pos_y = [[0, 1]] * len(pos_x)
    neg_y = [[1, 0]] * len(neg_x)
    y_data = np.concatenate([pos_y, neg_y])
    
    max_sentence_length = max([len(x.split(" ")) for x in x_data])
    
    words = set([item for x in x_data for item in x.split(" ")])
    index_to_word = dict(enumerate(words))
    word_to_index = dict(zip(index_to_word.values(), index_to_word.keys()))

    # First, create a zero matrix of shape (len(x_data)) x max_sentence_length
    # We pad each sentence with zeros so that all sentences have the same length
    # We then convert each word of each sentence to it's index calculated in the
    # 'word_to_index' dictionary.
    # The padded zeros remain unchanged for each sentence
    x_index = np.zeros(shape=(len(x_data), max_sentence_length), dtype=np.int32)
    for i in range(len(x_index)): 
        word_row = x_data[i].split(" ")
        for j in range(len(word_row)):
            x_index[i][j] = word_to_index[word_row[j]]

    # permute the x_index rows and the y_data rows according to the 
    # list of permutations defined by np.random.permutation
    permutations = np.random.permutation(data_size)
    x_index = x_index[permutations]
    y_data = y_data[permutations]
    
    # mould x_index and y_data into batches
    num_batches = data_size // batch_size
    x_index = x_index[:num_batches * batch_size]
    y_data = y_data[:num_batches * batch_size]
    x_index = np.reshape(x_index, newshape=(num_batches, batch_size, max_sentence_length))
    y_data = np.reshape(y_data, newshape=(num_batches, batch_size, 2))

    return(x_index, y_data)


def generate_epochs(x_data, y_data, num_epochs):
    for epoch in range(num_epochs):
        for batch in range(len(x_data)):
            yield(x_data[batch], y_data[batch])
