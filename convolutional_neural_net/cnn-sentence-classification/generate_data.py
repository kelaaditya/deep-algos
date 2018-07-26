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


def get_data(positive_examples, negative_examples, batch_size, num_epoch):
    with open(positive_examples) as pos_file:
        pos_lines = pos_file.readlines()
        pos_x = [clean(pos_line.strip()) for pos_line in pos_lines]
    with open(negative_examples) as neg_file:
        neg_lines = neg_file.readlines()
        neg_x = [clean(neg_line.strip()) for neg_line in neg_lines]
    x_data = np.array(pos_data + neg_data)
                                                            
    data_size = len(x_data)

    pos_y = [[0, 1]] * len(pos_x)
    neg_y = [[1, 0]] * len(neg_x)
    y_data = np.concatenate([pos_y, neg_y])
                                                                            
    permutations = np.random.permutation(data_size)
    x_data = x_data[permutations]
    y_data = y_data[permutations]
                                                                                                
    num_batches = data_size // batch_size
    x_data = x_data[:num_batches * batch_size]
    y_data = y_data[:num_batches * batch_size]
    x_data = np.reshape(x_data, newshape=(num_batches, batch_size))
    y_data = np.reshape(y_data, newshape=(num_batches, batch_size, 2))
                    
    for epoch in range(num_epoch):
        for batch in range(num_batches):
            yield(x_data[batch], y_data[batch])
