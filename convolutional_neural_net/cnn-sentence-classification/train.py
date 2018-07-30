import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import re

def train(positive_examples, 
          negative_examples,
          batch_size,
          num_epochs,
          embedding_size,
          num_filters,
          filter_sizes,
          learning_rate,
          save_location):
        
    tf.reset_default_graph()
            
    vocab_size, max_sentence_length, x_data, y_data = get_data(positive_examples, negative_examples, batch_size)
                    
    cnn_sentence = CNN(max_sentence_length, vocab_size, 2, embedding_size, num_filters, filter_sizes)
    cnn_sentence_graph = cnn_sentence.build_graph(mode="training")
                                
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(cnn_sentence_graph["loss"])
                                        
    init = tf.global_variables_initializer()
                                                    
    saver = tf.train.Saver()
                                                            
    loss_list = []
    with tf.Session() as sess:
        sess.run(init)
        for x_batch, y_batch in generate_epochs(x_data, y_data, num_epochs):
            feed_dict = {cnn_sentence_graph["x"]: x_batch, cnn_sentence_graph["y"]: y_batch}
            loss = sess.run(cnn_sentence_graph["loss"], feed_dict)
            loss_list.append(loss)
        
        saver.save(sess, save_location)
    return(loss_list)


if __name__ == "__main__":
    batch_size = 32
    num_epochs = 10
    embedding_size = 100
    num_filters = 32
    filter_sizes = [10, 20, 30]
    learning_rate = 0.001
    save_location = './checkpoints/cnn-sentence'

    loss_list = train(positive_examples,
                      negative_examples,
                      batch_size,
                      num_epochs,
                      embedding_size,
                      num_filters, 
                      filter_sizes,
                      learning_rate,
                      save_location)

    plt.plot(loss_list)
    plt.show()

