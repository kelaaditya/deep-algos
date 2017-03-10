import tensorflow as tf
import numpy as np


def word2vec(input_data, classified_data, vocab_size, embedding_size, batch_size, num_sampled):
    '''word2vec skip-gram using NCE loss
    
    params:
        input_data: input word data
        classified_data: target values for the input data
        vocab_size (int): vocabulary size
        embedding_size (int): dimension of the word embedding vectors
        batch_size (int): batch size
        num_sampled (int): number of negative examples sampled
    '''
    
    #Using namescopes for better graph visualization
    with tf.name_scope('data'):
        #Not using the one-hot representation
        #Index of the word fed in directly rather than the one-hot representation
        input_words = tf.placeholder(tf.int32, shape=[batch_size], name='input_words')
        label_words = tf.placeholder(tf.int32, shape=[batch_size, 1], name='label_words')
    
    with tf.name_scope('embedding'):
        #Randomly initialized embedding matrix
        embedding_matrix = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), 
                                   name='embedding_matrix')
        
        #lookup for the embedding matrix
        embedding = tf.nn.embedding_lookup(embedding_matrix, input_words, name='embedding')
    
    with tf.name_scope('hidden_layer'):
        hidden_layer_weights = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                                           name='hidden_layer_weight')
        
        hidden_layer_bais = tf.Variable(tf.random_uniform(tf.zeros([vocab_size])),
                                        name = 'hidden_layer_bias')
        
    with tf.name_scope('loss'):
        #Use the NCE loss function
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=hidden_layer_weights,
                                             biases=hidden_layer_bais,
                                             labels=label_words,
                                             inputs=embedding,
                                             num_sampled=num_sampled,
                                             num_classes=vocab_size),
                              name='loss')
        
        #Gradient Descent Optimizer with learning rate=1.0
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)
        
        init = tf.global_variables_initializer()
        
    with tf.Session() as sess:
        sess.run(init)

        writer = tf.summary.FileWriter('graphs/', sess.graph)
        
        loss_list = []
        
        for epoch in range(epochs):
            for batch in batchify(input_data, classified_data, batch_size):
                input_batch, label_batch = batch
                loss_value, _ = sess.run([loss, optimizer], 
                                         feed_dict={input_words: input_batch, label_words: label_batch})
                loss_list.append(loss_value)
        
        writer.close()


if __name__ == '__main__':
    input_data, classified_data = load_data()
    word2vec(input_data, classified_data, vocab_size, embedding_size, batch_size, num_sampled)
