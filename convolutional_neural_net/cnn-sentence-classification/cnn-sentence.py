import tensorflow as tf
import numpy as np

class CNN:
    def __init__(self, length_sentence, vocab_size, num_classes, embedding_size, num_filters, filter_sizes):
        
        self.length_sentence = length_sentence
        # vocab size is the total number of unique words from the dataset
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        
        # num_filters for each filter size in list:filter_size
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        
    
    def build_graph(self, mode="training"):
        x = tf.placeholder(dtype=tf.int32, shape=[None, self.length_sentence])
        y = tf.placeholder(dtype=tf.int32, shape=[None, self.num_classes])
        
        with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
            # embedding matrix of size vocab_size x state_size
            # used to get the lookups for each sentence in batch
            embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0))
            # shape of lookup: (batch_size, length_sentence, state_size)
            lookup = tf.nn.embedding_lookup(embeddings, x)
            lookup_with_dummy_channel = tf.expand_dims(lookup, axis=-1)
            
            
        max_pool_layers = []
        for filter_size in self.filter_sizes:
            conv_layer = tf.layers.conv2d(inputs=lookup_with_dummy_channel,
                                      filters=self.num_filters,
                                      kernel_size=[filter_size, self.embedding_size],
                                      padding="valid",
                                      activation=tf.nn.relu)
            
            # for the pool layer, we calculate the max_pooling for the entire convolution output
            # We need to have one maximum as the output for each filter:
            # Eg: for batch_size=10, filter_sizes=[10, 10], length_sentence=65, we get an output of 
            # shape (10, 47, 1, 32). We want to find the maximum output from each filter -- therefore, 
            # we max_pool with a filter size of (47, 1) to get back a tensor of shape (10, 1, 1, 32)
            # this new tensor holds the maximum of all 47 entries along each filter
            pool_layer = tf.layers.max_pooling2d(inputs=conv_layer,
                                                 pool_size=[self.length_sentence - filter_size + 1, 1],
                                                 strides=1)
            max_pool_layers.append(pool_layer)
        max_pool_layer = tf.squeeze(tf.concat(max_pool_layers, axis=-1))
        
        dropout = tf.layers.dropout(inputs=max_pool_layer,
                                    rate=0.1,
                                    training=True)
        
        with tf.variable_scope('logits', reuse=tf.AUTO_REUSE):
            W = tf.get_variable('W', shape=[len(self.filter_sizes) * self.num_filters, self.num_classes], initializer=tf.random_normal_initializer())
            b = tf.get_variable('b', shape=[self.num_classes], initializer=tf.zeros_initializer())
            loss_l2 = tf.nn.l2_loss(W)
            loss_l2 += tf.nn.l2_loss(b)
            logits = tf.nn.xw_plus_b(dropout, W, b)
            
        predictions = {
            "probabilities": tf.nn.softmax(logits, name='softmax'),
            "classes": tf.argmax(input=logits, axis=1)
        }
            
        loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits) + loss_l2
            
            
        graph = {
            "x": x,
            "y": y,
            "embeddings": embeddings, 
            "prediction_dict": predictions,
            "loss": loss,
        }
        return(graph)