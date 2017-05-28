import tensorflow as tf
import numpy as np
import os
import urllib.request
import scipy.io

VGG_MEAN = [103.939, 116.779, 123.68]


class VGG19:
    def __init__(self, link_to_vgg=None, file_name=None):
        
        if os.path.exists(file_name) and os.stat(file_name).st_size == 534904783:
            print('VGGNet raw_file present')
        else:
            print('Downloading VGGNet Data File')
            urllib.request.urlretrieve(link_to_vgg, file_name)
        
        
        self.layers = ('conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
                       'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
                       'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
                       'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
                       'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5',
                       'full6_1', 'drop6_1',
                       'full7_1', 'drop7_1',
                       'full8_1'
                      )
        
        
    def _conv_layer(self, input_layer, weights, bias):
        conv_layer = tf.nn.conv2d(input_layer, 
                                  filter=tf.constant(weights, name='weights'),
                                  strides=(1, 1, 1, 1),
                                  padding='SAME')
        return(tf.nn.bias_add(conv_layer, bias))
    
    
    def _full_layer(self, input_layer, weights, bias):
        shape = bottom.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        X = tf.reshape(bottom, [-1, dim])
        
        # I chose this to not be trainable
        # Can replace tf.constant by tf.Variable to make trainable
        weights = tf.constant(initial_value=weights, trainable=True)
        bias = tf.constant(initial_value=bias, trainable=True)
        return(tf.nn.bias_add(tf.matmul(X, weights), bias))
        
        
    def _relu_layer(self, input_layer):
        return(tf.nn.relu(input_layer))
    
    
    def _pool_layer(self, input_layer, pool_func='avg'):
        '''
        pool_func has two options:
        - 'avg': Average pooling
        - Else : Max pooling
        '''
        if pool_func == 'avg':
            return(tf.nn.avg_pool(input_layer,
                                  ksize=(1, 2, 2, 1),
                                  strides=(1, 2, 2, 1),
                                  padding='SAME')
                  )
        else:
            return(tf.nn.max_pool(input_layer,
                                  ksize=(1, 2, 2, 1),
                                  strides=(1, 2, 2, 1),
                                  padding='SAME')
                  )
        
    
    def _drop_layer(self, input_layer, keep_prob=0.5):
        intermediate_layer = tf.nn.relu(input_layer) 
        return(tf.nn.dropout(intermediate_layer, keep_prob))

    
    def build(self, input_image, pooling_func='avg', dropout_prob=0.5):
        '''
        Loads VGGNet parameters
            
        INPUT:
        - path         : path to VGGNet
        - input_image  : input image
        - pooling_func : Two options
              'avg': average pooling 
              Else : max pooling
            
        OUTPUT:
        - graph : dictonary that holds the convolutional layers
        '''
            
        vgg = scipy.io.loadmat('./imagenet-vgg-verydeep-19.mat')
        vgg_layers = vgg['layers']
        graph = {}

        # converting RGB to BGR
        input_image = input_image[::-1]
        input_image -= VGG_MEAN
        current_layer = input_image
                   
        for i, layer in enumerate(self.layers):
            if layer[:4] == 'conv' or layer[:4] == 'full':
                weights, bias = vgg_layers[0][i][0][0][2][0]
                bias = bias.reshape(bias.size)
                current_layer = self._conv_layer(current_layer, weights, bias)
            elif layer[:4] == 'relu':
                current_layer = self._relu_layer(current_layer)
            elif layer[:4] == 'pool':
                current_layer = self._pool_layer(current_layer, pooling_func)
            elif layer[:4] == 'drop':
                current_layer = self._drop_layer(current_layer, dropout_prob)
            graph[layer] = current_layer
        return(graph)