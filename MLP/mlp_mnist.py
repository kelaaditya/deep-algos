import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


    ###########################
    #         batchify        #
    ###########################

def batchify(X, y, batch_size):
    for i in range(0, len(X)-batch_size+1, batch_size):
        yield(X[i: i+batch_size], y[i: i+batch_size])



    ###########################
    #       load MNIST        #
    ###########################

def get_mnist_data():
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    return(train_data, train_labels, eval_data, eval_labels)



    ###########################
    #      MLP for MNIST      #
    ###########################

def mlp(batch_size, epochs, learning_rate=0.01):
    X = tf.placeholder(tf.float32, shape=(batch_size, 784), name='X')
    Y = tf.placeholder(tf.float32, shape=(batch_size, 10), name='Y')
    
    #Hidden Layer 1 has 800 hidden units
    hidden_layer_1_weights = tf.Variable(tf.random_normal([784, 800]), name='w_1')
    hidden_layer_1_bias = tf.Variable(tf.random_normal([800]), name='b_1')
    
    #Hidden Layer 2 has 800 hidden units
    hidden_layer_2_weights = tf.Variable(tf.random_normal([800, 800]), name='w_2')
    hidden_layer_2_bias = tf.Variable(tf.random_normal([800]), name='b_2')
    
    #Output Layer has 800 hidden units and 10 output classes
    output_layer_weights = tf.Variable(tf.random_normal([800, 10]), name='w_o')
    output_layer_bias = tf.Variable(tf.random_normal([10]), name='b_o')
    
    #Hidden Layer activated with a RELU
    hidden_layer_1 = tf.add(tf.matmul(X, hidden_layer_1_weights), hidden_layer_1_bias)
    hidden_layer_1 = tf.nn.relu(hidden_layer_1)
    
    #Hidden Layer 2 activated with a RELU
    hidden_layer_2 = tf.add(tf.matmul(hidden_layer_1, hidden_layer_2_weights), hidden_layer_2_bias)
    hidden_layer_2 = tf.nn.relu(hidden_layer_2)
    
    output_layer = tf.add(tf.matmul(hidden_layer_2, output_layer_weights), output_layer_bias)
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        number_batches = int(mnist.train.num_examples/batch_size)
        training_loss_list = []
        for epoch in range(epochs):
            for i in range(number_batches):
                X_batch, Y_batch = mnist.train.next_batch(batch_size)
                _, training_loss_value = sess.run([optimizer, loss], feed_dict={X: X_batch, Y: Y_batch})
                training_loss_list.append(training_loss_value)

    return(training_loss_list)



    ###########################
    #   MLP (HighLevelAPIs)   #
    ###########################

def mlp_model_fn(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 784])
    
    hidden1 = tf.layers.dense(inputs=input_layer, units=1024, activation=tf.nn.relu)
    
    hidden2 = tf.layers.dense(inputs=hidden1, units=1024, activation=tf.nn.relu)
    
    logits = tf.layers.dense(inputs=hidden2, units=10)
    
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax")
    }
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    else:
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
            spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = {
                "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
            }
            spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    return(spec)



if __name__ == "__main__":
    x_train, y_train, x_eval, y_eval = get_mnist_data()

    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": x_train},
                                                    y=y_train,
                                                    num_epochs=2,
                                                    batch_size=100,
                                                    shuffle=True)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": x_eval},
                                                   y=y_eval,
                                                   num_epochs=1,
                                                   batch_size=1,
                                                   shuffle=False)
    
    mnist_classifier = tf.estimator.Estimator(model_fn=mlp_model_fn, model_dir="./checkpoints/mnist_mlp")
    mnist_classifier.train(input_fn=train_input_fn, steps=20000)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
