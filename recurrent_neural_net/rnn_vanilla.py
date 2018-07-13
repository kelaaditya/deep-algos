import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

num_classes = 2
length_step = 10
length_train = 100000
length_echo = 5
batch_size = 10
state_size = 10
learning_rate = 0.001


    ################################
    #     Low Level TensorFlow     #
    ################################

def rnn_lowlevel(num_classes=2, length_step=10, length_train=100000, length_echo=5, state_size=10, batch_size=10, learning_rate=0.001):

    def generate_data(length_train, length_echo, batch_size, length_step):
        assert length_train % (batch_size * length_step) == 0
    
        x_train = np.random.randint(0, 2, size=length_train)
        y_train = np.roll(x_train, length_echo)
        y_train[0 : length_echo] = 0
    
        x_train = np.reshape(x_train, newshape=(-1, batch_size, length_step))
        y_train = np.reshape(y_train, newshape=(-1, batch_size, length_step))
    
        return(x_train, y_train)
    
    
    def RNN_cell(state, input_vector):
        with tf.variable_scope('RNN_cell', reuse=tf.AUTO_REUSE):
            W = tf.get_variable('W', [num_classes + state_size, state_size], initializer=tf.random_normal_initializer())
            b = tf.get_variable('b', [state_size], initializer=tf.zeros_initializer())
        concatenation = tf.concat([input_vector, state], axis=1)
        nonlinear_output = tf.tanh(tf.add(tf.matmul(concatenation, W), b))
        return(nonlinear_output)
    
    def RNN_cell_outputs(initial_state, list_of_inputs):
        cell_outputs = []
        state = initial_state
        for input_vector in list_of_inputs:
            state = RNN_cell(state, input_vector)
            cell_outputs.append(state)
        return(cell_outputs)
    
    
    def RNN_logits(initial_state, list_of_inputs):
        with tf.variable_scope('softmax', reuse=tf.AUTO_REUSE):
            W = tf.get_variable('W', [state_size, num_classes], initializer=tf.random_normal_initializer())
            b = tf.get_variable('b', [num_classes], initializer=tf.zeros_initializer())
                                        
        cell_outputs = RNN_cell_outputs(initial_state, list_of_inputs)
                                                
        logits = [tf.add(tf.matmul(cell_output, W), b) for cell_output in cell_outputs]
                                                            
        return(logits)
    
    
    def RNN_train(num_epochs):
        x = tf.placeholder(dtype=tf.int32, shape=[batch_size, length_step])
        y = tf.placeholder(dtype=tf.int32, shape=[batch_size, length_step])
        init_state = tf.placeholder(dtype=tf.float32, shape=[batch_size, state_size])
    
        x_one_hot = tf.one_hot(x, depth=2)
        x_one_hot_stack = tf.unstack(x_one_hot, axis=1)
    
        logits = RNN_logits(init_state, x_one_hot_stack)
        
        y_unstack = tf.unstack(y, axis=1)
        
        loss = tf.reduce_mean([tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logit) for logit, label in zip(logits, y_unstack)])
        
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        
        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            sess.run(init)
            loss_list = []
            for epoch in range(num_epochs):
                print("epoch #: ", epoch)
                x_train, y_train = generate_data(length_train, length_echo, batch_size, length_step)
                training_state = np.zeros((batch_size, state_size))
                for x_data, y_data in zip(x_train, y_train):
                    training_loss, _ = sess.run([loss, train_op], feed_dict={x: x_data, y: y_data, init_state: training_state})
                    loss_list.append(training_loss)
        return(loss_list)

    loss_list = RNN_train(1)
    return(loss_list)





    ################################
    #   TensorFlow Estimator API   #
    ################################

def rnn_estimator(num_classes=2, length_step=25, length_train=100000, length_echo=3, state_size=100, batch_size=10, learning_rate=0.001):

    def generate_data(num_classes, length_train, length_echo, length_step):
        assert length_train % length_step == 0
    
        x_train = np.random.randint(0, num_classes, size=length_train)
        y_train = np.roll(x_train, length_echo)
        y_train[0 : length_echo] = 0
    
        x_train = np.reshape(x_train, newshape=(-1, length_step))
        y_train = np.reshape(y_train, newshape=(-1, length_step))
    
        return(x_train, y_train)
    
    
    def rnn_model_fn(features, labels, mode, params):
        num_classes = params["num_classes"]
        length_step = params["length_step"]
        state_size = params["state_size"]
        batch_size = params["batch_size"]
        
        #init_state = tf.placeholder(dtype=tf.float32, shape=[None, state_size])
        init_state = tf.random_normal(shape=[batch_size, state_size], dtype=tf.float32)
        
        # features["x"] here sends in input of size "batch_size x length_step"
        # the input_fn fed to the estimator takes care of the batching
        input_vector_one_hot = tf.one_hot(features["x"], depth=num_classes)
        input_vector_one_hot_stack = tf.unstack(input_vector_one_hot, axis=1)
        
        #
        def RNN_cell(state, input_vector):
            with tf.variable_scope('RNN_cell', reuse=tf.AUTO_REUSE):
                W = tf.get_variable('W', [num_classes + state_size, state_size], initializer=tf.random_normal_initializer())
                b = tf.get_variable('b', [state_size], initializer=tf.zeros_initializer())
            concatenation = tf.concat([input_vector, state], axis=1)
            nonlinear_output = tf.tanh(tf.add(tf.matmul(concatenation, W), b))
            return(nonlinear_output)
    
        #
        def RNN_cell_outputs(initial_state, list_of_inputs):
            cell_outputs = []
            state = initial_state
            for input_vector in list_of_inputs:
                state = RNN_cell(state, input_vector)
                cell_outputs.append(state)
            return(cell_outputs)
        
        #
        def RNN_logits(initial_state, list_of_inputs):
            with tf.variable_scope('softmax', reuse=tf.AUTO_REUSE):
                W = tf.get_variable('W', [state_size, num_classes], initializer=tf.random_normal_initializer())
                b = tf.get_variable('b', [num_classes], initializer=tf.zeros_initializer())
            cell_outputs = RNN_cell_outputs(initial_state, list_of_inputs)
            logits = [tf.add(tf.matmul(cell_output, W), b) for cell_output in cell_outputs]
            return(logits)
        
        logits = RNN_logits(init_state, input_vector_one_hot_stack)
        
        predictions = {
            "classes": tf.convert_to_tensor([tf.argmax(input=logit, axis=1) for logit in logits]),
            "probabilities": tf.convert_to_tensor([tf.nn.softmax(logit) for logit in logits])
        }
        
        if mode == tf.estimator.ModeKeys.PREDICT:
            spec = tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
        else:
            labels_unstack = tf.unstack(labels, axis=1)
            loss = tf.reduce_mean([tf.losses.sparse_softmax_cross_entropy(labels=label, logits=logit) for logit, label in zip(logits, labels_unstack)])
            
            if mode == tf.estimator.ModeKeys.TRAIN:
                learning_rate = params["learning_rate"]
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
                spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
            if mode == tf.estimator.ModeKeys.EVAL:
                eval_metric_ops = {
                    "accuracy": tf.metrics.accuracy(labels=labels_unstack[-1], predictions=predictions["classes"][-1])
                }
                spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
        return(spec)
    
    rnn_sequencer = tf.estimator.Estimator(model_fn=rnn_model_fn,
                                           params={
                                               "num_classes": num_classes,
                                               "length_step": length_step,
                                               "state_size": state_size,
                                               "learning_rate": learning_rate,
                                               "batch_size": batch_size
                                           },
                                           model_dir="../checkpoints/rnn_echo")
    
    x_train, y_train = generate_data(num_classes, length_train, length_echo, length_step)
    
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": x_train},
                                                        y=y_train,
                                                        num_epochs=1,
                                                        batch_size=batch_size,
                                                        shuffle=False)
    
    rnn_sequencer.train(input_fn=train_input_fn, steps=200000)
    
    x_eval, y_eval = generate_data(num_classes, batch_size * length_step, length_echo, length_step)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": x_eval},
                                                        y=y_eval,
                                                        num_epochs=1,
                                                        batch_size=batch_size,
                                                        shuffle=False)
    
    evaluation_results = rnn_sequencer.evaluate(input_fn=eval_input_fn)
    
    
    x_pred, y_pred = generate_data(num_classes, batch_size * length_step, length_echo, length_step)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": x_pred},
                                                          num_epochs=1,
                                                          batch_size=batch_size,
                                                          shuffle=False)
    
    prediction_results = rnn_sequencer.predict(input_fn=predict_input_fn)
    
    prediction_list = []
    for _, p in enumerate(prediction_results):
        prediction_list.append(p["classes"])
    prediction_list = np.asarray(prediction_list)
    prediction_list = prediction_list.T

    return(y_pred, prediction_list)


if __name__=="__main__":
    label_list, predicted_label_list = rnn_estimator()
    print("label list: ", label_list)
    print("predicted label list: ", predicted_label_list)

    loss_list = rnn_lowlevel()
    plt.plot(loss_list)
    plt.show()
