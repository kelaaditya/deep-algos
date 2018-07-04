import tensorflow as tf
import numpy as np

def get_mnist_data():
    # loads the MNIST dataset
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    return(train_data, train_labels, eval_data, eval_labels)


def cnn_model_fn(features, labels, mode):
    # the input layer reads in from the "features" dictionary
    # the "features" dictionary is specified in the "train_input_fn" 
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    conv1 = tf.layers.conv2d(inputs=input_layer,
                             filters=32,
                             kernel_size=[5, 5],
                             padding="same",
                             activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size=[2, 2],
                                    strides=2)

    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=64,
                             kernel_size=[5, 5],
                             padding="same",
                             activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=[2, 2],
                                    strides=2)

    pool2_to_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_to_flat,
                            units=1034,
                            activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense,
                                rate=0.4,
                                training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout,
                             units=10)

    # the "predictions" dictionary contains the various prediction ops
    # Here, we can opt for predicting either the "classes" or the "probabilities"
    predictions = {
        "classes": tf.argmax(input=logits,
                             axis=1),
        "probabilities": tf.nn.softmax(logits, name='softmax')
    }

    # the PREDICT mode has to be before TRAIN and EVAL
    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    else:
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = {
                "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
            }
            spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    return(spec)


if __name__ == "__main__":
    x_train, y_train, x_eval, y_eval = get_mnist_data()

    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="../checkpoints/mnist")

    # input function for training
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": x_train},
                                                        y=y_train,
                                                        num_epochs=1,
                                                        batch_size=1,
                                                        shuffle=True)
    mnist_classifier.train(input_fn=train_input_fn, steps=20000)

    # input function for evaluation
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": x_eval},
                                                       y=y_eval,
                                                       num_epochs=1,
                                                       batch_size=1,
                                                       shuffle=False)
    evaluation_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(evaluation_results)



