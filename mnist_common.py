import tensorflow as tf


INPUT_DIM = 784
LAYER_DIM = 100
OUTPUT_DIM = 10


def get_weight(shape, regularizer):
    weights = tf.get_variable(
        "weights", shape,
        initializer=tf.truncated_normal_initializer(stddev=0.1)

    )

    if regularizer is not None:
        tf.add_to_collection('loss', regularizer(weights))

    return weights


def inference(input, regularizer):
    with tf.variable_scope('layer1'):
        weights = get_weight([INPUT_DIM, LAYER_DIM], regularizer)
        biases = tf.get_variable(
            "biases", [LAYER_DIM]
        )
        layer1 = tf.nn.relu(tf.matmul(input, weights) + biases)
    with tf.variable_scope('layer2'):
        weights = get_weight([LAYER_DIM, OUTPUT_DIM], regularizer)
        biases = tf.get_variable(
            "biases", [OUTPUT_DIM],
            initializer=tf.constant_initializer(0.0)
        )
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2









