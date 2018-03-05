import os
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

import mnist_common

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 80000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = "./"
MODEL_NAME = "mnist_model.ckpt"



def train(mnist):
# with tf.device('/cpu:0'):
    xx = tf.placeholder(
        tf.float32, [None, mnist_common.INPUT_DIM], name="sample"
    )
    yy = tf.placeholder(
        tf.float32, [None, mnist_common.OUTPUT_DIM], name="label"
    )

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    y = mnist_common.inference(xx, regularizer)

    train_step = tf.Variable(0, trainable=False)

    variables_averager = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, train_step)
    variable_average = variables_averager.apply(tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(yy, 1))

    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean + tf.add_n(tf.get_collection("loss"))

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        train_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY
    )

    train_process = tf.train\
                    .GradientDescentOptimizer(learning_rate)\
                    .minimize(loss, global_step=train_step)

    with tf.control_dependencies([train_process, variable_average]):
        train_op = tf.no_op(name="train")


    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        tf.initialize_all_variables().run()

        for ii in range(TRAINING_STEPS):
            xxs, yys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_val, step = sess.run([train_op, loss, train_step], feed_dict={xx:xxs, yy:yys})

            if ii % 1000 == 0:
                print("Training step %d, loss is %g" % (step, loss_val))
#       saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=train_step)
        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))

def main(argv=None):
    mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
    train(mnist)


if __name__ == "__main__":
    tf.app.run()

































