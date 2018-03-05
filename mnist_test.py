import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_common
import mnist_train


def test(mnist):
    with tf.Graph().as_default() as graph:           # notice Graph()
        xx = tf.placeholder(tf.float32, [None, mnist_common.INPUT_DIM], name="sample")
        yy = tf.placeholder(tf.float32, [None, mnist_common.OUTPUT_DIM], name="lable")

        y = mnist_common.inference(xx, None)

        prediction = tf.argmax(y, 1)
        reference = tf.argmax(yy, 1)

        accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, reference), tf.float32))

        variables_averager = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variables_restore = variables_averager.variables_to_restore()

        saver = tf.train.Saver(variables_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                # train_step = ckpt.model_checkpoint_path.split('/')[-1].split
                score = sess.run(accuracy, feed_dict={xx: mnist.test.images, yy: mnist.test.labels})
                print("accuracy = %g" % score)






def main(argv=None):
    mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
    test(mnist)



if __name__ == "__main__":
    tf.app.run()



